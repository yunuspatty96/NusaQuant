"""
NusaQuant — core library
========================

Everything the training script and the dashboard both need: the Sectors API v2
client, the on-disk cache, point-in-time feature engineering, forward-return
targets, historical risk and the explanation templates.

Three rules shape this file.

1. **Credits are spent once.** Every fetch goes through the cache, and a meter
   charges each call before it is sent. A crashed run resumes; it never
   re-buys.
2. **No look-ahead.** A financial statement dated 31 March was not public on
   31 March, so fundamentals are held back by a reporting lag before they may
   be used as a feature.
3. **No API key in code.** The key arrives from the environment or a terminal
   argument, is passed only in a request header, and is never logged, printed,
   cached or embedded in an error.

NusaQuant is a research tool. Probabilities are model estimates, not
guarantees and not financial advice. There is no LLM anywhere in it — every
sentence the user reads comes from a template below.
"""

from __future__ import annotations

import datetime as dt
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import requests

__version__ = "0.3.0"

API_BASE_URL = "https://api.sectors.app/v2"
MAX_DAILY_WINDOW_DAYS = 90          # hard limit of the daily endpoint
REPORTING_LAG_DAYS = 90             # conservative point-in-time delay
HORIZON_TRADING_DAYS = {"6m": 126, "12m": 252}
RANDOM_STATE = 42

TRADING_DAYS_PER_YEAR = 252
MIN_QUARTERS_FOR_PREDICTION = 8     # 4 for TTM, 4 more for one-year growth
MIN_DATA_COMPLETENESS = 0.70
MAX_FEATURE_MISSINGNESS = 0.30

# A ratio earns its place by ranking stocks, not by existing. MIN_FEATURE_IC is
# the information coefficient a ratio must reach to be fed to the model: the
# average within-quarter Spearman correlation between the ratio and the return
# that followed.
#
# The bar is set from the panel's own noise floor, not from the literature. Feed
# this measurement a column of random numbers and it does not return zero: on a
# cross-section this narrow it returns |IC| of about 0.05 on average, with a
# 90th percentile near 0.11 and a 95th near 0.13 (400 permutations, both
# horizons). An earlier draft of this constant sat at 0.02 and screened nothing
# whatsoever, because a random column clears 0.02 nearly every time.
#
# 0.06 is therefore the honest floor and not a significance test. It asks only
# that a ratio beat what a random column typically manages — the median of that
# permutation run, on the narrowest panel the tests exercise, is 0.051. Raising it to the
# 95th percentile would be the stronger claim, and on the current panel it is
# not one the data supports: of six ratios only NPM and ROE clear 0.13 at six
# months, and that is before any correction for having looked at six of them.
# The screen removes the ratios that are visibly worse than noise; it does not
# certify the survivors as signal, and nothing downstream treats them as such.
MIN_FEATURE_IC = 0.06
MIN_SCREENED_FEATURES = 3
MIN_CROSS_SECTION = 8               # stocks needed before a quarter can be ranked

# A model has to out-discriminate a coin flip by a visible margin before its
# probabilities are allowed to be described as a signal. 0.55 is deliberately
# modest; on a panel this small anything below it is inside the noise.
MIN_EDGE_AUC = 0.55

RELIABILITY_WEIGHTS = {"roc_auc": 0.40, "pr_auc": 0.25, "calibration": 0.20, "stability": 0.15}
RISK_WEIGHTS = {"volatility": 0.40, "max_drawdown": 0.35, "downside_volatility": 0.15, "liquidity": 0.10}


# ══════════════════════════════════════════════════════════════════════
# API KEY
# ══════════════════════════════════════════════════════════════════════

def get_api_key(explicit: str | None = None) -> str:
    """Resolve the Sectors API key without it ever appearing in source.

    Order: explicit argument (from a CLI flag), then ``SECTORS_API_KEY`` in the
    environment. Nothing else. There is deliberately no file default and no
    hard-coded fallback.
    """
    key = (explicit or os.environ.get("SECTORS_API_KEY") or "").strip()
    if not key:
        raise RuntimeError(
            "No Sectors API key found.\n\n"
            "Set it in your terminal before running:\n"
            "  export SECTORS_API_KEY=your-key-here      (macOS / Linux)\n"
            "  set SECTORS_API_KEY=your-key-here         (Windows CMD)\n"
            "  $env:SECTORS_API_KEY='your-key-here'      (PowerShell)"
        )
    return key


def get_api_headers(api_key: str) -> dict[str, str]:
    return {"Authorization": api_key, "Accept": "application/json"}


# ══════════════════════════════════════════════════════════════════════
# CREDIT METER
# ══════════════════════════════════════════════════════════════════════

class CreditLimitExceeded(RuntimeError):
    """The configured credit budget was reached, so the run stopped."""


class CreditMeter:
    """Counts credits and refuses to spend past a budget.

    Charged *before* each request, so an overrun halts the run instead of
    being discovered afterwards with an empty account.
    """

    def __init__(self, budget: int | None = None):
        self.budget = budget
        self.spent = 0
        self.by_endpoint: dict[str, int] = {}

    def charge(self, endpoint: str, credits: int = 1) -> None:
        if self.budget is not None and self.spent + credits > self.budget:
            raise CreditLimitExceeded(
                f"Stopping before this call: it would take the total to "
                f"{self.spent + credits} credits but the budget is {self.budget}. "
                f"Everything fetched so far is saved — re-run to resume."
            )
        self.spent += credits
        self.by_endpoint[endpoint] = self.by_endpoint.get(endpoint, 0) + credits

    def refund(self, endpoint: str, credits: int = 1) -> None:
        """Give back a charge Sectors did not actually bill.

        The meter charges *before* a request is sent, because discovering an
        overrun afterwards is useless. But Sectors bills on the response, and
        400, 401/403, 429 and 5xx are free. Without a refund the meter drifts
        upward on every failure and eventually halts a run that still had
        budget — and the "credits spent" line the run prints at the end
        overstates what the account was really charged.
        """
        self.spent = max(0, self.spent - credits)
        if endpoint in self.by_endpoint:
            self.by_endpoint[endpoint] = max(0, self.by_endpoint[endpoint] - credits)

    @property
    def remaining(self) -> int | None:
        return None if self.budget is None else self.budget - self.spent

    def report(self) -> str:
        head = f"Credits spent: {self.spent:,}"
        if self.budget is not None:
            head += f" of {self.budget:,} (remaining {self.remaining:,})"
        rows = [f"  {k:<24}{v:>6,}" for k, v in
                sorted(self.by_endpoint.items(), key=lambda kv: -kv[1])]
        return "\n".join([head, *rows])


_METER: CreditMeter | None = None


def set_credit_meter(meter: CreditMeter | None) -> None:
    global _METER
    _METER = meter


def get_credit_meter() -> CreditMeter | None:
    return _METER


def _price_of(path: str, params: dict[str, Any]) -> tuple[str, int]:
    """Sectors' published pricing, so the meter charges the real amount."""
    if path.startswith("financials/quarterly"):
        return "financials/quarterly", int(params.get("n_quarters") or 1)
    if path.startswith("company/report"):
        sections = str(params.get("sections") or "overview")
        return "company/report", len([s for s in sections.split(",") if s])
    if path.startswith("daily/"):
        return "daily", 1
    if path.startswith("companies/"):
        return "companies", 3 if params.get("q") else 1
    return path.split("/")[0] or "other", 1


# ══════════════════════════════════════════════════════════════════════
# API CLIENT
# ══════════════════════════════════════════════════════════════════════

class SectorsAPIError(RuntimeError):
    """A Sectors call failed. Carries a user-safe message, never the key."""

    def __init__(self, message: str, *, status: int | None = None, detail: str = ""):
        super().__init__(message)
        self.message, self.status, self.detail = message, status, detail


_STATUS = {
    400: ("Sectors rejected the request (bad parameter or date range).", False),
    401: ("Invalid Sectors API key.", False),
    403: ("Your Sectors plan does not include this data.", False),
    404: ("That ticker or endpoint was not found.", False),
    429: ("Sectors rate or credit limit reached.", True),
    500: ("Sectors server error. Try again shortly.", True),
    502: ("Sectors server error. Try again shortly.", True),
    503: ("Sectors is temporarily unavailable.", True),
    504: ("Sectors took too long to respond.", True),
}

API_HELP = (
    "Unable to retrieve Sectors data. Please check:\n"
    "1. API key   2. Plan access   3. Internet   4. Credit or rate limit"
)


def api_request(path: str, api_key: str, params: dict[str, Any] | None = None,
                *, timeout: int = 30, max_retries: int = 5) -> Any:
    """One GET against Sectors v2. Retries 429 and 5xx, never leaks the key.

    A 429 is backed off much harder than a server error. Sectors returns 429
    for a per-second rate limit as well as for an exhausted quota, and the two
    look identical from here — but they want opposite things. Waiting one
    second and trying twice, which is what this used to do, is far too
    impatient for the first and pointless for the second. One collection run
    lost nine tickers to 429s that were interleaved with successes, which is
    the signature of a rate limit rather than an empty account: an empty
    account fails everything after the first failure, and this did not.
    """
    url = f"{API_BASE_URL}/{path.lstrip('/')}"
    clean = {k: v for k, v in (params or {}).items() if v is not None}

    endpoint, price = _price_of(path, clean)
    if _METER is not None:
        _METER.charge(endpoint, price)

    def unbilled():
        """Sectors charged nothing, so neither should the meter."""
        if _METER is not None:
            _METER.refund(endpoint, price)

    detail = ""
    for attempt in range(max_retries):
        try:
            response = requests.get(url, headers=get_api_headers(api_key),
                                    params=clean, timeout=timeout)
        except requests.Timeout:
            detail = f"Timeout after {timeout}s"
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt); continue
            unbilled()
            raise SectorsAPIError("The request to Sectors timed out.", detail=detail) from None
        except requests.RequestException as exc:
            unbilled()
            raise SectorsAPIError(
                "Could not reach Sectors. Check your internet connection.",
                detail=f"{type(exc).__name__}: {exc}") from None

        if response.status_code == 200:
            try:
                return response.json()
            except ValueError:
                raise SectorsAPIError("Sectors returned an unreadable response.",
                                      status=200, detail=response.text[:400]) from None

        message, retryable = _STATUS.get(
            response.status_code,
            (f"Sectors returned status {response.status_code}.", False))
        detail = f"HTTP {response.status_code} :: {response.text[:300]}"
        if retryable and attempt < max_retries - 1:
            wait = response.headers.get("Retry-After")
            if wait and wait.isdigit():
                time.sleep(float(wait))
            else:
                # 3s, 6s, 12s, 24s for a rate limit; 1s, 2s, 4s, 8s otherwise.
                base = 3 if response.status_code == 429 else 1
                time.sleep(base * 2 ** attempt)
            continue
        # Sectors bills on the response: 2xx costs the endpoint's stated price
        # and 404 costs 1, but 400, 401/403, 429 and 5xx are free.
        if response.status_code not in (200, 404):
            unbilled()
        raise SectorsAPIError(message, status=response.status_code, detail=detail)

    unbilled()
    raise SectorsAPIError(API_HELP, detail=detail)


def normalise_symbol(symbol: str) -> str:
    """``'bbca.jk'`` -> ``'BBCA'``."""
    return str(symbol).strip().upper().removesuffix(".JK")


def validate_api_key(api_key: str) -> bool:
    """Cheapest possible connectivity check (1 credit)."""
    if not api_key or not api_key.strip():
        return False
    try:
        api_request("subsectors/", api_key)
        return True
    except SectorsAPIError:
        return False


def get_companies(api_key: str, *, where: str | None = None,
                  order_by: str = "-market_cap", limit: int = 50) -> pd.DataFrame:
    """The IDX universe from the screener: symbol and name, 1 credit."""
    payload = api_request("companies/", api_key,
                          {"where": where, "order_by": order_by, "limit": min(limit, 200)})
    rows = payload.get("results", []) if isinstance(payload, dict) else []
    if not rows:
        return pd.DataFrame(columns=["symbol", "company_name"])
    frame = pd.DataFrame(rows)
    frame["symbol"] = frame["symbol"].map(normalise_symbol)
    return frame[["symbol", "company_name"]].drop_duplicates("symbol").reset_index(drop=True)


#: The classification columns a screen can attach to each company.
CLASSIFICATION_FIELDS = ("sector", "sub_sector", "industry")

#: Dividend fields the screener carries, mapped to the metric names used here.
#: These are TRAILING values as of the screen, not point-in-time history, so
#: they are shown and never modelled — see FeatureSpec.point_in_time.
DIVIDEND_FIELDS: dict[str, str] = {
    "dividend_ttm": "dividend",
    "payout_ratio": "dpr",
    "yield_ttm": "dividend_yield",
}


def _names_field(field: str, *, keep_nulls: bool = False) -> str:
    """A condition that makes the screener return a field.

    ``query_values`` carries whatever the query mentions, so a field is
    requested by putting it in the where clause. ``IS NOT NULL`` does that
    without narrowing anything for a field every company has — but for
    dividends it would silently drop every company that does not pay one, so
    those are named with a condition that is true for every row instead.
    """
    return f"({field} IS NOT NULL or {field} IS NULL)" if keep_nulls \
        else f"{field} IS NOT NULL"


def get_company_classification(api_key: str, *, where: str | None = None,
                               order_by: str = "-market_cap", limit: int = 200,
                               dividends: bool = False,
                               raw: list | None = None) -> pd.DataFrame:
    """Symbol, name and IDX classification for the whole universe. 1 credit.

    The screener returns only symbol and company_name by default. Everything
    else arrives through ``query_values``, which the endpoint fills in with the
    fields the query referenced — so the classification columns are requested
    by naming them in the ``where`` clause.

    ``IS NOT NULL`` is used deliberately rather than an equality test: it names
    the field, which is what makes it come back, without narrowing the universe
    the way a real filter would. Every listed company has a sector, so the
    condition costs nothing in coverage.

    Pass ``raw`` to receive the untouched first result alongside the frame; the
    call costs a credit either way and the payload is worth seeing when the
    parse comes back thinner than expected.
    """
    conditions = ([where] if where else []) \
        + [_names_field(f) for f in CLASSIFICATION_FIELDS]
    if dividends:
        conditions += [_names_field(f, keep_nulls=True) for f in DIVIDEND_FIELDS]
    payload = api_request("companies/", api_key, {
        "where": " and ".join(conditions),
        "order_by": order_by,
        "limit": min(limit, 200),
        "include_query_values": "true"})

    rows = payload.get("results", []) if isinstance(payload, dict) else []
    if raw is not None and rows:
        raw.append(rows[0])
    if not rows:
        return pd.DataFrame(columns=["symbol", "company_name", *CLASSIFICATION_FIELDS])

    records = []
    for row in rows:
        values = row.get("query_values") or {}
        record = {"symbol": normalise_symbol(row.get("symbol", "")),
                  "company_name": row.get("company_name")}
        for field in CLASSIFICATION_FIELDS:
            # The endpoint has been seen to key these either at the top level
            # or inside query_values, so both are accepted.
            record[field] = values.get(field, row.get(field))
        if dividends:
            for source, target in DIVIDEND_FIELDS.items():
                record[target] = _to_float(values.get(source, row.get(source)))
        records.append(record)
    frame = pd.DataFrame(records)
    return frame.drop_duplicates("symbol").reset_index(drop=True)


#: Ratios the screener can attach to a peer screen, and the NusaQuant metric
#: each one corresponds to. Sectors computes these itself, so they are close
#: cousins of the locally computed ratios rather than the same number: the
#: dashboard says which source it is showing rather than blending the two.
SCREENER_RATIOS: dict[str, str] = {
    "pe_ttm": "pe", "ps_ttm": "ps", "pb_mrq": "pbv",
    "der_mrq": "der", "roa_ttm": "roa", "roe_ttm": "roe",
}


def get_sector_peers(api_key: str, *, level: str, name: str,
                     limit: int = 200) -> pd.DataFrame:
    """Every company in one sector with Sectors' own ratios. 1 credit.

    ``level`` is ``sector`` or ``sub_sector``. The ratios come back through
    ``query_values`` for the same reason the classification does — the endpoint
    returns the fields the query names — so each is added to the where clause
    as ``IS NOT NULL or <field> IS NULL``, a condition that is true for every
    row and therefore names the field without filtering on it.
    """
    if level not in ("sector", "sub_sector"):
        raise ValueError("level must be 'sector' or 'sub_sector'")
    escaped = str(name).replace("'", "''")
    clauses = [f"{level} = '{escaped}'"]
    # Naming a ratio must not drop the companies that lack it: a bank with no
    # gross margin still belongs in the comparison, shown as a blank.
    clauses += [f"({f} IS NOT NULL or {f} IS NULL)" for f in SCREENER_RATIOS]
    payload = api_request("companies/", api_key, {
        "where": " and ".join(clauses),
        "order_by": "-market_cap",
        "limit": min(limit, 200),
        "include_query_values": "true"})

    rows = payload.get("results", []) if isinstance(payload, dict) else []
    if not rows:
        return pd.DataFrame()
    records = []
    for row in rows:
        values = row.get("query_values") or {}
        record = {"symbol": normalise_symbol(row.get("symbol", "")),
                  "company_name": row.get("company_name")}
        for source, target in SCREENER_RATIOS.items():
            record[target] = _to_float(values.get(source, row.get(source)))
        record["market_cap"] = _to_float(values.get("market_cap",
                                                    row.get("market_cap")))
        records.append(record)
    return pd.DataFrame(records).drop_duplicates("symbol").reset_index(drop=True)


def rank_ascending(name: str) -> bool:
    """Cheapest first for a multiple, most profitable first for a percentage.

    A low P/E is cheap and a low DER is safer, so multiples sort upward; a high
    ROE is better, so percentages sort downward. Neither direction is a claim
    that the stock is a better investment — only that the ratio reads better.
    """
    spec = FEATURE_BY_NAME.get(name)
    return bool(spec is not None and spec.unit == "multiple")


def peer_percentile(values: pd.Series, name: str) -> pd.Series:
    """Where each company sits among its peers, 0 worst to 100 best."""
    numeric = pd.to_numeric(values, errors="coerce")
    if numeric.notna().sum() < 2:
        return pd.Series(np.nan, index=values.index)
    ranked = numeric.rank(pct=True, ascending=not rank_ascending(name))
    return ranked * 100


def get_company_report(api_key: str, ticker: str,
                       sections: Sequence[str] = ("overview",)) -> dict[str, Any]:
    """Company report. Costs 1 credit PER SECTION, so ask for one."""
    return api_request(f"company/report/{normalise_symbol(ticker)}/", api_key,
                       {"sections": ",".join(sections)})


def get_quarterly_financials(api_key: str, ticker: str, n_quarters: int) -> pd.DataFrame:
    """Quarterly fundamentals. Costs 1 credit PER QUARTER returned."""
    payload = api_request(f"financials/quarterly/{normalise_symbol(ticker)}/",
                          api_key, {"n_quarters": n_quarters})
    if not isinstance(payload, list) or not payload:
        return pd.DataFrame()
    rows = [{k: v for k, v in item.items() if k != "financials_sector_metrics"}
            for item in payload if isinstance(item, dict)]
    frame = pd.DataFrame(rows)
    if "date" in frame.columns:
        frame = frame.rename(columns={"date": "report_date"})
    if "report_date" not in frame.columns:
        return pd.DataFrame()
    frame["report_date"] = pd.to_datetime(frame["report_date"])
    return frame.drop_duplicates("report_date").sort_values("report_date").reset_index(drop=True)


def _chunks(start: dt.date, end: dt.date, size: int = MAX_DAILY_WINDOW_DAYS):
    """Inclusive windows of at most ``size`` days."""
    out, cursor = [], start
    while cursor <= end:
        stop = min(cursor + dt.timedelta(days=size - 1), end)
        out.append((cursor, stop))
        cursor = stop + dt.timedelta(days=1)
    return out


def get_daily_history(api_key: str, ticker: str, start_date, end_date) -> pd.DataFrame:
    """Daily close, volume and market cap, split to the 90-day API limit.

    The endpoint silently clamps any wider window to the most recent 90 days,
    so a naive five-year request returns three months and no error.
    """
    start, end = pd.Timestamp(start_date).date(), pd.Timestamp(end_date).date()
    end = min(end, dt.date.today())          # a future end is a 400
    if start > end:
        return pd.DataFrame(columns=["date", "close", "volume", "market_cap"])

    frames = []
    for chunk_start, chunk_end in _chunks(start, end):
        payload = api_request(f"daily/{normalise_symbol(ticker)}/", api_key,
                              {"start": chunk_start.isoformat(),
                               "end": chunk_end.isoformat()})
        if isinstance(payload, list) and payload:
            frames.append(pd.DataFrame(payload))
    if not frames:
        return pd.DataFrame(columns=["date", "close", "volume", "market_cap"])

    history = pd.concat(frames, ignore_index=True)
    history["date"] = pd.to_datetime(history["date"])
    # Open, high and low are coerced too where the endpoint returns them: the
    # dashboard offers a candlestick, and it can only draw one for a company
    # whose filings actually carry the full bar.
    for column in ("close", "volume", "market_cap", "open", "high", "low"):
        if column in history.columns:
            history[column] = pd.to_numeric(history[column], errors="coerce")
    return history.drop_duplicates("date").sort_values("date").reset_index(drop=True)


# ══════════════════════════════════════════════════════════════════════
# ON-DISK CACHE — this is what makes the credit spend a one-off
# ══════════════════════════════════════════════════════════════════════

def cache_dir(base: str = "data") -> Path:
    path = Path(base) / "cache"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _cache_path(ticker: str, kind: str, base: str = "data") -> Path:
    return cache_dir(base) / f"{normalise_symbol(ticker)}_{kind}.parquet"


def is_cached(ticker: str, kind: str, base: str = "data") -> bool:
    return _cache_path(ticker, kind, base).exists()


def save_to_cache(frame: pd.DataFrame, ticker: str, kind: str, base: str = "data") -> None:
    if frame is None or frame.empty:
        return
    out = frame.copy()
    out["ticker"] = normalise_symbol(ticker)
    out.to_parquet(_cache_path(ticker, kind, base), index=False)


def load_from_cache(ticker: str, kind: str, base: str = "data") -> pd.DataFrame:
    path = _cache_path(ticker, kind, base)
    if not path.exists():
        return pd.DataFrame()
    try:
        frame = pd.read_parquet(path)
    except Exception:
        return pd.DataFrame()
    for column in ("date", "report_date"):
        if column in frame.columns:
            frame[column] = pd.to_datetime(frame[column])
    return frame


def cached_tickers(kind: str = "quarterly", base: str = "data",
                   complete_only: bool = True) -> list[str]:
    """Tickers on disk. By default only those with BOTH halves present.

    A ticker whose price history failed after its fundamentals were saved is
    real, paid-for data worth keeping for the next run to finish — but it is
    not a usable observation, and nothing downstream should see it.
    """
    suffix = f"_{kind}.parquet"
    names = sorted(p.name[: -len(suffix)] for p in cache_dir(base).glob(f"*{suffix}"))
    if not complete_only:
        return names
    return [t for t in names
            if is_cached(t, "quarterly", base) and is_cached(t, "prices", base)]


def cache_as_of(base: str = "data") -> str:
    """Newest report date in the cache, so a snapshot can be labelled honestly."""
    newest = None
    for path in cache_dir(base).glob("*_quarterly.parquet"):
        try:
            frame = pd.read_parquet(path, columns=["report_date"])
        except Exception:
            continue
        if frame.empty:
            continue
        latest = pd.to_datetime(frame["report_date"]).max()
        newest = latest if newest is None or latest > newest else newest
    return "unknown" if newest is None else f"{newest:%Y-%m-%d}"


def merge_universe(existing: pd.DataFrame, fresh: pd.DataFrame) -> pd.DataFrame:
    """Add columns and rows from a newer screen without losing older ones.

    A later screen can miss a company that has since fallen below the filter,
    and its cached price and fundamental data are already bought and paid for.
    Columns are filled in rather than replaced wholesale for the same reason.
    """
    if existing is None or existing.empty:
        return fresh.reset_index(drop=True) if fresh is not None else pd.DataFrame()
    if fresh is None or fresh.empty:
        return existing.reset_index(drop=True)
    merged = existing.set_index("symbol").combine_first(fresh.set_index("symbol"))
    for column in fresh.columns:
        if column == "symbol":
            continue
        incoming = fresh.set_index("symbol")[column]
        merged[column] = incoming.reindex(merged.index).fillna(merged.get(column))
    return merged.reset_index()


def cache_universe(universe: pd.DataFrame, base: str = "data") -> None:
    """Cache the screener result too — otherwise a re-run still costs 1 credit."""
    if universe is not None and not universe.empty:
        universe.to_parquet(cache_dir(base) / "universe.parquet", index=False)


def company_names(base: str = "data") -> dict[str, str]:
    """Ticker to registered company name, from the cached universe screen.

    The screener result is already on disk, so the dashboard can show
    "BBCA — PT Bank Central Asia Tbk." in cached mode without spending a
    credit to learn a name it already paid for once.
    """
    universe = load_universe(base)
    if universe.empty or "company_name" not in universe.columns:
        return {}
    return {normalise_symbol(row.symbol): str(row.company_name)
            for row in universe.itertuples()
            if isinstance(getattr(row, "company_name", None), str)}


def company_dividends(ticker: str, base: str = "data") -> dict[str, float]:
    """Trailing dividend figures from the cached screen, where present."""
    universe = load_universe(base)
    if universe.empty or "symbol" not in universe.columns:
        return {}
    match = universe[universe["symbol"].map(normalise_symbol) == normalise_symbol(ticker)]
    if match.empty:
        return {}
    row = match.iloc[0]
    values = {m: _to_float(row[m]) for m in DIVIDEND_FIELDS.values()
              if m in match.columns and np.isfinite(_to_float(row.get(m)))}

    # yield_ttm comes back as exactly 0 when the screener has no dividend for a
    # company, not when the company pays nothing. Across the 200-name screen
    # every one of the 29 zero yields had a missing dividend_ttm beside it and
    # none had a real one, and the list is BBNI, BBTN, CPIN, GEMS, HRUM — all
    # routine payers. Printing 0.0% for them would state something untrue about
    # a real company, so a zero without a dividend is read as unknown.
    if values.get("dividend_yield") == 0 and "dividend" not in values:
        values.pop("dividend_yield")
    return values


def screen_as_of(base: str = "data") -> str:
    """The date the screener snapshot was taken, or an empty string."""
    universe = load_universe(base)
    if universe.empty or "screened_at" not in universe.columns:
        return ""
    values = universe["screened_at"].dropna()
    return str(values.iloc[0]) if len(values) else ""


def company_classification(ticker: str, base: str = "data") -> dict[str, str]:
    """Sector, sub-sector and industry from the cached screen, where present."""
    universe = load_universe(base)
    if universe.empty or "symbol" not in universe.columns:
        return {}
    match = universe[universe["symbol"].map(normalise_symbol) == normalise_symbol(ticker)]
    if match.empty:
        return {}
    row = match.iloc[0]
    return {f: str(row[f]) for f in CLASSIFICATION_FIELDS
            if f in match.columns and isinstance(row.get(f), str) and row[f].strip()}


def company_name(ticker: str, base: str = "data") -> str:
    """The registered name, or the ticker itself when it is not known."""
    return company_names(base).get(normalise_symbol(ticker), normalise_symbol(ticker))


def load_universe(base: str = "data") -> pd.DataFrame:
    path = cache_dir(base) / "universe.parquet"
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_parquet(path)
    except Exception:
        return pd.DataFrame()


def collect_ticker(api_key: str, ticker: str, *, n_quarters: int,
                   price_start: str, price_end: str, base: str = "data",
                   force: bool = False) -> tuple[pd.DataFrame, pd.DataFrame, bool]:
    """Fetch one ticker, or reuse the cache. Returns (quarterly, prices, paid).

    Each half is banked the moment it arrives, and each half is fetched only if
    it is missing. Waiting for a complete pair before writing anything looks
    tidier but throws away credits that have already been charged: a ticker
    whose fundamentals were bought and whose price history then hit a 429 lost
    the fundamentals too, and the next run bought them again. One real run lost
    around a hundred credits that way, nine tickers at a time.

    The complete-pair rule still holds, but where it belongs — at read time.
    ``cached_tickers`` reports only tickers with both halves present, so a
    half-collected ticker is never treated as usable data.
    """
    have_quarterly = not force and is_cached(ticker, "quarterly", base)
    have_prices = not force and is_cached(ticker, "prices", base)
    if have_quarterly and have_prices:
        return (load_from_cache(ticker, "quarterly", base),
                load_from_cache(ticker, "prices", base), False)

    if have_quarterly:
        quarterly = load_from_cache(ticker, "quarterly", base)
    else:
        quarterly = get_quarterly_financials(api_key, ticker, n_quarters)
        if not quarterly.empty:
            save_to_cache(quarterly, ticker, "quarterly", base)

    if have_prices:
        prices = load_from_cache(ticker, "prices", base)
    else:
        prices = get_daily_history(api_key, ticker, price_start, price_end)
        if not prices.empty:
            save_to_cache(prices, ticker, "prices", base)

    return quarterly, prices, True


# ══════════════════════════════════════════════════════════════════════
# FEATURES — exactly ten
# ══════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class FeatureSpec:
    name: str
    label: str
    category: str
    meaning: str
    unit: str = "ratio"          # "percent" | "multiple" | "currency" | "ratio"
    # What the acronym stands for. Empty where the label is already words.
    # "ROE" is only obvious to someone who already knows what it is, and this
    # dashboard is aimed at people deciding whether they trust the number.
    expansion: str = ""
    # Whether the model may read it. Absolute rupiah amounts are shown because
    # a reader wants them, but they cannot be model inputs: a bank with IDR
    # 1,300T of assets and a small cap with IDR 2T are not on one scale, and a
    # tree that splits on the level is splitting on company size. Only
    # scale-free ratios are modelled.
    modelled: bool = True
    # False for a figure that arrives as a snapshot from the screener rather
    # than reconstructed at each historical date. Such a value is true as of
    # the screen and only as of the screen, so it may be displayed but must
    # never reach the model: feeding today's dividend yield to a 2022
    # observation is look-ahead, the precise failure the leakage audit exists
    # to catch.
    point_in_time: bool = True

    @property
    def title(self) -> str:
        return f"{self.label} ({self.expansion})" if self.expansion else self.label


# The metric set, in the order the dashboard presents it. Six categories, and
# not every row is a model input — see FeatureSpec.modelled.
#
# NPL, LDR, NIM and the dividend metrics are deliberately absent. They need
# gross loans, deposits, net interest income and a dividend history, none of
# which the quarterly financials endpoint returns, so every company would show
# an empty row forever. The limitation is recorded in the README instead of
# occupying six lines of a table nobody can read a number from.
FEATURE_SCHEMA: tuple[FeatureSpec, ...] = (
    # — 1. Valuation -------------------------------------------------
    FeatureSpec("pe", "P/E", "Valuation",
                "Price relative to trailing 12-month earnings.", "multiple",
                "Price to Earnings"),
    FeatureSpec("ps", "P/S", "Valuation",
                "Price relative to trailing 12-month revenue.", "multiple",
                "Price to Sales"),
    FeatureSpec("pbv", "PBV", "Valuation",
                "Price relative to book value of equity.", "multiple",
                "Price to Book Value"),
    FeatureSpec("pcf", "P/CF", "Valuation",
                "Price relative to trailing 12-month operating cash flow. "
                "Harder to flatter than earnings.", "multiple",
                "Price to Cash Flow"),
    FeatureSpec("ev_ebitda", "EV/EBITDA", "Valuation",
                "Enterprise value — market cap plus debt, less cash — against "
                "trailing 12-month EBITDA. Neutral to how a company is financed.",
                "multiple", "Enterprise Value to EBITDA"),

    # — 2. Per Share -------------------------------------------------
    # Rupiah amounts, so shown but never modelled. The share count is inferred
    # as market cap divided by close, which is exact on the day it is taken.
    FeatureSpec("eps", "EPS", "Per Share",
                "Trailing 12-month earnings attributable to one share.",
                "currency", "Earning per Share", modelled=False),
    FeatureSpec("rps", "RPS", "Per Share",
                "Trailing 12-month revenue per share.",
                "currency", "Revenue per Share", modelled=False),
    FeatureSpec("cps", "CPS", "Per Share",
                "Cash and equivalents held per share.",
                "currency", "Cash per Share", modelled=False),
    FeatureSpec("bvps", "BVPS", "Per Share",
                "Book value of equity per share.",
                "currency", "Book Value per Share", modelled=False),
    FeatureSpec("cfps", "CFPS", "Per Share",
                "Trailing 12-month operating cash flow per share.",
                "currency", "Cash Flow per Share", modelled=False),

    # — 3. Solvency --------------------------------------------------
    FeatureSpec("der", "DER", "Solvency",
                "Total liabilities relative to shareholder equity. The API does "
                "not separate interest-bearing debt, so this is the broader "
                "measure.", "multiple", "Debt to Equity"),

    # — 4. Profitability ---------------------------------------------
    FeatureSpec("roa", "ROA", "Profitability",
                "Return generated from total assets.", "percent",
                "Return on Asset"),
    FeatureSpec("roe", "ROE", "Profitability",
                "Return generated on shareholder equity.", "percent",
                "Return on Equity"),
    FeatureSpec("gpm", "GPM", "Profitability",
                "Revenue left after the direct cost of producing it.", "percent",
                "Gross Profit Margin"),
    FeatureSpec("opm", "OPM", "Profitability",
                "Revenue left after operating costs, before financing and tax.",
                "percent", "Operating Profit Margin"),
    FeatureSpec("npm", "NPM", "Profitability",
                "Profit generated per unit of revenue.", "percent",
                "Net Profit Margin"),

    # — 5. Dividend ---------------------------------------------------
    # Trailing figures from the screener, not point-in-time history. Shown,
    # never modelled — see FeatureSpec.point_in_time.
    FeatureSpec("dividend", "Dividend", "Dividend",
                "Cash paid per share over the trailing twelve months.",
                "currency", "", modelled=False, point_in_time=False),
    FeatureSpec("dpr", "DPR", "Dividend",
                "Share of earnings paid out rather than retained.", "percent",
                "Dividend Payout Ratio", modelled=False, point_in_time=False),
    FeatureSpec("dividend_yield", "Dividend Yield", "Dividend",
                "Trailing dividend against the current price.", "percent", "",
                modelled=False, point_in_time=False),

    # — 6. Income Statement ------------------------------------------
    FeatureSpec("revenue", "Revenue", "Income Statement",
                "Trailing 12-month revenue.", "currency", "", modelled=False),
    FeatureSpec("gross_profit", "Gross Profit", "Income Statement",
                "Trailing 12-month revenue less the direct cost of producing it.",
                "currency", "", modelled=False),
    FeatureSpec("ebitda", "EBITDA", "Income Statement",
                "Trailing 12-month earnings before interest, tax, depreciation "
                "and amortisation.", "currency",
                "Earnings Before Interest, Tax, Depreciation and Amortisation",
                modelled=False),
    FeatureSpec("net_income", "Net Income", "Income Statement",
                "Trailing 12-month profit after everything.", "currency", "",
                modelled=False),

    # — 7. Balance Sheet ---------------------------------------------
    FeatureSpec("cash", "Cash", "Balance Sheet",
                "Cash and equivalents at the reporting date.", "currency", "",
                modelled=False),
    FeatureSpec("total_assets", "Total Assets", "Balance Sheet",
                "Everything the company owns at the reporting date.", "currency",
                "", modelled=False),
    FeatureSpec("total_liabilities", "Total Liabilities", "Balance Sheet",
                "Everything the company owes at the reporting date.", "currency",
                "", modelled=False),
    FeatureSpec("total_equity", "Total Equity", "Balance Sheet",
                "What is left for shareholders: assets less liabilities.",
                "currency", "", modelled=False),
)

METRIC_NAMES: list[str] = [f.name for f in FEATURE_SCHEMA]
FEATURE_BY_NAME = {f.name: f for f in FEATURE_SCHEMA}

# What the model is allowed to read: the scale-free ratios, and only those.
FEATURE_NAMES: list[str] = [f.name for f in FEATURE_SCHEMA if f.modelled]

CATEGORY_ORDER: list[str] = list(dict.fromkeys(f.category for f in FEATURE_SCHEMA))


def _to_float(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return np.nan
    return result if np.isfinite(result) else np.nan


def _ratio(numerator: Any, denominator: Any, *, positive_denominator: bool = False) -> float:
    """Divide, returning NaN rather than infinity or a meaningless sign."""
    num, den = _to_float(numerator), _to_float(denominator)
    if not np.isfinite(num) or not np.isfinite(den) or den == 0:
        return np.nan
    if positive_denominator and den <= 0:
        return np.nan
    result = num / den
    return result if np.isfinite(result) else np.nan


def _growth(current: Any, past: Any) -> float:
    """Year-on-year growth. Growth off a zero or negative base is undefined."""
    now, before = _to_float(current), _to_float(past)
    if not np.isfinite(now) or not np.isfinite(before) or before <= 0:
        return np.nan
    return now / before - 1.0


def detect_quarterly_basis(quarterly: pd.DataFrame) -> str:
    """Are the income-statement figures cumulative year-to-date?

    Indonesian issuers commonly file cumulative statements: Q2 is the half
    year, Q4 the full year. Summing four of those as standalone quarters
    overstates trailing revenue by roughly 2.5x and corrupts every margin and
    multiple downstream.

    Monotonicity alone cannot tell the two apart — a steadily growing company
    also reports four rising quarters. The discriminator is the *size* of the
    rise: cumulative predicts Qn ≈ n × Q1, discrete predicts Qn ≈ Q1.
    """
    if quarterly.empty or "revenue" not in quarterly.columns:
        return "discrete"
    frame = quarterly.dropna(subset=["revenue"]).copy()
    if frame.empty:
        return "discrete"
    frame["year"] = frame["report_date"].dt.year
    frame["quarter"] = frame["report_date"].dt.quarter

    votes = []
    for _, group in frame.groupby("year"):
        group = group.sort_values("quarter")
        if 1 not in set(group["quarter"]):
            continue
        first = _to_float(group.loc[group["quarter"] == 1, "revenue"].iloc[0])
        last_row = group.iloc[-1]
        quarter_number, last = int(last_row["quarter"]), _to_float(last_row["revenue"])
        if first <= 0 or quarter_number < 2 or not np.isfinite(last):
            continue
        votes.append((last / first) >= (1.0 + quarter_number) / 2.0)
    if not votes:
        return "discrete"
    return "cumulative" if sum(votes) / len(votes) >= 0.60 else "discrete"


def build_panel(quarterly: pd.DataFrame, basis: str = "auto") -> pd.DataFrame:
    """Turn raw quarterly filings into a trailing-twelve-month panel.

    De-cumulates where needed, then adds TTM sums, four-quarter average
    balance-sheet denominators, and the date each row becomes usable.
    """
    if quarterly.empty:
        return quarterly

    resolved = detect_quarterly_basis(quarterly) if basis == "auto" else basis
    panel = quarterly.sort_values("report_date").reset_index(drop=True).copy()
    panel["quarterly_basis"] = resolved

    # Flows accumulate through the year on a cumulative filer and must be
    # de-cumulated; levels are a photograph of one date and must not be.
    flows = ("revenue", "earnings", "operating_cash_flow", "gross_profit",
             "operating_pnl", "ebitda", "cost_of_revenue", "operating_expense")
    levels = ("total_assets", "total_equity", "total_liabilities",
              "cash_only", "total_debt")
    for column in (*flows, *levels):
        if column in panel.columns:
            panel[column] = pd.to_numeric(panel[column], errors="coerce")
        else:
            panel[column] = np.nan

    if resolved == "cumulative":
        panel["_year"] = panel["report_date"].dt.year
        panel["_q"] = panel["report_date"].dt.quarter
        for column in flows:
            previous = panel.groupby("_year")[column].shift(1)
            previous_q = panel.groupby("_year")["_q"].shift(1)
            adjacent = (panel["_q"] - previous_q) == 1
            panel[column] = np.where(
                adjacent & previous.notna(), panel[column] - previous,
                np.where(panel["_q"] == 1, panel[column], np.nan))
        panel = panel.drop(columns=["_year", "_q"])

    for source, target in (("revenue", "revenue_ttm"),
                           ("earnings", "earnings_ttm"),
                           ("operating_cash_flow", "ocf_ttm"),
                           ("gross_profit", "gross_profit_ttm"),
                           ("operating_pnl", "operating_profit_ttm"),
                           ("ebitda", "ebitda_ttm"),
                           ("cost_of_revenue", "cost_of_revenue_ttm"),
                           ("operating_expense", "operating_expense_ttm")):
        panel[target] = panel[source].rolling(4, min_periods=4).sum()
    panel["avg_equity"] = panel["total_equity"].rolling(4, min_periods=2).mean()
    panel["avg_assets"] = panel["total_assets"].rolling(4, min_periods=2).mean()
    panel["available_date"] = panel["report_date"] + pd.Timedelta(days=REPORTING_LAG_DAYS)
    return panel


def compute_features(panel: pd.DataFrame, market_cap: float | None,
                     index: int | None = None,
                     close: float | None = None) -> dict[str, float]:
    """Every metric in FEATURE_SCHEMA. THE single source of truth for both paths.

    The training script calls this at every historical observation; the app
    calls it once with the latest quarter. Because both go through here, a
    metric cannot mean one thing in training and another in production.

    Ratios that would be economically meaningless are NaN, never zero: a P/E
    built on a loss is a category error, not a cheap stock. A negative margin
    or return IS meaningful and is kept, so only a negative denominator voids
    those.
    """
    metrics = {name: np.nan for name in METRIC_NAMES}
    if panel is None or panel.empty:
        return metrics
    position = len(panel) - 1 if index is None else index
    if position < 0 or position >= len(panel):
        return metrics

    row = panel.iloc[position]
    cap, price = _to_float(market_cap), _to_float(close)
    revenue, earnings = row.get("revenue_ttm"), row.get("earnings_ttm")
    ocf, ebitda = row.get("ocf_ttm"), row.get("ebitda_ttm")
    gross, operating = row.get("gross_profit_ttm"), row.get("operating_profit_ttm")
    equity, assets = row.get("total_equity"), row.get("total_assets")
    liabilities, cash = row.get("total_liabilities"), row.get("cash_only")
    debt = row.get("total_debt")

    # — 1. Valuation. Built from market cap, so share counts and splits do
    #       not enter the ratio at all.
    metrics["pe"] = _ratio(cap, earnings, positive_denominator=True)
    metrics["ps"] = _ratio(cap, revenue, positive_denominator=True)
    metrics["pbv"] = _ratio(cap, equity, positive_denominator=True)
    metrics["pcf"] = _ratio(cap, ocf, positive_denominator=True)

    # Enterprise value needs interest-bearing debt, which only some filings
    # carry. Substituting total liabilities would quietly turn EV/EBITDA into a
    # different ratio for banks than for miners, so it stays NaN instead.
    if np.isfinite(cap) and pd.notna(debt):
        enterprise = cap + _to_float(debt) - (_to_float(cash) if pd.notna(cash) else 0.0)
        metrics["ev_ebitda"] = _ratio(enterprise, ebitda, positive_denominator=True)

    # — 2. Per share. The share count is market cap divided by close, which is
    #       exact on the day it is taken and needs no separate field.
    shares = cap / price if np.isfinite(cap) and np.isfinite(price) and price > 0 else np.nan
    if np.isfinite(shares) and shares > 0:
        metrics["eps"] = _ratio(earnings, shares)
        metrics["rps"] = _ratio(revenue, shares)
        metrics["cps"] = _ratio(cash, shares)
        metrics["bvps"] = _ratio(equity, shares)
        metrics["cfps"] = _ratio(ocf, shares)

    # — 3. Solvency. NPL and LDR need loan and deposit fields the endpoint
    #       does not return; they stay NaN and the interface says why.
    if pd.notna(liabilities) and pd.notna(equity) and _to_float(equity) > 0:
        metrics["der"] = _ratio(liabilities, equity)
    elif pd.notna(assets) and pd.notna(equity) and _to_float(equity) > 0:
        # Older cached filings carry assets and equity but not liabilities.
        metrics["der"] = _ratio(_to_float(assets) - _to_float(equity), equity)

    # — 4. Profitability.
    metrics["roe"] = _ratio(earnings, row.get("avg_equity"), positive_denominator=True)
    metrics["roa"] = _ratio(earnings, row.get("avg_assets"), positive_denominator=True)
    metrics["gpm"] = _ratio(gross, revenue, positive_denominator=True)
    metrics["opm"] = _ratio(operating, revenue, positive_denominator=True)
    metrics["npm"] = _ratio(earnings, revenue, positive_denominator=True)

    # — 5. Dividend. Nothing to compute: see FeatureSpec.unavailable.

    # — 6/7. Levels, reported as they stand. Shown, never modelled.
    metrics["revenue"] = _to_float(revenue)
    metrics["gross_profit"] = _to_float(gross)
    metrics["ebitda"] = _to_float(ebitda)
    metrics["net_income"] = _to_float(earnings)
    metrics["cash"] = _to_float(cash)
    metrics["total_assets"] = _to_float(assets)
    metrics["total_liabilities"] = _to_float(liabilities)
    metrics["total_equity"] = _to_float(equity)
    return metrics


def features_frame(quarterly: pd.DataFrame, market_cap: float | None,
                   close: float | None = None) -> pd.DataFrame:
    """One-row DataFrame carrying every metric, in schema order."""
    panel = build_panel(quarterly)
    return pd.DataFrame([compute_features(panel, market_cap, close=close)],
                        columns=METRIC_NAMES)


def income_statement_series(quarterly: pd.DataFrame) -> pd.DataFrame:
    """Per-quarter revenue, cost and net income for the comparison chart.

    De-cumulated first, so a cumulative filer's Q4 is one quarter rather than
    the whole year. Cost is the direct cost of revenue where a company reports
    it and operating expense where it does not — banks report the second and
    not the first — and the caller is told which, because plotting two
    different quantities under one label would be worse than plotting neither.
    """
    if quarterly is None or quarterly.empty:
        return pd.DataFrame()
    panel = build_panel(quarterly)
    if panel.empty:
        return pd.DataFrame()

    for column, label in (("cost_of_revenue", "Cost of revenue"),
                          ("operating_expense", "Operating expense")):
        series = panel.get(column)
        if series is not None and series.notna().sum() >= 4:
            cost, cost_label = series.abs(), label
            break
    else:
        cost, cost_label = pd.Series(np.nan, index=panel.index), "Cost"

    frame = pd.DataFrame({
        "report_date": panel["report_date"],
        "revenue": panel.get("revenue"),
        "cost": cost,
        "net_income": panel.get("earnings"),
    }).dropna(subset=["revenue"], how="all")
    frame.attrs["cost_label"] = cost_label
    return frame


def data_quality(features: pd.DataFrame, model_features: Sequence[str]) -> float:
    """Share of the model's own inputs actually present, 0..1.

    Not the same thing as model reliability: this measures the data for one
    company, not how well the model has performed.
    """
    if features is None or features.empty or not model_features:
        return 0.0
    present = sum(1 for name in model_features
                  if name in features.columns and pd.notna(features.iloc[0][name]))
    return present / len(model_features)


# ══════════════════════════════════════════════════════════════════════
# OBSERVATIONS AND TARGETS
# ══════════════════════════════════════════════════════════════════════

def build_observations(ticker: str, quarterly: pd.DataFrame,
                       prices: pd.DataFrame) -> pd.DataFrame:
    """Point-in-time observation rows for one ticker.

        report date -> +90d lag -> next trading day -> observation

    Features use only the quarters public by that date, and the market cap
    observed on that date — never today's.
    """
    if quarterly.empty or prices.empty:
        return pd.DataFrame()

    panel = build_panel(quarterly)
    history = prices.sort_values("date").reset_index(drop=True)
    dates = history["date"].to_numpy()
    first_price_date = history["date"].iloc[0]

    rows = []
    for index in range(len(panel)):
        available = panel.iloc[index]["available_date"]
        # If the data became public before the price history starts, there is
        # no honest observation to make. searchsorted would return position 0
        # and silently pair an old report with the wrong price, inventing a
        # row that never existed.
        if available < first_price_date:
            continue
        position = int(np.searchsorted(dates, np.datetime64(available), side="left"))
        if position >= len(history):
            continue                                  # not tradable yet
        price_row = history.iloc[position]
        market_cap = _to_float(price_row.get("market_cap"))
        rows.append({
            "ticker": normalise_symbol(ticker),
            "observation_date": price_row["date"],
            "report_date": panel.iloc[index]["report_date"],
            "price_index": position,
            "close": _to_float(price_row.get("close")),
            "market_cap": market_cap,
            **compute_features(panel.iloc[: index + 1], market_cap, index,
                               close=_to_float(price_row.get("close"))),
        })
    return pd.DataFrame(rows)


def add_targets(observations: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    """Forward returns and binary targets, on trading-day offsets.

    Uses offsets into the ticker's own price series (126 and 252 observations
    ahead), never calendar arithmetic, because IDX closes for weekends and a
    long list of national holidays.

    Rows whose window has not closed keep NaN targets. Casting those to 0 would
    turn "unknown" into a confident "went down".
    """
    if observations.empty or prices.empty:
        return observations

    history = prices.sort_values("date").reset_index(drop=True)
    closes = history["close"].to_numpy(dtype=float)
    dates = history["date"].to_numpy()
    total = len(history)
    result = observations.copy()

    for horizon, offset in HORIZON_TRADING_DAYS.items():
        returns, resolved = [], []
        for position in result["price_index"].to_numpy(dtype=int):
            future = position + offset
            if future < total and np.isfinite(closes[position]) and closes[position] > 0:
                returns.append(closes[future] / closes[position] - 1.0)
                resolved.append(dates[future])
            else:
                returns.append(np.nan); resolved.append(pd.NaT)
        series = pd.Series(returns, index=result.index)
        result[f"forward_return_{horizon}"] = series
        result[f"target_{horizon}"] = np.where(series.notna(),
                                               (series > 0).astype(float), np.nan)
        result[f"target_available_{horizon}"] = pd.to_datetime(
            pd.Series(resolved, index=result.index))
    return result


def build_dataset(frames) -> pd.DataFrame:
    """Concatenate, de-duplicate and clean the per-ticker observation frames."""
    frames = [f for f in frames if f is not None and not f.empty]
    if not frames:
        return pd.DataFrame()
    dataset = pd.concat(frames, ignore_index=True)
    dataset = dataset.drop_duplicates(subset=["ticker", "observation_date"])
    dataset = dataset.sort_values(["observation_date", "ticker"]).reset_index(drop=True)

    present = [c for c in METRIC_NAMES if c in dataset.columns]
    dataset[present] = dataset[present].replace([np.inf, -np.inf], np.nan)
    # A price multiple cannot be negative: the denominator was already
    # required to be positive, so a negative here is arithmetic noise.
    for column in ("pe", "ps", "pbv", "pcf", "ev_ebitda"):
        if column in dataset.columns:
            dataset.loc[dataset[column] <= 0, column] = np.nan
    return dataset


def usable_features(dataset: pd.DataFrame,
                    max_missing: float = MAX_FEATURE_MISSINGNESS) -> tuple[list[str], pd.DataFrame]:
    """Drop features too often missing to be trusted, and report why."""
    present = [c for c in FEATURE_NAMES if c in dataset.columns]
    if not present:
        return [], pd.DataFrame(columns=["feature", "missing_rate", "retained"])
    report = (dataset[present].isna().mean().rename("missing_rate")
              .reset_index().rename(columns={"index": "feature"}))
    report["retained"] = report["missing_rate"] <= max_missing
    return report.loc[report["retained"], "feature"].tolist(), report.sort_values("missing_rate")


def information_coefficient(dataset: pd.DataFrame, feature: str, horizon: str,
                            min_cross_section: int = MIN_CROSS_SECTION) -> float:
    """How well a ratio ranks stocks against what they went on to do.

    Measured within each quarter and then averaged, never pooled. Pooling would
    let a quarter in which everything rose sit above a quarter in which
    everything fell, and reward the ratio for a market move it had no part in —
    the same trap the fold-averaged ROC-AUC avoids elsewhere in this project.

    Spearman rather than Pearson because only the ordering is claimed, and
    because one company on a 900x P/E would otherwise set the correlation by
    itself. Computed from ranks with pandas so the project keeps its
    dependency list; scipy is not installed here.
    """
    forward = f"forward_return_{horizon}"
    if feature not in dataset.columns or forward not in dataset.columns:
        return float("nan")
    frame = dataset.dropna(subset=[feature, forward])
    quarterly = []
    for _, quarter in frame.groupby("observation_date"):
        if len(quarter) < min_cross_section or quarter[feature].nunique() < 2:
            continue
        correlation = quarter[feature].rank().corr(quarter[forward].rank())
        if pd.notna(correlation):
            quarterly.append(float(correlation))
    return float(np.mean(quarterly)) if len(quarterly) >= 3 else float("nan")


def screen_features(dataset: pd.DataFrame, features: Sequence[str], horizon: str,
                    min_ic: float = MIN_FEATURE_IC,
                    keep_at_least: int = MIN_SCREENED_FEATURES) -> list[str]:
    """Keep the ratios that rank; drop the ones that only take up room.

    THE SLICE PASSED IN IS THE WHOLE ARGUMENT. Call this with a fold's training
    rows and it is a screen. Call it with the full panel and it is look-ahead:
    the ratios it likes were chosen partly by the returns it is about to be
    scored against, and the resulting figure is unreproducible in front of a
    live market. The caller owns that distinction, so it is stated here rather
    than assumed.

    A ratio whose IC cannot be computed at all — too few quarters wide enough
    to rank — is kept rather than dropped. Absence of measurement is not
    evidence of absence, and the missingness gate already removed the ratios
    that are genuinely too sparse to use.
    """
    scores = {f: information_coefficient(dataset, f, horizon) for f in features}
    unmeasurable = [f for f in features if not np.isfinite(scores[f])]
    measured = sorted((f for f in features if np.isfinite(scores[f])),
                      key=lambda f: -abs(scores[f]))
    kept = [f for f in measured if abs(scores[f]) >= min_ic] + unmeasurable
    if len(kept) >= keep_at_least:
        return [f for f in features if f in set(kept)]
    ranked = measured + unmeasurable
    return [f for f in features if f in set(ranked[:keep_at_least])]


def walk_forward_folds(dataset: pd.DataFrame, horizon: str,
                       scheme: str = "date", min_train: int = 40,
                       min_validation: int = 5,
                       min_train_years: int = 2) -> list[dict[str, Any]]:
    """Expanding chronological folds, purged at the boundary.

    Purging is the subtle part. A 12-month target observed in June 2022 does
    not resolve until June 2023. If June 2023 is on the validation side, that
    2022 row already encodes information about the validation period, so it is
    excluded from training. A training row is admitted only once its own
    forward window has closed.

    Two schemes, and the choice matters more than it looks:

    ``date``  one fold per observation date — that is, per quarterly rebalance.
              On this panel that is nine folds at 6M and five at 12M.
    ``year``  one fold per calendar year. Two folds at 6M, one at 12M.

    ``date`` is the default because a stability figure computed from two folds
    is not a measurement. The old yearly scheme reported the 6M model as 85/100
    stable off a standard deviation taken across exactly two numbers; scored
    per rebalance the same model is nowhere near that consistent. Fewer, larger
    folds do not make a model better, they make its variance invisible.
    """
    target = f"target_{horizon}"
    resolution = f"target_available_{horizon}"
    if dataset.empty or target not in dataset.columns:
        return []
    usable = dataset.dropna(subset=[target]).copy()
    if usable.empty:
        return []
    usable["observation_date"] = pd.to_datetime(usable["observation_date"])

    if scheme == "year":
        years = sorted(usable["observation_date"].dt.year.unique())
        if len(years) <= min_train_years:
            return []
        boundaries = [(pd.Timestamp(year=y, month=1, day=1),
                       pd.Timestamp(year=y, month=12, day=31), int(y))
                      for y in years[min_train_years:]]
    else:
        boundaries = [(pd.Timestamp(d), pd.Timestamp(d), pd.Timestamp(d).date().isoformat())
                      for d in sorted(usable["observation_date"].unique())]

    folds = []
    for start, end, label in boundaries:
        closed = (usable[resolution] < start if resolution in usable
                  else usable["observation_date"] < start)
        train = (usable["observation_date"] < start) & closed
        validate = usable["observation_date"].between(start, end)
        if train.sum() < min_train or validate.sum() < min_validation:
            continue
        folds.append({"validation_year": label,
                      "validation_start": start,
                      "train_index": usable.index[train],
                      "validation_index": usable.index[validate],
                      "n_train": int(train.sum()),
                      "n_validation": int(validate.sum())})
    return folds


# ══════════════════════════════════════════════════════════════════════
# MODEL WRAPPERS
# ══════════════════════════════════════════════════════════════════════

class PriorShrunk:
    """Pull a model's probability back toward the training base rate.

        p = weight * p_model + (1 - weight) * prior

    A model that cannot separate winners from losers out of sample should not
    be allowed to say 85%. Shrinking toward the base rate keeps whatever
    ordering the model found — the ranking is unchanged, because the blend is
    monotone — while capping how far any single number may travel from what
    the historical frequency actually supports.

    ``weight`` is not a taste setting. ``train.py`` fits it out of sample on
    log loss, and on a panel this small it comes back small, which is the
    honest answer rather than a disappointing one.

    Lives here rather than in ``train.py`` so that ``joblib`` can resolve the
    class when the dashboard loads an exported model.
    """

    def __init__(self, estimator: Any, weight: float = 1.0):
        self.estimator = estimator
        self.weight = float(weight)

    def fit(self, X, y, **kwargs):
        labels = np.asarray(y, dtype=float)
        self.prior_ = float(np.clip(np.nanmean(labels), 1e-6, 1 - 1e-6))
        self.classes_ = np.array([0, 1])
        self.estimator.fit(X, labels.astype(int), **kwargs)
        return self

    def predict_proba(self, X):
        raw = np.asarray(self.estimator.predict_proba(X), dtype=float)[:, 1]
        weight = float(np.clip(self.weight, 0.0, 1.0))
        blended = np.clip(weight * raw + (1 - weight) * self.prior_, 1e-6, 1 - 1e-6)
        return np.column_stack([1 - blended, blended])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


# ══════════════════════════════════════════════════════════════════════
# RISK — measured separately from any model probability
# ══════════════════════════════════════════════════════════════════════

def risk_metrics(prices: pd.DataFrame, window_years: int = 1) -> dict[str, float]:
    """Raw historical risk from a daily price series. Descriptive, not predictive."""
    empty = {"volatility": np.nan, "max_drawdown": np.nan,
             "downside_volatility": np.nan, "turnover": np.nan, "n_observations": 0}
    if prices is None or prices.empty or "close" not in prices.columns:
        return empty
    history = prices.sort_values("date").reset_index(drop=True)
    history = history[history["date"] >= history["date"].max() - pd.DateOffset(years=window_years)]
    if len(history) < 30:
        return empty

    close = history["close"].astype(float)
    returns = close.pct_change().dropna()
    if returns.empty:
        return empty
    downside = returns[returns < 0]

    turnover = np.nan
    if {"volume", "market_cap"} <= set(history.columns):
        daily = ((history["volume"].astype(float) * close) /
                 history["market_cap"].astype(float)).replace([np.inf, -np.inf], np.nan)
        if daily.notna().any():
            turnover = float(daily.median())

    return {
        "volatility": float(returns.std() * math.sqrt(TRADING_DAYS_PER_YEAR)),
        "max_drawdown": float((close / close.cummax() - 1.0).min()),
        "downside_volatility": (float(downside.std() * math.sqrt(TRADING_DAYS_PER_YEAR))
                                if len(downside) > 1 else np.nan),
        "turnover": turnover,
        "n_observations": int(len(history)),
    }


def moving_averages(prices: pd.DataFrame, windows=(50, 200)) -> pd.DataFrame:
    """Close plus one column per moving-average window, indexed by date."""
    if prices is None or prices.empty or "close" not in prices.columns:
        return pd.DataFrame()
    history = prices.sort_values("date").reset_index(drop=True)
    frame = pd.DataFrame({"date": history["date"],
                          "close": history["close"].astype(float)})
    for window in windows:
        frame[f"ma{window}"] = frame["close"].rolling(window, min_periods=window).mean()
    return frame


#: How wide the projected range has to be to hold the price the stated share of
#: the time. These are MEASURED on this project's own cached panel, not taken
#: from a normal distribution: every observation where 252 days of history were
#: available was projected forward and checked against what actually happened.
#:
#: The textbook multipliers are 1.00 for 68% and 2.00 for 95%. The 68% figure
#: holds up here — a 6-month band drawn at 1.00 covered 66.7% against a
#: theoretical 68.3% — but 2.00 covered only 87.5% rather than 95.4%. IDX
#: returns have far fatter tails than a bell curve, and a true 95% needs a
#: multiplier near 3.7.
#:
#: Only the 50% band ships. The wider ones are correctly calibrated — split by
#: volatility, the most volatile quarter of observations saw an 80% band catch
#: 74.9% at six months, so if anything it is narrow — but half this panel's
#: 12-month 80% ranges spanned more than five times bottom to top and MORA's
#: spanned sixty-five. A range that wide is an accurate statement and a useless
#: one, and printing it invited a reader to anchor on a number that meant
#: nothing. One band, honestly labelled, says more.
CONE_MULTIPLIERS: dict[int, dict[int, float]] = {
    126: {50: 0.65},
    252: {50: 0.75},
}
CONE_LOOKBACK = 252          # one year of daily returns behind each estimate

#: Above this the range stops being information. Flagged when the band's top is
#: more than three times its bottom, which on this panel is six companies of
#: twenty-four against a median span of 2.1x. MORA's 12-month 50% range runs
#: 6.1x. That is a correctly calibrated statement about a stock with 121%
#: annualised volatility and a useless one to plan around, and saying so is
#: better than drawing it narrower than the evidence supports.
CONE_MAX_USEFUL_SPAN = 3.0


def volatility_cone(prices: pd.DataFrame,
                    lookback: int = CONE_LOOKBACK) -> dict[str, Any]:
    """Where the price could sit in 6 and 12 months, from its own volatility.

    A range, never a target. The width comes from how much the stock has
    actually moved over the past year, scaled to the horizon by the
    square-root-of-time rule and then widened by the measured multipliers
    above. It is deliberately symmetric around the last close: this says how
    far the price might travel, not which way.
    """
    empty = {"available": False}
    if prices is None or prices.empty or "close" not in prices.columns:
        return empty
    close = prices.sort_values("date")["close"].astype(float).dropna().to_numpy()
    if len(close) < lookback + 1:
        return empty

    returns = np.diff(np.log(np.clip(close[-(lookback + 1):], 1e-9, None)))
    returns = returns[np.isfinite(returns)]
    daily = float(np.std(returns, ddof=1)) if returns.size > 2 else np.nan
    last = float(close[-1])
    if not np.isfinite(daily) or daily <= 0 or last <= 0:
        return empty

    cone = {"available": True, "last": last,
            "annual_volatility": daily * math.sqrt(TRADING_DAYS_PER_YEAR),
            "lookback": lookback, "bands": {}}
    cone["too_wide"] = {}
    for horizon, levels in CONE_MULTIPLIERS.items():
        sigma = daily * math.sqrt(horizon)
        cone["bands"][horizon] = {
            level: (last * math.exp(-k * sigma), last * math.exp(k * sigma))
            for level, k in levels.items()}
        low, high = cone["bands"][horizon][max(levels)]
        cone["too_wide"][horizon] = bool(low > 0
                                         and high / low > CONE_MAX_USEFUL_SPAN)
    return cone


def cone_path(cone: dict[str, Any], level: int = 50,
              steps: int = 36) -> pd.DataFrame:
    """The band traced step by step from today to the 12-month horizon.

    Drawn as a curve rather than straight lines between 0, 126 and 252 days,
    because the width grows with the square root of time and a straight line
    between the three anchors understates it in between. It also gives the
    chart a value at every point to report on hover, which three anchors
    cannot.
    """
    if not cone.get("available"):
        return pd.DataFrame()
    horizon = max(CONE_MULTIPLIERS)
    multiplier = CONE_MULTIPLIERS[horizon][level]
    daily = cone["annual_volatility"] / math.sqrt(TRADING_DAYS_PER_YEAR)
    days = np.linspace(0, horizon, steps + 1)
    # The multiplier itself drifts between the two calibrated horizons, so it
    # is interpolated rather than held at the 12-month value throughout.
    short, long = sorted(CONE_MULTIPLIERS)
    k = np.interp(days, [0, short, long],
                  [CONE_MULTIPLIERS[short][level],
                   CONE_MULTIPLIERS[short][level],
                   CONE_MULTIPLIERS[long][level]])
    width = k * daily * np.sqrt(days)
    return pd.DataFrame({"days": days,
                         "low": cone["last"] * np.exp(-width),
                         "high": cone["last"] * np.exp(width)})


def support_resistance(prices: pd.DataFrame, window: int = 10,
                       levels: int = 3, tolerance: float = 0.02) -> dict[str, list]:
    """Prices the market has repeatedly turned at, above and below today.

    A swing high is a bar whose high is the highest in the window either side
    of it, and a swing low the mirror of that. Swings within ``tolerance`` of
    each other are one level, and the levels that collected the most swings win
    — a price the market turned at four times says more than one it touched
    once.

    This is descriptive and has no horizon. A level is where the price has
    stopped before, not where it will stop; the further back it was set, the
    less anyone still remembers it.
    """
    empty = {"support": [], "resistance": []}
    if prices is None or prices.empty or "close" not in prices.columns:
        return empty
    history = prices.sort_values("date").reset_index(drop=True)
    high = (history["high"] if "high" in history.columns
            else history["close"]).astype(float)
    low = (history["low"] if "low" in history.columns
           else history["close"]).astype(float)
    if len(history) < window * 3:
        return empty

    swings = []
    for index in range(window, len(history) - window):
        around_high = high.iloc[index - window:index + window + 1]
        around_low = low.iloc[index - window:index + window + 1]
        if high.iloc[index] >= around_high.max():
            swings.append(float(high.iloc[index]))
        if low.iloc[index] <= around_low.min():
            swings.append(float(low.iloc[index]))
    if not swings:
        return empty

    # Compared against the cluster's own anchor, not its last member. Chaining
    # off the last member lets a dense run of swings walk a single cluster
    # across a huge range: BBCA produced one "level" holding 75 touches that
    # actually spanned from 7,000 to 11,000, which is not a level at all.
    clusters: list[list[float]] = []
    for price in sorted(swings):
        if clusters and price <= clusters[-1][0] * (1 + tolerance):
            clusters[-1].append(price)
        else:
            clusters.append([price])

    last = float(history["close"].iloc[-1])
    scored = sorted(({"price": float(np.median(c)), "touches": len(c)}
                     for c in clusters),
                    key=lambda level: (-level["touches"], -level["price"]))
    below = [l for l in scored if l["price"] < last]
    above = [l for l in scored if l["price"] > last]
    return {"support": sorted(below[:levels], key=lambda l: -l["price"]),
            "resistance": sorted(above[:levels], key=lambda l: l["price"])}


def trading_conditions(prices: pd.DataFrame,
                       recent: int = 25) -> dict[str, Any]:
    """How this stock is trading right now against its own past year.

    Three separate readings, deliberately not combined into one score. A
    composite invites the reading that a "fear and greed" dial invites, and
    that reading does not survive contact with this panel: bucketed by such a
    score, the pattern that appears is driven by SRAJ contributing a third of
    the extreme-greed observations and DSSA and BYAN supplying one commodity
    run, while the rank correlation with the forward six-month return is
    +0.017. Three honest gauges beat one number that means nothing.

    None of these is good or bad on its own. A stock near its 52-week high may
    be running or may be expensive, and heavy volume accompanies both panic and
    conviction.
    """
    empty = {"available": False}
    if prices is None or prices.empty or "close" not in prices.columns:
        return empty
    history = prices.sort_values("date").reset_index(drop=True)
    if len(history) < TRADING_DAYS_PER_YEAR:
        return empty

    year = history.tail(TRADING_DAYS_PER_YEAR)
    close = year["close"].astype(float)
    high = (year["high"] if "high" in year else year["close"]).astype(float)
    low = (year["low"] if "low" in year else year["close"]).astype(float)
    last = float(close.iloc[-1])
    peak, trough = float(high.max()), float(low.min())

    out = {"available": True, "last": last,
           "high_52w": peak, "low_52w": trough,
           "range_position": (100 * (last - trough) / (peak - trough)
                              if peak > trough else np.nan)}

    returns = np.diff(np.log(np.clip(close.to_numpy(), 1e-9, None)))
    baseline = float(np.std(returns, ddof=1)) if returns.size > 2 else np.nan
    latest = (float(np.std(returns[-recent:], ddof=1))
              if returns.size > recent else np.nan)
    out["volatility_ratio"] = (latest / baseline
                               if np.isfinite(baseline) and baseline > 0 else np.nan)

    if "volume" in year:
        volume = year["volume"].astype(float)
        average = float(volume.mean())
        out["volume_ratio"] = (float(volume.tail(recent).mean()) / average
                               if average > 0 else np.nan)
    else:
        out["volume_ratio"] = np.nan
    return out


def range_band(position: Any) -> str:
    value = _to_float(position)
    if not np.isfinite(value):
        return "Unavailable"
    if value >= 80: return "Near its 52-week high"
    if value <= 20: return "Near its 52-week low"
    return "Mid-range"


def activity_band(ratio: Any, quiet: float = 0.75, busy: float = 1.5) -> str:
    """Trading volume against this stock's own normal, not against the market."""
    value = _to_float(ratio)
    if not np.isfinite(value):
        return "Unavailable"
    if value >= busy: return "Heavier than usual"
    if value <= quiet: return "Quieter than usual"
    return "About normal"


def turbulence_band(ratio: Any, calm: float = 0.8, rough: float = 1.3) -> str:
    value = _to_float(ratio)
    if not np.isfinite(value):
        return "Unavailable"
    if value >= rough: return "More volatile than usual"
    if value <= calm: return "Calmer than usual"
    return "About normal"


def rsi_series(close: pd.Series, window: int = 14) -> pd.Series:
    """Wilder's RSI across the whole series, for charting."""
    values = pd.Series(close, dtype=float)
    delta = values.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / window, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / window, adjust=False).mean()
    strength = gain / loss.replace(0, np.nan)
    result = 100 - 100 / (1 + strength)
    # A stretch with no down days has infinite strength, which is RSI 100.
    result[(loss == 0) & (gain > 0)] = 100.0
    result.iloc[:window] = np.nan
    return result


def macd_series(close: pd.Series, fast: int = 12, slow: int = 26,
                signal: int = 9) -> pd.DataFrame:
    """MACD line, signal and histogram across the whole series, for charting."""
    values = pd.Series(close, dtype=float)
    line = (values.ewm(span=fast, adjust=False).mean()
            - values.ewm(span=slow, adjust=False).mean())
    smoothed = line.ewm(span=signal, adjust=False).mean()
    frame = pd.DataFrame({"macd": line, "signal": smoothed,
                          "histogram": line - smoothed})
    frame.iloc[:slow + signal] = np.nan
    return frame


def relative_strength_index(close: pd.Series, window: int = 14) -> float:
    """Wilder's RSI on the last bar. NaN until there is enough history."""
    values = pd.Series(close, dtype=float).dropna()
    if len(values) <= window:
        return np.nan
    delta = values.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / window, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / window, adjust=False).mean()
    last_gain, last_loss = float(gain.iloc[-1]), float(loss.iloc[-1])
    if not np.isfinite(last_gain) or not np.isfinite(last_loss):
        return np.nan
    if last_loss == 0:
        return 100.0 if last_gain > 0 else 50.0
    return float(100 - 100 / (1 + last_gain / last_loss))


def macd(close: pd.Series, fast: int = 12, slow: int = 26,
         signal: int = 9) -> dict[str, float]:
    """MACD on the last bar: line, signal, and the gap between them.

    Two exponential averages of different lengths; the line is their
    difference and the signal is a further average of that line. The histogram
    — line less signal — is what actually turns, so it is the number the
    dashboard reads for direction.
    """
    empty = {"macd": np.nan, "macd_signal": np.nan, "macd_histogram": np.nan}
    values = pd.Series(close, dtype=float).dropna()
    if len(values) < slow + signal:
        return empty
    line = (values.ewm(span=fast, adjust=False).mean()
            - values.ewm(span=slow, adjust=False).mean())
    smoothed = line.ewm(span=signal, adjust=False).mean()
    return {"macd": float(line.iloc[-1]),
            "macd_signal": float(smoothed.iloc[-1]),
            "macd_histogram": float(line.iloc[-1] - smoothed.iloc[-1])}


def macd_band(histogram: Any) -> str:
    number = _to_float(histogram)
    if not np.isfinite(number):
        return "Unavailable"
    return "Bullish" if number > 0 else "Bearish" if number < 0 else "Flat"


def technical_state(prices: pd.DataFrame) -> dict[str, Any]:
    """Descriptive trend indicators from the price series alone.

    These describe what the price has already done. They are NOT a forecast and
    they are NOT part of the model — they were tested as model features on this
    panel and did not earn a place. They sit beside the risk block, which makes
    the same promise: descriptive, not predictive. Presenting a moving-average
    crossover as a validated signal would be exactly the kind of claim the rest
    of this project refuses to make.
    """
    empty = {"available": False, "trend": "Unknown"}
    if prices is None or prices.empty or "close" not in prices.columns:
        return empty
    history = prices.sort_values("date").reset_index(drop=True)
    close = history["close"].astype(float)
    if len(close) < 60:
        return empty

    last = float(close.iloc[-1])
    ma50 = float(close.rolling(50).mean().iloc[-1]) if len(close) >= 50 else np.nan
    ma200 = float(close.rolling(200).mean().iloc[-1]) if len(close) >= 200 else np.nan
    peak = float(close.rolling(min(252, len(close))).max().iloc[-1])

    def change(offset: int) -> float:
        if len(close) <= offset or close.iloc[-1 - offset] <= 0:
            return np.nan
        return float(last / close.iloc[-1 - offset] - 1.0)

    # Four states, not three. Price above its 50-day but below its 200-day, and
    # price below its 50-day but above its 200-day, are both "neither trend" —
    # and collapsing them into one bucket loses the only interesting thing
    # about either. On this snapshot fifteen of nineteen tickers sit in the
    # first case, a market that fell hard and is bouncing; calling all of them
    # "sideways" said nothing, and calling them "recovering" says what happened.
    short = last > ma50 if np.isfinite(ma50) else None
    long = last > ma200 if np.isfinite(ma200) else None
    if short is None and long is None:
        trend = "Insufficient history"
    elif short is None or long is None:
        only = short if long is None else long
        trend = "Above average" if only else "Below average"
    elif short and long:
        trend = "Uptrend"
    elif short and not long:
        trend = "Recovering"
    elif long and not short:
        trend = "Weakening"
    else:
        trend = "Downtrend"

    return {
        "available": True,
        "last": last, "ma50": ma50, "ma200": ma200,
        "rsi14": relative_strength_index(close),
        **macd(close),
        "from_52w_high": (last / peak - 1.0) if peak > 0 else np.nan,
        "return_6m": change(HORIZON_TRADING_DAYS["6m"]),
        "return_12m": change(HORIZON_TRADING_DAYS["12m"]),
        "trend": trend,
    }


def rsi_band(value: Any) -> str:
    number = _to_float(value)
    if not np.isfinite(number):
        return "Unavailable"
    if number >= 70: return "Overbought"
    if number <= 30: return "Oversold"
    return "Neutral"


#: Which way each band points, so the dashboard can pick an arrow and a colour
#: without re-deriving the meaning of the word at the call site. Overbought is
#: a stretched market and reads as the downward case; oversold the reverse.
BAND_DIRECTION: dict[str, int] = {
    "Overbought": -1, "Bearish": -1, "Downtrend": -1, "Weakening": -1,
    "Oversold": 1, "Bullish": 1, "Uptrend": 1, "Recovering": 1,
    "Neutral": 0, "Flat": 0, "Unavailable": 0, "Insufficient history": 0,
    "Above average": 0, "Below average": 0,
}


#: Where each trend state sits on a red-to-green scale, 0 worst and 1 best.
#: Recovering and Weakening are the two halves of what used to be one
#: "sideways" bucket, and they are not equivalent: one is a price that has
#: climbed back above its short average while still under its long one, the
#: other is the same picture running the other way.
TREND_SCALE: dict[str, float] = {
    "Downtrend": 0.0,
    "Below average": 0.25,
    "Weakening": 0.35,
    "Recovering": 0.65,
    "Above average": 0.75,
    "Uptrend": 1.0,
}


def trend_position(band: str) -> float:
    """0 for the weakest state, 1 for the strongest, NaN when unknown."""
    return TREND_SCALE.get(band, np.nan)


def band_direction(band: str) -> int:
    """+1 up and green, -1 down and red, 0 neutral and grey."""
    return BAND_DIRECTION.get(band, 0)


def _band(value: Any, low: float, high: float) -> float:
    number = _to_float(value)
    if not np.isfinite(number) or high <= low:
        return np.nan
    return float(np.clip((number - low) / (high - low), 0, 1) * 100)


def _rank(value: float, population) -> float:
    values = np.array([v for v in population if v is not None and np.isfinite(v)])
    if not np.isfinite(value) or values.size == 0:
        return np.nan
    return float((values < value).mean() * 100)


def risk_score(metrics: dict[str, float], peers=None) -> dict[str, Any]:
    """Combine risk metrics into 0-100. Independent of any model probability.

    A stock with a high probability of a positive return can still be violently
    volatile, and NusaQuant refuses to blur those two things together.
    """
    if not metrics or metrics.get("n_observations", 0) < 30:
        return {"score": np.nan, "band": "Unknown", "components": {}, **(metrics or {})}

    peers = list(peers or [])
    components: dict[str, float] = {}
    if peers:
        components["volatility"] = _rank(metrics.get("volatility", np.nan),
                                         [p.get("volatility") for p in peers])
        deeper = _rank(-abs(metrics.get("max_drawdown", np.nan)),
                       [-abs(p.get("max_drawdown", np.nan)) for p in peers])
        components["max_drawdown"] = 100.0 - deeper if np.isfinite(deeper) else np.nan
        components["downside_volatility"] = _rank(
            metrics.get("downside_volatility", np.nan),
            [p.get("downside_volatility") for p in peers])
        # Thin trading is risky, so LOW turnover means HIGH liquidity risk.
        traded = _rank(metrics.get("turnover", np.nan), [p.get("turnover") for p in peers])
        components["liquidity"] = 100.0 - traded if np.isfinite(traded) else np.nan
    else:
        components["volatility"] = _band(metrics.get("volatility"), 0.15, 0.60)
        components["max_drawdown"] = _band(abs(metrics.get("max_drawdown", np.nan)), 0.15, 0.70)
        components["downside_volatility"] = _band(metrics.get("downside_volatility"), 0.10, 0.45)
        liquid = _band(metrics.get("turnover"), 0.0002, 0.01)
        components["liquidity"] = 100.0 - liquid if np.isfinite(liquid) else np.nan

    weight = total = 0.0
    for name, w in RISK_WEIGHTS.items():
        value = components.get(name, np.nan)
        if np.isfinite(value):
            total += w * value; weight += w
    score = total / weight if weight > 0 else np.nan
    return {"score": score, "band": risk_band(score), "components": components, **metrics}


def risk_band(score: float) -> str:
    if score is None or not np.isfinite(score):
        return "Unknown"
    return "Low" if score <= 33 else ("Medium" if score <= 66 else "High")


# ══════════════════════════════════════════════════════════════════════
# RELIABILITY
# ══════════════════════════════════════════════════════════════════════

def _norm_auc(auc) -> float:
    """0.50 (a coin flip) maps to 0, 1.00 maps to 100."""
    value = _to_float(auc)
    return np.nan if not np.isfinite(value) else float(np.clip((value - .5) / .5, 0, 1) * 100)


def _norm_pr(pr_auc, base_rate) -> float:
    """PR-AUC against the class base rate, which is its real floor.

    A PR-AUC of 0.65 sounds fine until you notice 65% of cases are positive.
    """
    value, rate = _to_float(pr_auc), _to_float(base_rate)
    if not np.isfinite(value):
        return np.nan
    if not np.isfinite(rate) or rate >= 1:
        return float(np.clip(value, 0, 1) * 100)
    return float(np.clip((value - rate) / max(1e-9, 1 - rate), 0, 1) * 100)


def _norm_brier(brier, base_rate, baseline_brier=None) -> float:
    """Calibration as a Brier SKILL score against the baseline model.

    The reference has to be something a model could actually have achieved.
    The obvious-looking choice, ``base_rate * (1 - base_rate)``, is the Brier
    score of a forecaster who already knows the validation base rate — an
    oracle. Measured against it, even a perfectly honest model that predicts
    the prior it learned in training scores zero, because the prior it learned
    and the rate that turned up are never quite the same number. That is what
    the old formula did, and it is why every calibration component in the
    shipped artifacts read exactly 0.

    The reference here is instead the ``DummyClassifier`` baseline's own
    out-of-sample Brier on the same folds: a real forecaster, fitted on the
    same information, scored on the same rows. Beating it is meaningful and
    possible. The oracle form survives only as a fallback for callers that do
    not have a baseline to hand.
    """
    value = _to_float(brier)
    if not np.isfinite(value):
        return np.nan
    reference = _to_float(baseline_brier)
    if not np.isfinite(reference):
        rate = _to_float(base_rate)
        reference = rate * (1 - rate) if np.isfinite(rate) else 0.25
    return np.nan if reference <= 0 else float(np.clip(1 - value / reference, 0, 1) * 100)


def _norm_stability(auc_std, roc_auc=None) -> float:
    """A model scoring 0.68 one year and 0.51 the next got lucky in one regime.

    Gated on the model discriminating at all, because otherwise this component
    pays out for degeneracy: a classifier that returns the same number for
    every stock has a fold-to-fold AUC standard deviation of exactly zero and
    collects a perfect 100 for it. Consistency is only a virtue once there is
    something being delivered consistently.
    """
    value = _to_float(auc_std)
    if not np.isfinite(value):
        return np.nan
    auc = _to_float(roc_auc)
    if np.isfinite(auc) and auc < MIN_EDGE_AUC:
        return np.nan
    return float(np.clip(1 - value / .10, 0, 1) * 100)


def has_measurable_edge(metrics: dict[str, Any]) -> bool:
    """Did this model out-rank a coin flip by a margin worth reporting?"""
    return bool(np.isfinite(_to_float(metrics.get("roc_auc")))
                and _to_float(metrics.get("roc_auc")) >= MIN_EDGE_AUC)


def reliability(metrics: dict[str, Any],
                baseline: dict[str, Any] | None = None) -> dict[str, Any]:
    """Turn out-of-sample metrics into a 0-100 score. Measured, never a vibe.

    ``baseline`` is the same metrics dict for the always-predict-the-prior
    model, measured on the same folds. Pass it: without it the calibration
    component falls back to an oracle reference no forecaster can beat.
    """
    base_rate = _to_float(metrics.get("base_rate", np.nan))
    baseline_brier = (baseline or {}).get("brier")
    components = {
        "roc_auc": _norm_auc(metrics.get("roc_auc")),
        "pr_auc": _norm_pr(metrics.get("pr_auc"), base_rate),
        "calibration": _norm_brier(metrics.get("brier"), base_rate, baseline_brier),
        "stability": _norm_stability(metrics.get("roc_auc_std"), metrics.get("roc_auc")),
    }
    weight = total = 0.0
    for name, w in RELIABILITY_WEIGHTS.items():
        value = components[name]
        if np.isfinite(value):
            total += w * value; weight += w
    score = total / weight if weight > 0 else np.nan

    # A model that cannot rank stocks must not be able to accumulate its way to
    # a reassuring label on calibration and consistency alone. Those are real
    # components, but they describe a well-behaved forecast of the base rate,
    # which is not the same claim as a signal.
    edge = has_measurable_edge(metrics)
    if not edge and np.isfinite(score):
        score = min(score, 49.9)
    return {"score": score, "label": reliability_label(score, edge),
            "components": components, "has_edge": edge}


def reliability_label(score: float, has_edge: bool = True) -> str:
    if score is None or not np.isfinite(score):
        return "Unknown"
    if not has_edge:
        return "No measurable edge"
    if score >= 80: return "High"
    if score >= 65: return "Moderate"
    if score >= 50: return "Limited"
    return "Weak"


def probability_band(probability: float, has_edge: bool = True) -> str:
    if probability is None or not np.isfinite(probability):
        return "Unavailable"
    if not has_edge:
        # Naming an "edge" the validation could not find would be the single
        # most misleading string in the product.
        return "No measurable edge — read as the historical base rate"
    if probability < .50: return "Lower probability"
    if probability < .60: return "Slight positive edge"
    if probability < .70: return "Moderate positive edge"
    if probability < .80: return "Strong positive edge"
    return "Very strong positive edge"


# ══════════════════════════════════════════════════════════════════════
# FORMATTING AND EXPLANATIONS (deterministic templates — no LLM)
# ══════════════════════════════════════════════════════════════════════

def format_rupiah(value: Any, compact: bool = True) -> str:
    number = _to_float(value)
    if not np.isfinite(number):
        return "—"
    if compact:
        for threshold, suffix in ((1e12, "T"), (1e9, "B"), (1e6, "M")):
            if abs(number) >= threshold:
                return f"Rp {number / threshold:,.2f}{suffix}"
    return f"Rp {number:,.0f}"


def format_percent(value: Any, decimals: int = 1) -> str:
    number = _to_float(value)
    return "—" if not np.isfinite(number) else f"{number * 100:.{decimals}f}%"


def format_multiple(value: Any) -> str:
    number = _to_float(value)
    return "—" if not np.isfinite(number) else f"{number:.2f}x"


def format_feature(name: str, value: Any) -> str:
    spec = FEATURE_BY_NAME.get(name)
    if spec is None:
        return "—"
    if spec.unit == "percent":
        return format_percent(value)
    if spec.unit == "multiple":
        return format_multiple(value)
    if spec.unit == "currency":
        return format_rupiah(value, compact=True)
    number = _to_float(value)
    # An em dash, matching every other formatter. A metric that does not apply
    # to this company — NPL outside a bank — reads as a dash, and the row's
    # Meaning column carries the reason so the dash is never the whole answer.
    return "—" if not np.isfinite(number) else f"{number:.2f}"


# Why a metric is absent, per metric. Every NaN in this project is deliberate
# — a ratio is dropped when it would be economically meaningless rather than
# quietly filled with zero — so a blank cell always has a real reason behind
# it and the interface should say what it is instead of printing a dash.
FEATURE_ABSENCE_REASON: dict[str, str] = {
    "pe": "Trailing 12-month earnings are zero or negative, so a P/E would be "
          "meaningless.",
    "ps": "Trailing 12-month revenue is zero or negative.",
    "pbv": "Book equity is zero or negative.",
    "pcf": "Trailing 12-month operating cash flow is zero or negative.",
    "ev_ebitda": "Needs interest-bearing debt and positive EBITDA. Many filings "
                 "report neither, and substituting total liabilities would make "
                 "this a different ratio for banks than for miners.",
    "eps": "Needs trailing earnings and a share count.",
    "rps": "Needs trailing revenue and a share count.",
    "cps": "Cash is not reported for this period, or the share count is unknown.",
    "bvps": "Needs book equity and a share count.",
    "cfps": "Needs trailing operating cash flow and a share count.",
    "der": "Liabilities are not reported for this period, or equity is zero or "
           "negative.",
    "roa": "Average assets over the period are zero or negative.",
    "roe": "Average equity over the period is zero or negative.",
    "gpm": "Gross profit is not reported. Banks and other financial issuers do "
           "not file a cost of revenue, so they have no gross margin.",
    "opm": "Operating profit is not reported for this period.",
    "npm": "Trailing 12-month revenue is zero or negative.",
    "gross_profit": "Not reported. Financial issuers do not file a cost of "
                    "revenue.",
    "dividend": "No trailing dividend in the last screen. Either the company "
                "pays none, or the universe has not been screened yet — run "
                "`python train.py --screen`.",
    "dpr": "Needs a trailing dividend and positive earnings.",
    "dividend_yield": "Needs a trailing dividend from the screener.",
    "ebitda": "Not reported for this period.",
}


def feature_absence_reason(name: str) -> str:
    """Why this cell is empty."""
    return FEATURE_ABSENCE_REASON.get(name, "Not reported for this period.")


def explain_probability(probability: float, horizon: str,
                        has_edge: bool = True) -> str:
    if probability is None or not np.isfinite(probability):
        return "A probability is not available for this company."
    months = "6" if horizon == "6m" else "12"
    percentage = probability * 100
    base = (f"The machine learning model estimates a {percentage:.0f}% probability "
            f"that this "
            f"stock's return will be positive over the next {months} months, "
            f"based on the financial information available to it. "
            f"It does not mean the stock will rise by {percentage:.0f}%.")
    if has_edge:
        return base
    return (base + " On out-of-sample validation this machine learning model did "
                   "not separate "
                   "winners from losers, so the number reflects how often "
                   "stocks in this universe rose historically rather than "
                   "anything specific to this company.")


def explain_risk(risk: dict[str, Any]) -> str:
    band = risk.get("band", "Unknown")
    if band == "Unknown":
        return "Not enough price history is available to measure historical risk."
    parts = [f"Historical risk is {band.lower()}."]
    if np.isfinite(_to_float(risk.get("volatility"))):
        parts.append(f"Annualised volatility has been {format_percent(risk['volatility'], 0)}.")
    if np.isfinite(_to_float(risk.get("max_drawdown"))):
        parts.append(f"The deepest drawdown was {format_percent(risk['max_drawdown'], 0)}.")
    return " ".join(parts)


def explain_reliability(label: str, horizon: str) -> str:
    months = "6" if horizon == "6m" else "12"
    if label == "Unknown":
        return "Out-of-sample reliability has not been measured."
    if label == "No measurable edge":
        return (f"Across purged walk-forward folds this machine learning model "
                f"did not rank "
                f"{months}-month winners above losers by more than chance. Its "
                f"probabilities are shrunk toward the historical base rate and "
                f"should be read as that base rate, not as a stock-specific "
                f"forecast. The constraint is the size of the panel, not the "
                f"choice of algorithm.")
    if label == "Weak":
        return (f"On {months}-month out-of-sample validation this machine learning "
                f"model provides "
                f"limited predictive separation. Treat its probabilities as weak "
                f"evidence rather than a signal.")
    return (f"On {months}-month out-of-sample validation this machine learning "
            f"model showed "
            f"{label.lower()} reliability.")


#: One tooltip per figure the dashboard puts a label on. They live here with
#: the rest of the prose because a reader deciding whether to trust a number
#: needs to know what it measures, and a label alone rarely says.
TOOLTIPS: dict[str, str] = {
    # — profile —
    "latest_close":
        "The most recent closing price in the data being shown. Check the date "
        "at the top of the page — it may not be today's market.",
    "market_cap":
        "What the whole company is worth at the current price: the share price "
        "times the number of shares. It is the size of the business as the "
        "market prices it, and every valuation ratio here is built from it.",
    "probability":
        "The estimated chance that this stock ends the period higher than it "
        "is today. It says nothing about how much it might rise, only whether "
        "it rises at all. Always read the reliability figure beside it before "
        "giving this number any weight.",

    # — technical state —
    "trend":
        "Price against its own 50- and 200-day averages. Above both is an "
        "uptrend and below both a downtrend; above the 50 but under the 200 is "
        "recovering, and the reverse is weakening. Descriptive of what the "
        "price has done, not a forecast.",
    "rsi":
        "Relative Strength Index over 14 days, on Wilder's smoothing. Above 70 "
        "is conventionally read as overbought and below 30 as oversold, with "
        "neutral in between. A stretched reading is not a signal to act on: "
        "prices can stay overbought for months.",
    "macd":
        "Moving Average Convergence Divergence, 12/26/9. The figure is the "
        "histogram — the MACD line less its signal line. Above zero the "
        "shorter average is pulling ahead of the longer one, below zero it is "
        "falling behind. The chart above shows all three.",
    "from_52w_high":
        "How far the last close sits below the highest close of the past 52 "
        "weeks. Zero means the stock is at its own one-year peak; -30% means "
        "it would need to rise about 43% to return there.",
    "return_6m":
        "Price change over the last 126 trading days — six months of actual "
        "IDX sessions, not calendar months. This is what already happened, and "
        "is not an input to the model.",
    "return_12m":
        "Price change over the last 252 trading days. As above: history, not "
        "a forecast, and not something the model reads.",

    # — outlook —
    "reliability":
        "How well these estimates held up when tested against years the model "
        "had never seen, scored 0 to 100. It weighs whether the model sorted "
        "winners above losers, whether its percentages meant what they said, "
        "and whether it kept doing so year after year. \"No measurable edge\" "
        "means it did no better than a coin flip, so its percentages are "
        "deliberately pulled close to the historical average and should not be "
        "read as a view on this particular company.",
    "roc_auc":
        "Pick a stock that rose and one that fell: this is how often the model "
        "gave the higher probability to the one that rose. 0.50 is a coin "
        "flip, 1.00 is perfect. Anything close to 0.50 means the ranking "
        "carries no information you could act on.",
    "folds":
        "How many separate past periods these scores were tested against. The "
        "model was only ever shown information available at the time, then "
        "judged on what happened next — so this is a record of past "
        "performance, not of fitting the past. More periods means a steadier "
        "measurement.",
    "data_quality":
        "How much of the information this estimate needs was actually "
        "available for this company. A low figure means it rests on gaps. It "
        "is not a measure of whether the estimate is any good — that is what "
        "reliability tells you.",

    # — historical risk —
    "risk_band":
        "How rough a ride this stock has given its holders, combining how "
        "sharply it moves, how far it has fallen from its peaks, and how "
        "easily it trades. Judged on price history alone and kept apart from "
        "the probabilities above: a stock can be likely to rise and still be "
        "punishing to hold.",
    "volatility":
        "Standard deviation of daily returns scaled to a year. Higher means "
        "the price moved more — in both directions, not only down.",
    "max_drawdown":
        "The deepest fall from a previous peak inside the window: what someone "
        "who bought at the worst moment would have sat through before any "
        "recovery.",
    "downside_volatility":
        "Volatility computed from losing days only. Two stocks can share an "
        "annualised volatility while one of them mostly moved upward, and this "
        "separates them.",
    "cone_range":
        "How far the price could drift by this horizon, from how much the "
        "stock has actually moved over the past year. It is where the price "
        "landed about half the time historically. It says how far, never which "
        "way, and it assumes the stock keeps moving as much as it has been.",
    "range_position":
        "Where the last close sits between the lowest and highest price of the "
        "past 52 weeks. 0% is at the low, 100% at the high. Neither end is good "
        "or bad on its own: a stock at its high may be running, or expensive.",
    "volume_ratio":
        "Average trading volume over the past month against this stock's own "
        "average for the year. Above 1.0 means more shares are changing hands "
        "than usual. Heavy trading accompanies panic and conviction alike, so "
        "it says attention, not direction.",
    "volatility_ratio":
        "How much the price has moved over the past month against its own "
        "average for the year. Above 1.0 means it is swinging harder than it "
        "normally does.",
    "support_resistance":
        "Prices this stock has repeatedly turned at before: a level is drawn "
        "where several swing highs or lows cluster together, and the more "
        "swings it collected the stronger the line. Descriptive and without a "
        "horizon — it marks where the price has stopped in the past, not where "
        "it will stop.",
    "turnover":
        "Median daily traded value as a share of market capitalisation. Thin "
        "trading is itself a risk: it is what makes a position hard to leave "
        "at the price on the screen.",

    # — ranking table —
    "rank":
        "Position by estimated chance of rising, highest first.",
    "ticker":
        "IDX ticker symbol.",
    "company":
        "Registered company name.",
    "risk_column":
        "Historical risk band, ranked against the other companies in this "
        "table rather than against a fixed threshold, so \"High\" means high "
        "relative to these peers.",
}


EXPLANATIONS = {
    "probability": ("The percentage is the estimated chance that this stock's "
                    "price ends the period higher than it is today. It is a "
                    "chance of going up, not a forecast of how much, and not a "
                    "price target."),
    "reliability": ("Reliability is how well these estimates held up when "
                    "tested against years the model had never seen. High "
                    "reliability means its past calls were consistent and its "
                    "percentages meant what they said. Low reliability means "
                    "they were not, and the number deserves little weight."),
    "risk": ("Risk summarises how sharply this stock has moved, how far it has "
             "fallen from its peaks, and how easily it trades. It describes "
             "what has already happened and carries no promise about what "
             "comes next."),
    "cone": ("<strong>How to read the shaded range.</strong> It shows how far "
             "this stock's price could drift over the next 6 and 12 months, "
             "based on how much it has actually moved over the past year. "
             "Hover anywhere inside it to read the range on that date. Half "
             "the time in the past, the price ended up inside this band."
             "<br><br><strong>It does not say which way.</strong> The range is "
             "the same size above and below today's price on purpose. Whether "
             "the stock rises or falls is a separate question, and the "
             "probability figures further down are what try to answer it."
             "<br><br><strong>It assumes the stock keeps moving as much as it "
             "has been.</strong> If the market goes quiet the real range will "
             "be narrower than this, and if it panics it will be wider. A "
             "range is not a promise, and prices can and do finish outside it."),
    "technical": ("These indicators describe what the price has already done. "
                  "They are not a forecast, and they are separate from the "
                  "probability estimates further down. Read them as context "
                  "for a decision, not as a signal to act on."),
    "data_quality": ("Data quality is how much of the information this "
                     "estimate needs was actually available for this company. "
                     "A low figure means the estimate rests on gaps. It says "
                     "nothing about whether the estimate itself is any good."),
}


DISCLAIMER = (
    "NusaQuant provides quantitative analysis to support research and "
    "decision-making. Model probabilities, forecasts, and other analyses are "
    "estimates and may be inaccurate; they are not guarantees of future "
    "outcomes and do not constitute financial advice. You are solely "
    "responsible for your own decisions and assume all associated risks.")

WELCOME = ("Understand the IDX market through data.\n\n"
           "NusaQuant combines fundamental financial features with machine "
           "learning to estimate the probability of positive stock returns over "
           "6 and 12 months.")
