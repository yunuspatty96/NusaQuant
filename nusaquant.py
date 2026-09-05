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
                *, timeout: int = 30, max_retries: int = 3) -> Any:
    """One GET against Sectors v2. Retries 429 and 5xx, never leaks the key."""
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
            time.sleep(float(wait) if wait and wait.isdigit() else 2 ** attempt)
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
    for column in ("close", "volume", "market_cap"):
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


def cache_universe(universe: pd.DataFrame, base: str = "data") -> None:
    """Cache the screener result too — otherwise a re-run still costs 1 credit."""
    if universe is not None and not universe.empty:
        universe.to_parquet(cache_dir(base) / "universe.parquet", index=False)


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
    unit: str = "ratio"          # "percent" | "multiple" | "ratio"


FEATURE_SCHEMA: tuple[FeatureSpec, ...] = (
    FeatureSpec("pe", "PE", "Valuation",
                "Price relative to trailing 12-month earnings.", "multiple"),
    FeatureSpec("pb", "PB", "Valuation",
                "Price relative to book value of equity.", "multiple"),
    FeatureSpec("ps", "PS", "Valuation",
                "Price relative to trailing 12-month sales.", "multiple"),
    FeatureSpec("roe", "ROE", "Profitability",
                "Return generated on shareholder equity.", "percent"),
    FeatureSpec("roa", "ROA", "Profitability",
                "Return generated from total assets.", "percent"),
    FeatureSpec("net_profit_margin", "Net Profit Margin", "Profitability",
                "Profit generated per unit of revenue.", "percent"),
    FeatureSpec("debt_to_equity", "Debt-to-Equity", "Leverage",
                "Total liabilities relative to shareholder equity.", "multiple"),
    FeatureSpec("earnings_growth_1y", "Earnings Growth 1Y", "Growth",
                "Trailing 12-month earnings versus a year earlier.", "percent"),
    FeatureSpec("revenue_growth_1y", "Revenue Growth 1Y", "Growth",
                "Trailing 12-month revenue versus a year earlier.", "percent"),
    FeatureSpec("accruals", "Accruals", "Earnings Quality",
                "Gap between reported profit and cash actually collected, "
                "relative to assets.", "percent"),
)

FEATURE_NAMES: list[str] = [f.name for f in FEATURE_SCHEMA]
FEATURE_BY_NAME = {f.name: f for f in FEATURE_SCHEMA}
assert len(FEATURE_NAMES) == 10, "The feature set is capped at ten."


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

    flows = ("revenue", "earnings", "operating_cash_flow")
    for column in (*flows, "total_assets", "total_equity"):
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

    panel["revenue_ttm"] = panel["revenue"].rolling(4, min_periods=4).sum()
    panel["earnings_ttm"] = panel["earnings"].rolling(4, min_periods=4).sum()
    panel["ocf_ttm"] = panel["operating_cash_flow"].rolling(4, min_periods=4).sum()
    panel["avg_equity"] = panel["total_equity"].rolling(4, min_periods=2).mean()
    panel["avg_assets"] = panel["total_assets"].rolling(4, min_periods=2).mean()
    panel["available_date"] = panel["report_date"] + pd.Timedelta(days=REPORTING_LAG_DAYS)
    return panel


def compute_features(panel: pd.DataFrame, market_cap: float | None,
                     index: int | None = None) -> dict[str, float]:
    """The ten-feature vector. THE single source of truth for both paths.

    The training script calls this at every historical observation; the app
    calls it once with the latest quarter. Because both go through here, a
    feature cannot mean one thing in training and another in production.
    """
    features = {name: np.nan for name in FEATURE_NAMES}
    if panel is None or panel.empty:
        return features
    position = len(panel) - 1 if index is None else index
    if position < 0 or position >= len(panel):
        return features

    row = panel.iloc[position]
    cap = _to_float(market_cap)
    revenue, earnings = row.get("revenue_ttm"), row.get("earnings_ttm")
    equity, assets = row.get("total_equity"), row.get("total_assets")

    # Valuation, built from market cap so share counts and splits never matter.
    # A PE on negative earnings is a category error, not a cheap stock -> NaN.
    features["pe"] = _ratio(cap, earnings, positive_denominator=True)
    features["pb"] = _ratio(cap, equity, positive_denominator=True)
    features["ps"] = _ratio(cap, revenue, positive_denominator=True)

    # Profitability. A negative ROE IS meaningful, so only a negative
    # denominator voids these.
    features["roe"] = _ratio(earnings, row.get("avg_equity"), positive_denominator=True)
    features["roa"] = _ratio(earnings, row.get("avg_assets"), positive_denominator=True)
    features["net_profit_margin"] = _ratio(earnings, revenue, positive_denominator=True)

    # The quarterly endpoint gives assets and equity but not liabilities, so
    # liabilities are inferred as (assets - equity). That makes this total
    # liabilities to equity, broader than interest-bearing debt to equity.
    if pd.notna(assets) and pd.notna(equity) and _to_float(equity) > 0:
        features["debt_to_equity"] = _ratio(_to_float(assets) - _to_float(equity), equity)

    # Profit a company reports but has not collected in cash is the classic
    # sign of aggressive accounting.
    ocf = row.get("ocf_ttm")
    if pd.notna(earnings) and pd.notna(ocf):
        features["accruals"] = _ratio(_to_float(earnings) - _to_float(ocf),
                                      row.get("avg_assets"), positive_denominator=True)

    # Growth: TTM against TTM four quarters back, so seasonality cancels.
    if position >= 4:
        past = panel.iloc[position - 4]
        features["earnings_growth_1y"] = _growth(earnings, past.get("earnings_ttm"))
        features["revenue_growth_1y"] = _growth(revenue, past.get("revenue_ttm"))

    return features


def features_frame(quarterly: pd.DataFrame, market_cap: float | None) -> pd.DataFrame:
    """One-row DataFrame with exactly the model's schema, in order."""
    panel = build_panel(quarterly)
    return pd.DataFrame([compute_features(panel, market_cap)], columns=FEATURE_NAMES)


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
            **compute_features(panel.iloc[: index + 1], market_cap, index),
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

    dataset[FEATURE_NAMES] = dataset[FEATURE_NAMES].replace([np.inf, -np.inf], np.nan)
    for column in ("pe", "pb", "ps"):
        dataset.loc[dataset[column] <= 0, column] = np.nan   # cannot be negative
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
    number = _to_float(value)
    return "—" if not np.isfinite(number) else f"{number:.2f}"


def explain_probability(probability: float, horizon: str,
                        has_edge: bool = True) -> str:
    if probability is None or not np.isfinite(probability):
        return "A probability is not available for this company."
    months = "6" if horizon == "6m" else "12"
    percentage = probability * 100
    base = (f"The model estimates a {percentage:.0f}% probability that this "
            f"stock's return will be positive over the next {months} months, "
            f"based on the financial information available to it. "
            f"It does not mean the stock will rise by {percentage:.0f}%.")
    if has_edge:
        return base
    return (base + " On out-of-sample validation this model did not separate "
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
        return (f"Across purged walk-forward folds this model did not rank "
                f"{months}-month winners above losers by more than chance. Its "
                f"probabilities are shrunk toward the historical base rate and "
                f"should be read as that base rate, not as a stock-specific "
                f"forecast. The constraint is the size of the panel, not the "
                f"choice of algorithm.")
    if label == "Weak":
        return (f"On {months}-month out-of-sample validation this model provides "
                f"limited predictive separation. Treat its probabilities as weak "
                f"evidence rather than a signal.")
    return (f"On {months}-month out-of-sample validation this model showed "
            f"{label.lower()} reliability.")


EXPLANATIONS = {
    "probability": ("The percentage is the model's estimated probability that the "
                    "return will be above 0% over the horizon. It is not a "
                    "guaranteed return and not a price target."),
    "reliability": ("Reliability describes how the model performed on historical "
                    "out-of-sample validation — higher means its past predictions "
                    "were more consistent and better calibrated."),
    "risk": ("Risk summarises historical volatility, drawdown, downside movement "
             "and liquidity. It describes what already happened and does not "
             "guarantee future risk."),
    "data_quality": ("Data quality is the share of the model's required inputs "
                     "available for this company. It is not model reliability."),
}

DISCLAIMER = ("NusaQuant provides quantitative analysis for research and decision "
              "support. Model probabilities are estimates, not guarantees or "
              "financial advice.")

WELCOME = ("Understand the IDX market through data.\n\n"
           "NusaQuant combines fundamental financial features with machine "
           "learning to estimate the probability of positive stock returns over "
           "6 and 12 months.")
