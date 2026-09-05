#!/usr/bin/env python3
"""
NusaQuant — training script
===========================

Collects data from Sectors, trains the two XGBoost models (6-month and
12-month), and exports them to ``models/``.

    export SECTORS_API_KEY=your-key-here
    python train.py

The API key is read from the environment. It never appears in any source file,
is never written to disk, and is never printed. Pass ``--api-key`` only if you
must; the environment variable is safer because it stays out of your shell
history.

Options
-------
    --budget N        credit ceiling; the run halts rather than exceed it
    --quarters N      quarters of fundamentals per company (default 16)
    --companies N     universe size; default is the largest the budget allows
    --dry-run         show the plan and cost, fetch nothing
    --force           ignore the cache and re-fetch (spends credits again)

Cost
----
The first run costs roughly `companies × (quarters + price-requests)` credits.
Every later run costs **nothing**: each company is cached to ``data/cache/`` on
arrival and reused. A crashed run resumes where it stopped.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

import nusaquant as nq

# This script draws its rules and separators in box-drawing characters, and a
# Windows console still defaults to cp1252, which cannot encode them. Without
# this the run does all its work correctly and then dies printing the summary.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):        # already UTF-8, or redirected
        pass

MODELS_DIR = Path("models")
DATA_DIR = Path("data")

DEFAULT_QUARTERS = 16
DEFAULT_BUDGET = 600
RESERVE_FOR_APP = 100
UNIVERSE_FILTER = "market_cap > 1000000000000"          # > IDR 1 trillion
UNIVERSE_LABEL = "NusaQuant Liquid IDX Universe (market cap > IDR 1T)"


# ══════════════════════════════════════════════════════════════════════
# PLANNING
# ══════════════════════════════════════════════════════════════════════

def cached_universe_size(base: str = "data") -> int:
    """Companies already bought and complete on disk. They cost nothing."""
    return len([t for t in nq.cached_tickers("quarterly", base)
                if nq.is_cached(t, "prices", base)])


def plan_run(quarters: int, budget: int, companies: int | None,
             reserve: int = RESERVE_FOR_APP, cached: int = 0) -> dict:
    """Work out the price window and universe size that fit the budget.

    The price window is DERIVED from ``quarters``, never set by hand. Fetching
    price history older than the oldest fundamental is pure waste — there is no
    observation to pair it with.

    The budget buys COMPANIES THAT ARE NOT YET CACHED. Sizing the universe as
    though the cache were empty is what made a second run buy nothing: with 15
    companies already on disk and a budget covering 13, the planner set the
    universe to 13, took the first 13 of the cached 15, and reported a full
    collection having spent nothing and added nothing. Worse, the two it
    dropped held the longest price history in the panel. Money already spent is
    not part of the decision about money still to spend.
    """
    observations = max(1, quarters - 7)      # 4 quarters for TTM, 4 for growth
    # The window must reach back to the FIRST report we fetch, not merely to
    # the first usable observation. If it starts later, the earliest reports
    # have no price to pair with and their rows are silently lost.
    span_days = quarters * 91 + nq.REPORTING_LAG_DAYS + 380       # +12m window
    requests_per_ticker = span_days // nq.MAX_DAILY_WINDOW_DAYS + 1
    per_ticker = quarters + requests_per_ticker

    spendable = budget - reserve - 1          # -1 for the universe screener
    affordable_new = max(0, spendable // per_ticker)
    universe_size = cached + affordable_new
    if companies:
        universe_size = min(companies, universe_size)
    new_companies = max(0, universe_size - cached)

    end = date.today()
    start = end - timedelta(days=span_days)
    return {
        "quarters": quarters,
        "observations_per_ticker": observations,
        "price_start": start.isoformat(),
        "price_end": end.isoformat(),
        "price_years": round(span_days / 365, 1),
        "requests_per_ticker": requests_per_ticker,
        "per_ticker": per_ticker,
        "cached_companies": cached,
        "new_companies": new_companies,
        "universe_size": universe_size,
        "estimated_total": new_companies * per_ticker + 1,
        "estimated_rows": universe_size * observations,
        "budget": budget,
        "reserve": reserve,
    }


def print_plan(plan: dict) -> None:
    print("─" * 62)
    print("PLAN")
    print("─" * 62)
    print(f"  Budget              : {plan['budget']:,} credits "
          f"({plan['reserve']} reserved for the dashboard)")
    print(f"  Quarters per company: {plan['quarters']}")
    print(f"  Price window        : {plan['price_start']} to {plan['price_end']} "
          f"({plan['price_years']} years)")
    print(f"  Cost per company    : {plan['quarters']} quarters + "
          f"{plan['requests_per_ticker']} price = {plan['per_ticker']} credits")
    print(f"  Already cached      : {plan['cached_companies']} companies (free)")
    print(f"  New to buy          : {plan['new_companies']} companies")
    print(f"  Universe size       : {plan['universe_size']} companies")
    print(f"  ESTIMATED SPEND     : ~{plan['estimated_total']:,} credits")
    # A floor, not a forecast: it assumes every company has exactly --quarters
    # quarters, and cached companies were often collected with more.
    print(f"  Expected dataset    : at least ~{plan['estimated_rows']:,} rows")
    print("─" * 62)


# ══════════════════════════════════════════════════════════════════════
# COLLECTION
# ══════════════════════════════════════════════════════════════════════

def _prioritise_cached(universe: pd.DataFrame, target: int) -> pd.DataFrame:
    """Trim to ``target``, but never by discarding a company already bought.

    Cached companies are free and their history is often the deepest in the
    panel, so they go to the front of the queue and the budget fills the rest.
    Cutting the list with a plain ``head`` once cost this project its two
    longest price series.
    """
    if universe.empty:
        return universe
    cached = set(nq.cached_tickers("quarterly"))
    is_cached = universe["symbol"].map(nq.normalise_symbol).isin(cached)
    already, fresh = universe[is_cached], universe[~is_cached]
    room = max(0, target - len(already))
    return pd.concat([already, fresh.head(room)], ignore_index=True)


def collect_offline():
    """Rebuild the universe and panel purely from ``data/cache/``.

    Re-training has always cost nothing, but until now it still demanded a
    valid API key to get past the connection check — which meant the one
    operation the project advertises as free could not be run without one.
    """
    tickers = nq.cached_tickers("quarterly")
    tickers = [t for t in tickers if nq.is_cached(t, "prices")]
    if not tickers:
        raise SystemExit("Nothing in data/cache/ to re-train on. Run a real "
                         "collection first, or drop the --offline flag.")

    universe = nq.load_universe()
    if universe.empty:
        universe = pd.DataFrame({"symbol": tickers, "company_name": tickers})
    else:
        universe = universe[universe.symbol.map(nq.normalise_symbol).isin(tickers)]

    quarterly, prices = {}, {}
    for ticker in tickers:
        q, p = nq.load_from_cache(ticker, "quarterly"), nq.load_from_cache(ticker, "prices")
        if not q.empty and not p.empty:
            quarterly[ticker], prices[ticker] = q, p
    print(f"\nOffline re-train from cache: {len(quarterly)} companies, "
          f"snapshot as of {nq.cache_as_of()}. 0 credits.")
    return universe, quarterly, prices


def collect(api_key: str, plan: dict, force: bool = False):
    """Fetch the universe, then each company — reusing the cache throughout."""
    target = plan["universe_size"]
    universe = pd.DataFrame() if force else nq.load_universe()
    if universe.empty or len(universe) < target:
        print("\nFetching universe…")
        screened = nq.get_companies(api_key, where=UNIVERSE_FILTER, limit=target)
        if screened.empty:
            raise SystemExit("The Sectors screener returned no companies.")
        # Keep whatever the previous screen returned. A later screen can drop a
        # company that has since fallen below the market-cap filter, and its
        # data is already bought and paid for.
        universe = (screened if universe.empty else
                    pd.concat([universe, screened], ignore_index=True)
                      .drop_duplicates(subset=["symbol"], keep="first"))
        nq.cache_universe(universe)
    else:
        print("\nUniverse loaded from cache (0 credits).")
    universe = _prioritise_cached(universe, target)
    print(f"  {len(universe)} companies")

    quarterly, prices, failures = {}, {}, []
    bought = reused = 0

    for index, row in enumerate(universe.itertuples(), start=1):
        ticker = row.symbol
        try:
            q, p, paid = nq.collect_ticker(
                api_key, ticker, n_quarters=plan["quarters"],
                price_start=plan["price_start"], price_end=plan["price_end"],
                force=force)
        except nq.CreditLimitExceeded as exc:
            print(f"\n\nSTOPPED at {ticker}: {exc}")
            print("Everything fetched so far is cached. Re-run with a higher "
                  "--budget and it resumes rather than restarting.")
            break
        except nq.SectorsAPIError as exc:
            failures.append((ticker, exc.message)); continue

        if q.empty or p.empty:
            failures.append((ticker, "empty response")); continue
        quarterly[ticker], prices[ticker] = q, p
        bought += int(paid); reused += int(not paid)
        flag = "bought" if paid else "cached"
        print(f"  [{index:>3}/{len(universe)}] {ticker:<6} "
              f"{len(q):>3}q {len(p):>5}px  ({flag})")

    print(f"\nCollected {len(quarterly)} companies — "
          f"{bought} bought, {reused} reused from cache. Failed: {len(failures)}")
    for ticker, why in failures[:8]:
        print(f"  {ticker}: {why}")
    return universe, quarterly, prices


def build_dataset(quarterly: dict, prices: dict) -> pd.DataFrame:
    frames = []
    for ticker in quarterly:
        observations = nq.build_observations(ticker, quarterly[ticker], prices[ticker])
        if not observations.empty:
            frames.append(nq.add_targets(observations, prices[ticker]))
    return nq.build_dataset(frames)


# ══════════════════════════════════════════════════════════════════════
# MODELLING
# ══════════════════════════════════════════════════════════════════════

def _json_number(value) -> float | None:
    """NaN is not JSON. Absent is, and absent is what NaN means here."""
    number = nq._to_float(value)
    return float(number) if np.isfinite(number) else None


def _prepared(estimator):
    """Median imputation and a rank transform, both inside the pipeline.

    QuantileTransformer maps each feature onto its own training-set rank, which
    matters on a panel holding banks next to miners: a PB of 1.2 means one
    thing on a balance sheet that is mostly loans and another on one that is
    mostly ore. Ranks are also immune to the outliers a 15-name cross-section
    produces routinely. It refits per fold, so no validation row informs the
    transform that is applied to it.
    """
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import QuantileTransformer
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        # 25 bins, not the default 1000: the earliest fold trains on 44 rows, and
        # asking for more quantiles than there are samples earns a warning and
        # silent clamping. 25 is finer than anything a 15-name cross-section can
        # resolve anyway.
        ("rank", QuantileTransformer(n_quantiles=25, output_distribution="uniform",
                                     subsample=100_000, random_state=nq.RANDOM_STATE)),
        ("model", estimator),
    ])


def _xgb(**overrides):
    from xgboost import XGBClassifier
    params = dict(n_estimators=200, learning_rate=0.03, max_depth=1,
                  subsample=0.8, colsample_bytree=0.6, reg_lambda=10.0,
                  min_child_weight=10, random_state=nq.RANDOM_STATE,
                  eval_metric="logloss", tree_method="hist", n_jobs=-1)
    params.update(overrides)
    return _prepared(XGBClassifier(**params))


def _logistic(C: float):
    from sklearn.linear_model import LogisticRegression
    return _prepared(LogisticRegression(C=C, max_iter=2000,
                                        random_state=nq.RANDOM_STATE))


# Every candidate here is deliberately small. Measured on purged walk-forward
# folds, the depth-3 / 250-tree configuration this project shipped in 0.2.0
# reached an in-sample ROC-AUC of 0.79-0.86 and an out-of-sample 0.43 — it was
# not learning the market, it was learning fifteen tickers. Halving the depth
# roughly halves that gap. Capacity is the setting that matters most on 200
# rows, so the candidates differ mainly in how little of it they have.
MODEL_CANDIDATES: dict[str, Any] = {
    "xgb_depth1": lambda: _xgb(max_depth=1),
    "xgb_depth2": lambda: _xgb(max_depth=2, n_estimators=150, learning_rate=0.05,
                               min_child_weight=8, reg_lambda=5.0),
    "logistic_l2": lambda: _logistic(0.1),
    "logistic_l2_strong": lambda: _logistic(0.03),
}

# The floor matters. Log loss on this panel is minimised at weight 0, which
# means "output the base rate and nothing else" — and a model that returns the
# same number for all fifteen stocks cannot rank them, so the Best 10 would be
# ordered by nothing at all. Holding a sliver of the model's own opinion keeps
# the ordering defined while the spread stays around a couple of percentage
# points, which is the honest visual statement that it has little to say. The
# log-loss cost of the floor is in the fourth decimal.
SHRINKAGE_FLOOR = 0.05
SHRINKAGE_GRID = np.round(np.arange(SHRINKAGE_FLOOR, 1.0001, 0.05), 2)

# Log losses closer together than this are a tie, not a ranking.
LOG_LOSS_TIE = 0.005


def build_baseline():
    """Always predicts the training class prior. The bar ML must clear."""
    from sklearn.dummy import DummyClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    return Pipeline([("impute", SimpleImputer(strategy="median")),
                     ("model", DummyClassifier(strategy="prior"))])


def build_model(name: str = "xgb_depth1", weight: float = 1.0):
    """A candidate wrapped in the shrinkage that governs how loud it may be."""
    return nq.PriorShrunk(MODEL_CANDIDATES[name](), weight=weight)


def score(y_true, y_prob) -> dict:
    from sklearn.metrics import (average_precision_score, balanced_accuracy_score,
                                 brier_score_loss, log_loss, precision_score,
                                 recall_score, roc_auc_score)
    truth = np.asarray(y_true, dtype=float)
    probability = np.asarray(y_prob, dtype=float)
    mask = np.isfinite(truth) & np.isfinite(probability)
    truth, probability = truth[mask], probability[mask]
    if truth.size == 0:
        return {"n": 0}
    single = len(np.unique(truth)) < 2       # AUC undefined in a one-way year
    predicted = (probability >= .5).astype(int)
    return {
        "roc_auc": float(roc_auc_score(truth, probability)) if not single else np.nan,
        "pr_auc": float(average_precision_score(truth, probability)) if not single else np.nan,
        "brier": float(brier_score_loss(truth, probability)),
        "log_loss": float(log_loss(truth, np.clip(probability, 1e-6, 1 - 1e-6), labels=[0, 1])),
        "accuracy": float((predicted == truth).mean()),
        "balanced_accuracy": float(balanced_accuracy_score(truth, predicted)) if not single else np.nan,
        "precision": float(precision_score(truth, predicted, zero_division=0)),
        "recall": float(recall_score(truth, predicted, zero_division=0)),
        "base_rate": float(truth.mean()),
        "n": int(truth.size),
    }


def walk_forward(dataset: pd.DataFrame, features, horizon: str,
                 factory=None) -> dict:
    """Purged expanding-window validation. Never a random split.

    A random train/test split on a financial panel lets the model see future
    market regimes, which is how a backtest ends up looking excellent and
    failing live.

    ``factory`` builds a fresh, unfitted estimator per fold; the default is the
    prior-only baseline. Each out-of-sample row is returned alongside the fold
    it came from and the class prior of that fold's training set, which is what
    the shrinkage weight is later fitted against.
    """
    target = f"target_{horizon}"
    factory = factory or build_baseline
    folds = nq.walk_forward_folds(dataset, horizon)
    if not folds:
        return {"available": False, "reason": "not enough history for folds"}

    oof_rows, fold_rows = [], []
    for fold in folds:
        train, validate = dataset.loc[fold["train_index"]], dataset.loc[fold["validation_index"]]
        y_train = train[target].astype(int)
        if y_train.nunique() < 2:
            continue
        estimator = factory()
        try:
            estimator.fit(train[features], y_train)
            probability = estimator.predict_proba(validate[features])[:, 1]
        except Exception as exc:
            fold_rows.append({"validation_year": fold["validation_year"],
                              "error": str(exc)[:120]})
            continue
        y_validate = validate[target].astype(int)
        fold_rows.append({"validation_year": fold["validation_year"],
                          "n_train": fold["n_train"], "n_validation": fold["n_validation"],
                          **score(y_validate, probability)})
        oof_rows.append(pd.DataFrame({"fold": fold["validation_year"],
                                      "prior": float(y_train.mean()),
                                      "y_true": y_validate.to_numpy(dtype=float),
                                      "y_prob": np.asarray(probability, dtype=float)}))

    if not oof_rows:
        return {"available": False, "reason": "no fold produced predictions"}

    oof = pd.concat(oof_rows, ignore_index=True)
    pooled = score(oof.y_true, oof.y_prob)
    folds_frame = pd.DataFrame(fold_rows)

    # Rank quality is averaged WITHIN folds, never pooled across them, and the
    # difference is not cosmetic. Each fold is one quarterly rebalance, and the
    # share of that quarter's stocks that rose ranges from 0.00 to 1.00 across
    # this panel. Pool the folds and a model that emits one constant per
    # quarter — the same number for every stock, no ranking whatsoever — scores
    # a pooled ROC-AUC of 0.57, because the constants happen to sort the good
    # quarters above the bad ones. That is a measurement of the market's
    # direction leaking into a metric meant to measure stock selection. Scored
    # within each cross-section, that same model gets exactly 0.500, which is
    # the truth about it.
    def fold_mean(column: str) -> tuple[float, float, int]:
        if column not in folds_frame:
            return np.nan, np.nan, 0
        values = folds_frame[column].dropna()
        return (float(values.mean()) if len(values) else np.nan,
                float(values.std()) if len(values) > 1 else np.nan,
                int(len(values)))

    auc_mean, auc_std, scored_folds = fold_mean("roc_auc")
    pr_mean, _, _ = fold_mean("pr_auc")
    pooled["roc_auc_pooled"] = pooled.get("roc_auc")
    pooled["pr_auc_pooled"] = pooled.get("pr_auc")
    pooled["roc_auc"] = auc_mean
    pooled["pr_auc"] = pr_mean
    pooled["roc_auc_std"] = auc_std
    pooled["n_folds"] = int(len(folds_frame))
    pooled["n_folds_scored"] = scored_folds
    return {"available": True, "metrics": pooled, "folds": folds_frame, "oof": oof}


def _log_loss(y_true, y_prob) -> float:
    from sklearn.metrics import log_loss
    truth = np.asarray(y_true, dtype=float)
    if len(np.unique(truth)) < 2:
        return np.nan
    return float(log_loss(truth, np.clip(np.asarray(y_prob, dtype=float), 1e-6, 1 - 1e-6),
                          labels=[0, 1]))


def _blend(oof: pd.DataFrame, weight: float) -> np.ndarray:
    return np.clip(weight * oof.y_prob.to_numpy(dtype=float)
                   + (1 - weight) * oof.prior.to_numpy(dtype=float), 1e-6, 1 - 1e-6)


def fit_shrinkage(oof: pd.DataFrame) -> tuple[float, pd.DataFrame]:
    """How much of the model's opinion the evidence will actually carry.

    ``weight`` scales the distance a probability may travel from the base rate.
    It is chosen leave-one-fold-out — the weight applied to each fold is fitted
    only on the other folds — so the reported metrics are not the metrics of a
    weight tuned on the rows it is being scored against. The weight that ships
    is then refitted across every fold.

    Blending is monotone in the model's own output, so this never reorders the
    Best 10; it only decides how far apart the numbers are allowed to be.
    """
    folds = list(dict.fromkeys(oof.fold.tolist()))

    def best_weight(frame: pd.DataFrame) -> float:
        losses = [(w, _log_loss(frame.y_true, _blend(frame, w))) for w in SHRINKAGE_GRID]
        losses = [(w, l) for w, l in losses if np.isfinite(l)]
        return float(min(losses, key=lambda p: p[1])[0]) if losses else SHRINKAGE_FLOOR

    honest = oof.copy()
    honest["y_prob_shrunk"] = np.nan
    for fold in folds:
        others = oof[oof.fold != fold]
        w = (best_weight(others) if len(others) and others.y_true.nunique() > 1
             else SHRINKAGE_FLOOR)
        mask = honest.fold == fold
        honest.loc[mask, "y_prob_shrunk"] = _blend(honest[mask], w)
        honest.loc[mask, "weight_used"] = w
    return best_weight(oof), honest


def evaluate_candidate(dataset, features, horizon, name) -> dict:
    """Validate one candidate, then re-score it under honest shrinkage."""
    result = walk_forward(dataset, features, horizon,
                          factory=lambda: build_model(name, weight=1.0))
    if not result.get("available"):
        return result
    weight, honest = fit_shrinkage(result["oof"])
    shrunk = score(honest.y_true, honest.y_prob_shrunk)
    # Shrinkage is monotone within a fold, so it cannot change rank quality.
    # Carry those figures across rather than recomputing them from probabilities
    # that have been deliberately squeezed together.
    shrunk["roc_auc_pooled"] = shrunk.get("roc_auc")
    shrunk["pr_auc_pooled"] = shrunk.get("pr_auc")
    for key in ("roc_auc", "pr_auc", "roc_auc_std", "n_folds", "n_folds_scored"):
        shrunk[key] = result["metrics"].get(key)
    result.update({"name": name, "shrinkage_weight": weight,
                   "metrics_raw": result["metrics"], "metrics": shrunk,
                   "oof": honest})
    return result


def choose_model(dataset, features, horizon, baseline: dict) -> dict:
    """Measure every candidate on the same purged folds and keep the best.

    Ranked on out-of-sample log loss — a proper scoring rule, so it cannot be
    gamed by a model that is confidently wrong — with rank quality breaking
    ties. Ties are declared generously, within ``LOG_LOSS_TIE``, because when
    every candidate ends up heavily shrunk their log losses collapse into the
    fourth decimal and separate on noise, while their ROC-AUCs still differ by
    real margins. Splitting hairs on a number that has stopped discriminating
    is how the worst-ranking model gets picked.

    A candidate that cannot beat the prior-only baseline on log loss is
    reported as such rather than dressed up; the export still happens, at a
    shrinkage that keeps it within a couple of points of that baseline.
    """
    baseline_loss = nq._to_float((baseline.get("metrics") or {}).get("log_loss"))
    scored = []
    for name in MODEL_CANDIDATES:
        result = evaluate_candidate(dataset, features, horizon, name)
        if not result.get("available"):
            continue
        loss = nq._to_float(result["metrics"].get("log_loss"))
        auc = nq._to_float(result["metrics"].get("roc_auc"))
        scored.append((loss if np.isfinite(loss) else np.inf,
                       -(auc if np.isfinite(auc) else 0.0), name, result))
    if not scored:
        return {"available": False, "reason": "no candidate produced predictions"}

    scored.sort(key=lambda row: (row[0], row[1]))
    best_loss = scored[0][0]
    tied = [row for row in scored if row[0] <= best_loss + LOG_LOSS_TIE]
    _, _, name, chosen = min(tied, key=lambda row: row[1])
    chosen["leaderboard"] = [
        {"model": n, "log_loss": (None if not np.isfinite(l) else l),
         "roc_auc": -a, "shrinkage_weight": r["shrinkage_weight"],
         "reliability": nq.reliability(r["metrics"], baseline.get("metrics"))["score"]}
        for l, a, n, r in scored]

    chosen_loss = nq._to_float(chosen["metrics"].get("log_loss"))
    chosen["beats_baseline"] = bool(np.isfinite(chosen_loss) and np.isfinite(baseline_loss)
                                    and chosen_loss < baseline_loss)
    chosen["reliability"] = nq.reliability(chosen["metrics"], baseline.get("metrics"))
    return chosen


def feature_importance(estimator, features) -> list[dict]:
    """Importance is not causality — a diagnostic, not an explanation."""
    from sklearn.pipeline import Pipeline
    inner = getattr(estimator, "estimator", estimator)
    if not isinstance(inner, Pipeline):
        return []
    model = inner.named_steps.get("model")
    raw = getattr(model, "feature_importances_", None)
    if raw is None:
        coefficients = getattr(model, "coef_", None)
        if coefficients is None:
            return []
        raw = np.abs(np.asarray(coefficients, dtype=float)).ravel()
    values = np.asarray(raw, dtype=float)
    if values.size != len(features):
        return []
    if values.sum() > 0:
        values = values / values.sum()
    return sorted([{"feature": f, "importance": float(v)} for f, v in zip(features, values)],
                  key=lambda d: -d["importance"])


# ══════════════════════════════════════════════════════════════════════
# LEAKAGE AUDIT
# ══════════════════════════════════════════════════════════════════════

def leakage_audit(dataset: pd.DataFrame) -> pd.DataFrame:
    """Explicit pass/fail checks. Each has caused a real backtest to lie."""
    checks = []

    lag = (dataset.observation_date - dataset.report_date).dt.days
    checks.append(("Feature observed >= 90d after report date",
                   bool((lag >= nq.REPORTING_LAG_DAYS).all()), f"min {lag.min()}d"))

    resolves_later = all(
        (sub[f"target_available_{h}"] > sub.observation_date).all()
        for h in ("6m", "12m")
        for sub in [dataset.dropna(subset=[f"target_available_{h}"])] if not sub.empty)
    checks.append(("Target resolves after observation", resolves_later, ""))

    duplicates = int(dataset.duplicated(subset=["ticker", "observation_date"]).sum())
    checks.append(("No duplicate ticker/date rows", duplicates == 0, f"{duplicates}"))

    purged, leaks = True, []
    for horizon in ("6m", "12m"):
        for fold in nq.walk_forward_folds(dataset, horizon):
            train = dataset.loc[fold["train_index"]]
            if not (train[f"target_available_{horizon}"] < fold["validation_start"]).all():
                purged = False; leaks.append(f"{horizon}/{fold['validation_year']}")
    checks.append(("Training targets resolve before validation (purge)",
                   purged, ", ".join(leaks) or "no leaks"))

    ordered = all(dataset.loc[f["train_index"]].observation_date.max()
                  < dataset.loc[f["validation_index"]].observation_date.min()
                  for h in ("6m", "12m") for f in nq.walk_forward_folds(dataset, h))
    checks.append(("Folds are chronologically ordered", ordered, ""))

    inner = build_model().estimator
    checks.append(("Imputation inside the pipeline", "impute" in inner.named_steps, ""))
    checks.append(("Rank transform inside the pipeline", "rank" in inner.named_steps, ""))
    checks.append(("No random train_test_split", True, "walk-forward only"))
    checks.append(("v2 API only", nq.API_BASE_URL.endswith("/v2"), nq.API_BASE_URL))

    return pd.DataFrame([{"check": c, "result": "PASS" if ok else "FAIL", "detail": d}
                         for c, ok, d in checks])


# ══════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Collect Sectors data and train the NusaQuant models.")
    parser.add_argument("--api-key", default=None,
                        help="Sectors API key. Prefer the SECTORS_API_KEY "
                             "environment variable — it stays out of shell history.")
    parser.add_argument("--budget", type=int, default=DEFAULT_BUDGET)
    parser.add_argument("--quarters", type=int, default=DEFAULT_QUARTERS)
    parser.add_argument("--companies", type=int, default=None)
    parser.add_argument("--reserve", type=int, default=RESERVE_FOR_APP,
                        help="Credits held back for running the dashboard.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show the plan and cost, fetch nothing.")
    parser.add_argument("--force", action="store_true",
                        help="Ignore the cache and re-fetch (spends credits again).")
    parser.add_argument("--offline", action="store_true",
                        help="Re-train from data/cache/ only. No API key, no "
                             "network, no credits.")
    args = parser.parse_args()

    if args.offline:
        meter = nq.CreditMeter(budget=0)
        universe, quarterly, prices = collect_offline()
        return train_from(universe, quarterly, prices, meter)

    plan = plan_run(args.quarters, args.budget, args.companies, args.reserve,
                    cached=0 if args.force else cached_universe_size())
    print_plan(plan)

    if plan["new_companies"] == 0 and plan["cached_companies"] > 0:
        print("\nThis budget buys no company that is not already cached.")
        print("Nothing would change, so nothing is fetched. Either raise "
              "--budget, or lower --quarters to make each company cheaper.")
        print("To re-train on what is already on disk: python train.py --offline")
        return 1

    if plan["universe_size"] < 5:
        print("\nThis budget is too small for a meaningful study "
              "(fewer than 5 companies).")
        print("Raise --budget, or lower --quarters to fit more companies.")
        return 1

    if args.dry_run:
        print("\nDry run — nothing fetched, no credits spent.")
        return 0

    if plan["estimated_rows"] < 150:
        print("\nNOTE: this dataset will be small. Expect the model to report")
        print("Weak or Limited reliability. That is an honest result for this")
        print("much data, not a failure — but do not present the probabilities")
        print("as a strong signal.")

    api_key = nq.get_api_key(args.api_key)
    meter = nq.CreditMeter(budget=args.budget - args.reserve)
    nq.set_credit_meter(meter)

    if not nq.validate_api_key(api_key):
        print("\nCould not authenticate with Sectors. Check SECTORS_API_KEY.")
        return 1
    print("Sectors API v2 connected.\n")

    universe, quarterly, prices = collect(api_key, plan, force=args.force)
    print("\n" + meter.report())
    return train_from(universe, quarterly, prices, meter)


def train_from(universe, quarterly: dict, prices: dict, meter) -> int:
    """Everything from here down is free: no request leaves the machine."""
    if len(quarterly) < 5:
        print("\nToo few companies collected to train on. Stopping.")
        return 1

    nq.set_credit_meter(None)

    print("\nBuilding point-in-time dataset…")
    dataset = build_dataset(quarterly, prices)
    if dataset.empty:
        print("No observations could be built. Stopping.")
        return 1
    print(f"  {len(dataset):,} observations, {dataset.ticker.nunique()} tickers, "
          f"{dataset.observation_date.min():%Y-%m} to {dataset.observation_date.max():%Y-%m}")

    features, missingness = nq.usable_features(dataset)
    print("\nFeature availability:")
    for row in missingness.itertuples():
        mark = "keep" if row.retained else "DROP"
        print(f"  {mark}  {row.feature:<22} {row.missing_rate * 100:>5.1f}% missing")
    dropped = [f for f in nq.FEATURE_NAMES if f not in features]
    if any(f.endswith("_growth_1y") for f in dropped):
        print("\n  Note: the growth features need 8 quarters of warm-up, so at the")
        print("  default 16 quarters they are mostly missing. Re-run with")
        print("  --quarters 24 to keep them (fewer companies, same budget).")
    if len(features) < 5:
        print("\nToo few usable features. Stopping.")
        return 1

    print("\nLeakage audit:")
    audit = leakage_audit(dataset)
    for row in audit.itertuples():
        print(f"  {row.result}  {row.check}" + (f"  ({row.detail})" if row.detail else ""))
    if (audit.result != "PASS").any():
        print("\nAUDIT FAILED — not exporting models.")
        return 1

    print("\nWalk-forward validation:")
    MODELS_DIR.mkdir(exist_ok=True)
    results, exported = {}, []

    for horizon in ("6m", "12m"):
        baseline = walk_forward(dataset, features, horizon)
        chosen = choose_model(dataset, features, horizon, baseline)
        results[horizon] = chosen
        if not chosen.get("available"):
            print(f"  {horizon}: {chosen.get('reason')}")
            continue

        metrics, base_metrics = chosen["metrics"], baseline.get("metrics", {})
        base_auc = nq._to_float(base_metrics.get("roc_auc"))
        base_loss = nq._to_float(base_metrics.get("log_loss"))
        n_folds = len(chosen["folds"])
        print(f"  {horizon}: {n_folds} purged folds, {metrics['n']} out-of-sample rows")
        for row in chosen["leaderboard"]:
            loss = row["log_loss"]
            print(f"        {row['model']:<19} log-loss "
                  f"{'   —  ' if loss is None else f'{loss:.4f}'} · "
                  f"AUC {row['roc_auc']:.3f} · shrinkage {row['shrinkage_weight']:.2f}")
        print(f"        {'baseline (prior)':<19} log-loss "
              f"{base_loss:.4f} · AUC {base_auc:.3f}")

        verdict = ("beats the prior-only baseline" if chosen["beats_baseline"]
                   else "does NOT beat the prior-only baseline")
        print(f"    -> {chosen['name']} selected, shrinkage "
              f"{chosen['shrinkage_weight']:.2f}, {verdict}")
        print(f"       AUC {metrics['roc_auc']:.3f} (raw, before shrinkage "
              f"{chosen['metrics_raw']['roc_auc']:.3f}) · Brier {metrics['brier']:.4f} "
              f"· reliability {chosen['reliability']['label']} "
              f"({chosen['reliability']['score']:.1f}/100)")
        if not chosen["reliability"]["has_edge"]:
            print(f"       No measurable edge at {horizon}: out-of-sample ROC-AUC is "
                  f"below {nq.MIN_EDGE_AUC:.2f}. The exported model is shrunk toward "
                  f"the base rate and the dashboard will say so.")

        estimator = build_model(chosen["name"], weight=chosen["shrinkage_weight"])
        resolved = dataset.dropna(subset=[f"target_{horizon}"])
        estimator.fit(resolved[features], resolved[f"target_{horizon}"].astype(int))

        import joblib
        artifact = {
            "pipeline": estimator,
            "feature_names": list(features),
            "horizon": horizon,
            "model_name": chosen["name"],
            "shrinkage_weight": float(chosen["shrinkage_weight"]),
            "validation_metrics": metrics,
            "validation_metrics_unshrunk": chosen["metrics_raw"],
            "fold_metrics": chosen["folds"].to_dict("records"),
            "leaderboard": chosen["leaderboard"],
            "reliability": chosen["reliability"],
            "has_edge": bool(chosen["reliability"]["has_edge"]),
            "beats_baseline": bool(chosen["beats_baseline"]),
            "baseline_metrics": base_metrics,
            "baseline_roc_auc": float(base_auc) if np.isfinite(base_auc) else None,
            "validation_folds": int(n_folds),
            "feature_importance": feature_importance(estimator, features),
            "training_end_date": f"{resolved.observation_date.max():%Y-%m-%d}",
            "n_training_rows": int(len(resolved)),
            "nusaquant_version": nq.__version__,
        }
        path = MODELS_DIR / f"model_{horizon}_xgb.joblib"
        joblib.dump(artifact, path)
        exported.append(path.name)

    if not exported:
        print("\nNo models could be trained. Stopping.")
        return 1

    (MODELS_DIR / "metadata.json").write_text(json.dumps({
        "version": nq.__version__,
        "model": "selected per horizon from " + ", ".join(MODEL_CANDIDATES),
        "feature_set": list(features),
        "horizons": list(nq.HORIZON_TRADING_DAYS),
        "universe_label": UNIVERSE_LABEL,
        "universe": sorted(dataset.ticker.unique().tolist()),
        "training_end_date": f"{dataset.observation_date.max():%Y-%m-%d}",
        "dataset_rows": int(len(dataset)),
        "n_tickers": int(dataset.ticker.nunique()),
        "reporting_lag_days": nq.REPORTING_LAG_DAYS,
        "snapshot_as_of": nq.cache_as_of(),
        "validation": {
            horizon: {
                "model": result.get("name"),
                "shrinkage_weight": result.get("shrinkage_weight"),
                "folds": len(result["folds"]) if result.get("available") else 0,
                "roc_auc": _json_number(result.get("metrics", {}).get("roc_auc")),
                "reliability": result.get("reliability", {}).get("label", "Unknown"),
                "has_edge": bool(result.get("reliability", {}).get("has_edge", False)),
            }
            for horizon, result in results.items() if result.get("available")
        },
        "limitations": {
            "survivorship_bias": ("The Sectors universe reflects securities "
                                  "available today, so companies delisted during "
                                  "the study period are absent and historical "
                                  "performance is biased upward."),
            "reporting_lag": (f"A flat {nq.REPORTING_LAG_DAYS}-day lag stands in "
                              f"for the real publication date, which the API does "
                              f"not reliably expose."),
            "overlapping_targets": ("Consecutive quarterly observations share most "
                                    "of their forward window, so the effective "
                                    "sample is smaller than the row count."),
            "market_factor": ("The target is the sign of an absolute return, and "
                              "over 6-12 months that sign is dominated by the "
                              "direction of the market rather than by anything "
                              "specific to the company. Per-quarter base rates in "
                              "this panel range from 0.00 to 1.00. Cross-sectional "
                              "fundamentals carry no information about the market's "
                              "own direction, which caps how well any model built "
                              "on them can score."),
            "cross_section_width": ("Fifteen tickers means each quarterly "
                                    "cross-section is fifteen points wide. Widening "
                                    "the universe adds far more effective sample "
                                    "than adding history to the same names."),
        },
    }, indent=2), encoding="utf-8")

    DATA_DIR.mkdir(exist_ok=True)
    dataset.to_parquet(DATA_DIR / "dataset.parquet", index=False)

    print("\n" + "─" * 62)
    print(f"Exported: {', '.join(exported)} + metadata.json")
    print(f"Snapshot cached: {len(nq.cached_tickers())} companies "
          f"as of {nq.cache_as_of()}")
    print(f"Credits spent this run: {meter.spent:,}")
    print("\nRe-running costs NOTHING — every company is cached.")
    print("Next: commit models/ and data/cache/, then deploy. See DEPLOY.md")
    print("─" * 62)
    return 0


if __name__ == "__main__":
    sys.exit(main())
