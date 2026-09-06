"""Full test of the rebuilt NusaQuant. SYNTHETIC fake API — no real IDX data."""
import sys, os, warnings, tempfile, shutil, subprocess, json
from pathlib import Path
SRC = Path(__file__).resolve().parent
HOME = Path.cwd()
WORK = Path(tempfile.mkdtemp())
for f in ("nusaquant.py", "train.py", "app.py"):
    shutil.copy(SRC / f, WORK / f)
(WORK / "models").mkdir(); (WORK / "data" / "cache").mkdir(parents=True)
os.chdir(WORK); sys.path.insert(0, str(WORK))
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, nusaquant as nq

fails = []
def check(n, c, x=""):
    print(("PASS " if c else "FAIL ") + n + (f"  {x}" if x else ""))
    if not c: fails.append(n)

CALLS = {"n": 0}
TICKERS = [f"TK{i:02d}" for i in range(16)]

def fake(path, api_key, params=None, **kw):
    CALLS["n"] += 1
    if _M[0] is not None: _M[0].charge(*nq._price_of(path, params or {}))
    if api_key != "GOODKEY": raise nq.SectorsAPIError("Invalid Sectors API key.", status=401)
    if path.startswith("subsectors"): return [{"sub_sector": "banks"}]
    if path.startswith("companies/"):
        lim = (params or {}).get("limit", 50)
        return {"results": [{"symbol": f"{t}.JK", "company_name": f"{t} Tbk"}
                            for t in TICKERS[:lim]], "pagination": {"has_next": False}}
    if path.startswith("company/report/"):
        return {"symbol": "X", "company_name": "X Tbk",
                "overview": {"sector": "Financials", "sub_sector": "banks",
                             "market_cap": 5e14, "last_close_price": 3390}}
    if path.startswith("financials/quarterly/"):
        t = path.split("/")[2]; r = np.random.default_rng(abs(hash(t)) % 9999)
        cumulative = abs(hash(t)) % 3 == 0
        out = []
        for i, d in enumerate(pd.date_range("2018-03-31", "2026-06-30", freq="QE")):
            base = 1e12 * (1.02 ** i) * (1 + r.normal(0, .05))
            rev = base * d.quarter if cumulative else base
            out.append({"symbol": t, "date": d.strftime("%Y-%m-%d"), "revenue": rev,
                        "earnings": rev * r.uniform(.05, .2),
                        "total_assets": base * r.uniform(8, 13),
                        "total_equity": base * r.uniform(3, 5),
                        "operating_cash_flow": rev * r.uniform(.06, .26)})
        n = (params or {}).get("n_quarters")
        return out[-n:] if n else out
    if path.startswith("daily/"):
        t = path.split("/")[1]
        s, e = pd.Timestamp(params["start"]), pd.Timestamp(params["end"])
        d = pd.bdate_range(s, e)
        if len(d) == 0: return []
        r = np.random.default_rng((abs(hash(t)) + s.toordinal()) % 99999)
        px = 3390 * np.exp(np.cumsum(r.normal(.0003, .016, len(d))))
        return [{"symbol": t, "date": x.strftime("%Y-%m-%d"), "close": float(p),
                 "volume": int(2e6), "market_cap": float(p * 1.5e11)} for x, p in zip(d, px)]
    return []

_M = [None]
nq.api_request = fake

# ── 1. Ten features, no more ─────────────────────────────────────────────
check("metric schema covers seven categories", len(nq.CATEGORY_ORDER) == 7,
      ", ".join(nq.CATEGORY_ORDER))
check("only scale-free ratios are modelled",
      all(nq.FEATURE_BY_NAME[f].unit in ("multiple", "percent", "ratio")
          for f in nq.FEATURE_NAMES), str(nq.FEATURE_NAMES))
check("rupiah amounts are shown but never modelled",
      not any(f.modelled for f in nq.FEATURE_SCHEMA if f.unit == "currency"))
# The banking ratios stay out: nothing this project fetches can produce them.
# Dividends came back once the screener proved able to supply them.
check("bank-only ratios are not listed",
      not any(f.name in ("npl", "ldr", "nim") for f in nq.FEATURE_SCHEMA))
# A screener snapshot must never reach training. Feeding today's trailing yield
# to a 2022 observation is look-ahead, so the separation is asserted, not
# assumed: compute_features is the only path the training set travels.
check("snapshot metrics are never modelled",
      not any(f.modelled for f in nq.FEATURE_SCHEMA if not f.point_in_time))
_panel = nq.build_panel(nq.load_from_cache(TICKERS[0], "quarterly"))
_computed = nq.compute_features(_panel, 1e14, close=1000)
check("compute_features leaves snapshot metrics empty",
      all(not np.isfinite(nq._to_float(_computed.get(f.name)))
          for f in nq.FEATURE_SCHEMA if not f.point_in_time))
check("no ensemble leftovers", not hasattr(nq, "calculate_model_agreement"))

# ── 2. API key never required in code ────────────────────────────────────
os.environ.pop("SECTORS_API_KEY", None)
try:
    nq.get_api_key(); check("missing key raises with guidance", False)
except RuntimeError as e:
    check("missing key raises with guidance", "export SECTORS_API_KEY" in str(e))
os.environ["SECTORS_API_KEY"] = "GOODKEY"
check("key read from environment", nq.get_api_key() == "GOODKEY")
check("explicit key wins", nq.get_api_key("OTHER") == "OTHER")
src = (WORK / "nusaquant.py").read_text(encoding="utf-8") + (WORK / "train.py").read_text(encoding="utf-8") + (WORK / "app.py").read_text(encoding="utf-8")
import re
check("no hard-coded key literal in source",
      not re.search(r'api_key\s*=\s*["\'][A-Za-z0-9_\-]{12,}["\']', src))
check("key never printed", not re.search(r'print\([^)]*api_key', src))

# ── 3. Planner ───────────────────────────────────────────────────────────
import train
plan = train.plan_run(16, 600, None)
check("plan fits the budget", plan["estimated_total"] <= 600 - plan["reserve"],
      f"{plan['estimated_total']} credits")
check("plan yields a usable universe", plan["universe_size"] >= 10,
      f"{plan['universe_size']} companies")
check("price window covers every fetched report",
      plan["price_years"] * 365 > plan["quarters"] * 91,
      f"{plan['price_years']}y for {plan['quarters']}q")
print(f"      -> plan: {plan['universe_size']} companies, "
      f"~{plan['estimated_total']} credits, ~{plan['estimated_rows']} rows")

# ── 4. Dry run spends nothing ────────────────────────────────────────────
CALLS["n"] = 0
result = subprocess.run([sys.executable, "train.py", "--dry-run", "--budget", "600"],
                        capture_output=True, text=True, cwd=WORK)
check("dry run exits cleanly", result.returncode == 0, result.stderr[-200:])
check("dry run shows the plan", "ESTIMATED SPEND" in result.stdout)
check("dry run mentions no credits spent", "no credits spent" in result.stdout)

# ── 5. Real training run ─────────────────────────────────────────────────
_M[0] = nq.CreditMeter(budget=500)
CALLS["n"] = 0
universe, quarterly, prices = train.collect("GOODKEY", plan)
check("collected the planned universe", len(quarterly) == plan["universe_size"],
      f"{len(quarterly)}")
first_spend = _M[0].spent
check("spend within budget", first_spend <= 500, f"{first_spend} credits")
print(f"      -> first collection: {first_spend} credits for {len(quarterly)} companies")

_M[0] = nq.CreditMeter(budget=500); CALLS["n"] = 0
train.collect("GOODKEY", plan)
check("RE-RUN COSTS ZERO", _M[0].spent == 0 and CALLS["n"] == 0,
      f"{_M[0].spent} credits, {CALLS['n']} calls")

# crash recovery
for t in TICKERS[10:]:
    for k in ("quarterly", "prices"):
        p = nq._cache_path(t, k)
        if p.exists(): p.unlink()
_M[0] = nq.CreditMeter(budget=500)
train.collect("GOODKEY", plan)
check("resume costs only the missing ones", 0 < _M[0].spent < first_spend * 0.6,
      f"{_M[0].spent} vs {first_spend}")

# ── 6. Training is free ──────────────────────────────────────────────────
_M[0] = nq.CreditMeter(budget=0); CALLS["n"] = 0
quarterly = {t: nq.load_from_cache(t, "quarterly") for t in nq.cached_tickers()}
prices = {t: nq.load_from_cache(t, "prices") for t in nq.cached_tickers()}
dataset = train.build_dataset(quarterly, prices)
check("dataset built", len(dataset) > 100, f"{len(dataset)} rows")
check("TRAINING SPENDS ZERO CREDITS", _M[0].spent == 0 and CALLS["n"] == 0)
_M[0] = None

features, missing = nq.usable_features(dataset)
check("features retained", 5 <= len(features) <= 10, f"{len(features)}: {features}")

audit = train.leakage_audit(dataset)
check("leakage audit all pass", (audit.result == "PASS").all(),
      audit[audit.result != "PASS"].to_string())

LABELS = {"High", "Moderate", "Limited", "Weak", "Unknown", "No measurable edge"}
for h in ("6m", "12m"):
    b = train.walk_forward(dataset, features, h)                     # prior baseline
    r = train.choose_model(dataset, features, h, b)
    check(f"walk-forward {h} runs", r.get("available"), r.get("reason", ""))
    if r.get("available"):
        check(f"{h} reliability labelled", r["reliability"]["label"] in LABELS)
        check(f"{h} more than two purged folds", len(r["folds"]) > 2, f"{len(r['folds'])}")
        check(f"{h} shrinkage within bounds",
              train.SHRINKAGE_FLOOR <= r["shrinkage_weight"] <= 1.0,
              f"{r['shrinkage_weight']}")
        check(f"{h} candidate leaderboard recorded", len(r["leaderboard"]) > 1)
        check(f"{h} leaderboard names its feature policy",
              all(row["features"] in train.FEATURE_POLICIES
                  for row in r["leaderboard"]))
        # A wide spread of probabilities is a claim about ranking. If the
        # validation could not establish ranking, the spread must not be there.
        if not r["reliability"]["has_edge"]:
            check(f"{h} edgeless model is held near the base rate",
                  r["shrinkage_weight"] <= train.NO_EDGE_SHRINKAGE_CAP,
                  f"weight {r['shrinkage_weight']}")

# ── 6b. The information-coefficient screen ─────────────────────
# Built by hand rather than sampled, because a random column is not a control
# here: on a cross-section this narrow a random column scores |IC| around 0.05
# by itself, which is exactly why MIN_FEATURE_IC is where it is. So the probes
# are deterministic. "steady" ranks the same way every quarter. "flips" ranks
# perfectly one quarter and perfectly backwards the next, which averages to
# zero while POOLING every quarter together would score it near perfect — the
# one test that would catch the screen being rewritten to pool.
rows = []
for q in range(12):
    for i in range(12):
        rows.append({"observation_date": pd.Timestamp("2021-01-01")
                     + pd.DateOffset(months=3 * q),
                     "forward_return_6m": float(i),
                     "steady": float(i),
                     "flips": float(i if q % 2 == 0 else -i),
                     "never_varies": 1.0})
probe = pd.DataFrame(rows)

check("IC finds a ratio that ranks every quarter",
      nq.information_coefficient(probe, "steady", "6m") > 0.99,
      f"{nq.information_coefficient(probe, 'steady', '6m'):.3f}")
check("IC averages quarters instead of pooling them",
      abs(nq.information_coefficient(probe, "flips", "6m")) < 1e-9,
      f"{nq.information_coefficient(probe, 'flips', '6m'):.3f}")
check("IC is undefined for a constant",
      not np.isfinite(nq.information_coefficient(probe, "never_varies", "6m")))

probes = ["steady", "flips", "never_varies"]
kept = nq.screen_features(probe, probes, "6m", keep_at_least=1)
check("screen keeps the ratio that ranks", "steady" in kept)
check("screen drops the ratio that nets out to nothing", "flips" not in kept, str(kept))
check("screen keeps a ratio it could not measure rather than guessing",
      "never_varies" in kept, str(kept))
check("screen never empties itself",
      len(nq.screen_features(probe, ["flips"], "6m", keep_at_least=3)) == 1)
check("screen preserves column order",
      nq.screen_features(probe, probes, "6m") == probes)

# The bar has to sit above what a random column scores on THIS panel, or the
# screen is decoration. Measured here rather than asserted in a comment.
sized = dataset.dropna(subset=["forward_return_6m"]).copy()
rng = np.random.default_rng(0)
floor = []
for _ in range(300):
    sized["_random"] = rng.normal(size=len(sized))
    value = nq.information_coefficient(sized, "_random", "6m")
    if np.isfinite(value):
        floor.append(abs(value))
_floor = float(np.median(floor))
# A band, not a strict inequality. The bar is set AT the noise floor rather
# than safely above it, so demanding bar >= floor asks a deliberately
# borderline comparison to come out the same way every run, and the floor is
# itself an estimate that moves in the third decimal between panels. What this
# needs to catch is a regression to the original 0.02, which screened nothing
# whatsoever because a random column clears 0.02 nearly every time. Half the
# floor is comfortably below the bar and comfortably above that mistake.
check("MIN_FEATURE_IC is set at the panel's own noise floor",
      nq.MIN_FEATURE_IC >= 0.5 * _floor,
      f"bar {nq.MIN_FEATURE_IC} vs median |IC| of a random column {_floor:.3f}")

# The screen must read the slice it is handed and nothing else, or the fold
# protocol is decorative: fitted on early quarters, judged on later ones. Two
# real slices can legitimately agree, so agreement proves nothing and the probe
# is built to force a disagreement: "fades" ranks perfectly for the first half
# of the history and inverts for the second, netting to zero over the whole.
fades = probe.copy()
half = fades.observation_date.median()
fades["fades"] = np.where(fades.observation_date <= half,
                          fades["forward_return_6m"], -fades["forward_return_6m"])
early = fades[fades.observation_date <= half]
check("screen answers from the slice it is given, not the panel",
      "fades" in nq.screen_features(early, ["steady", "fades"], "6m", keep_at_least=1)
      and "fades" not in nq.screen_features(fades, ["steady", "fades"], "6m",
                                            keep_at_least=1),
      f"early {nq.screen_features(early, ['steady', 'fades'], '6m', keep_at_least=1)} "
      f"vs all {nq.screen_features(fades, ['steady', 'fades'], '6m', keep_at_least=1)}")

# A constant forecaster ranks nothing, so it must not be credited with an edge
# or with perfect stability — the two ways the old scoring flattered degeneracy.
flat = nq.reliability({"roc_auc": 0.50, "pr_auc": 0.45, "brier": 0.25,
                       "base_rate": 0.45, "roc_auc_std": 0.0, "n": 100})
check("constant predictor gets no edge", flat["label"] == "No measurable edge",
      flat["label"])
check("constant predictor earns no stability credit",
      not np.isfinite(flat["components"]["stability"]),
      str(flat["components"]["stability"]))
check("no-edge model cannot claim a probability band",
      nq.probability_band(0.85, has_edge=False).startswith("No measurable edge"))
check("calibration scored against the baseline, not an oracle",
      nq.reliability({"roc_auc": 0.70, "brier": 0.20, "base_rate": 0.50,
                      "roc_auc_std": 0.02},
                     baseline={"brier": 0.25})["components"]["calibration"] > 0)

# ── 7. Full CLI end to end (in-process, so the fake API applies) ────────
import io, contextlib
argv = sys.argv[:]
sys.argv = ["train.py", "--budget", "600"]
buffer = io.StringIO()
try:
    with contextlib.redirect_stdout(buffer):
        code = train.main()
finally:
    sys.argv = argv
output = buffer.getvalue()
print("      -> train.main() exit:", code)
if code != 0: print(output[-2500:])
check("train.py completes", code == 0)
check("exports both models", (WORK/"models"/"model_6m_xgb.joblib").exists()
      and (WORK/"models"/"model_12m_xgb.joblib").exists())
check("writes metadata", (WORK/"models"/"metadata.json").exists())
import joblib as _joblib
for h in ("6m", "12m"):
    art = _joblib.load(WORK/"models"/f"model_{h}_xgb.joblib")
    check(f"{h} artifact records its feature policy",
          art["feature_policy"] in train.FEATURE_POLICIES)
    check(f"{h} artifact ships only screened features",
          set(art["feature_names"]) <= set(art["feature_pool"]))
    check(f"{h} artifact records an IC per candidate ratio",
          set(art["feature_ic"]) == set(art["feature_pool"]))
    if not art["has_edge"]:
        check(f"{h} exported weight respects the no-edge cap",
              art["shrinkage_weight"] <= train.NO_EDGE_SHRINKAGE_CAP,
              f"weight {art['shrinkage_weight']}")
check("reports baseline comparison", "baseline" in output)
check("says re-running is free", "costs NOTHING" in output)
check("prints leakage audit", "Leakage audit" in output)
check("prints feature availability", "Feature availability" in output)

# ── 8. App: zero credits, no API key ─────────────────────────────────────
import streamlit as _st
from streamlit.testing.v1 import AppTest
_st.cache_resource.clear(); _st.cache_data.clear()
CALLS["n"] = 0
at = AppTest.from_file(str(WORK/"app.py"), default_timeout=240)
at.run()
# Radios are found by label, not index: adding the chart-style toggle shifted
# every index and a positional lookup fails silently at the wrong widget.
def radio(app, label):
    return next(r for r in app.radio if r.label == label)

# Every view must actually open. Renaming a sidebar label twice left the
# dispatch matching a string nothing produced any more, and selecting Sector
# Ranking silently opened Top Picks instead — no error, just the wrong page.
import importlib.util as _ilu
_spec = _ilu.spec_from_file_location("appmod", WORK / "app.py")
appmod = _ilu.module_from_spec(_spec)
try:
    _spec.loader.exec_module(appmod)
except Exception:
    pass
for _mode in (appmod.MODE_SINGLE, appmod.MODE_PICKS, appmod.MODE_PORTFOLIO):
    _probe = AppTest.from_file(str(WORK / "app.py"), default_timeout=240)
    _probe.run()
    radio(_probe, "Analysis").set_value(_mode).run()
    _heads = [str(h.value) for h in _probe.markdown if "nq-sec" in str(h.value)]
    _opened = any(_mode in h for h in _heads) and not _probe.exception
    check(f"view opens: {_mode[:34]}", _opened,
          "" if _opened else (str(_probe.exception)[:120] if _probe.exception
                              else "the heading for this view never rendered"))

# "Universe size N" claims the N largest by market cap. It used to take them
# alphabetically, so the claim was false and nothing said so.
_caps = appmod.snapshot_market_caps()
if len(_caps) >= 3:
    _ordered = appmod.by_market_cap(
        pd.DataFrame({"symbol": sorted(_caps)}))["symbol"].tolist()
    _values = [_caps[t] for t in _ordered if t in _caps]
    check("universe is ordered largest market cap first",
          _values == sorted(_values, reverse=True),
          ", ".join(_ordered[:5]))

# Every figure the dashboard labels should be able to say what it measures.
# A label alone rarely does, and these are numbers people act on.
_untipped = [m.label for m in at.metric if not m.help]
check("every metric carries a tooltip", not _untipped, ", ".join(_untipped[:4]))

# The projected range is a measured claim, not a decoration, so its shape is
# asserted: symmetric in log space around the last close, wider at 12 months
# than at 6, wider at 80% than at 50%, and absent without a year of prices.
_cone = nq.volatility_cone(nq.load_from_cache(TICKERS[0], "prices"))
check("projected range available", _cone.get("available"))
if _cone.get("available"):
    _last = _cone["last"]
    _w = {(h, l): (hi / lo) for h, lv in _cone["bands"].items()
          for l, (lo, hi) in lv.items()}
    check("wider at 12 months than at 6", _w[(252, 50)] > _w[(126, 50)])
    # One band only: the wider ones were correct but too broad to be read.
    check("only the 50% band is produced",
          set(nq.CONE_MULTIPLIERS[126]) == {50},
          str(sorted(nq.CONE_MULTIPLIERS[126])))
    check("range is symmetric around the last close in log space",
          all(abs(np.log(hi / _last) + np.log(lo / _last)) < 1e-9
              for lv in _cone["bands"].values() for lo, hi in lv.values()))
check("no projected range without a year of prices",
      not nq.volatility_cone(
          nq.load_from_cache(TICKERS[0], "prices").head(120)).get("available"))

# Support and resistance sit either side of the last close by construction,
# and a cluster must not chain its way across a wide range: comparing each
# swing to the cluster's last member instead of its anchor once produced a
# single "level" holding 75 touches that spanned 7,000 to 11,000.
_px = nq.load_from_cache(TICKERS[0], "prices")
_sr = nq.support_resistance(_px)
_last_close = float(_px.sort_values("date")["close"].iloc[-1])
check("resistance sits above the last close",
      all(l["price"] > _last_close for l in _sr["resistance"]))
check("support sits below the last close",
      all(l["price"] < _last_close for l in _sr["support"]))
check("no level chains across a wide range",
      all(l["touches"] < 40 for l in _sr["support"] + _sr["resistance"]),
      str([l["touches"] for l in _sr["support"] + _sr["resistance"]]))

# The cone is drawn as a curve, and its anchors must still match the table.
_cone = nq.volatility_cone(_px)
if _cone.get("available"):
    _path = nq.cone_path(_cone, level=50)
    _at252 = _path.iloc[-1]
    _lo, _hi = _cone["bands"][252][50]
    check("the drawn curve ends where the tabulated range ends",
          abs(_at252["low"] - _lo) < 1 and abs(_at252["high"] - _hi) < 1,
          f"{_at252['low']:.0f}/{_lo:.0f}")

# Trading conditions are three readings, never one score. A composite would be
# read as a fear-and-greed dial, and that reading was tested on this panel and
# did not hold — so the absence of a combined number is asserted, not assumed.
_tc = nq.trading_conditions(nq.load_from_cache(TICKERS[0], "prices"))
check("trading conditions available", _tc.get("available"))
if _tc.get("available"):
    check("range position is a percentage of the 52-week range",
          0 <= _tc["range_position"] <= 100, f"{_tc['range_position']:.0f}")
    check("the 52-week low sits below the high",
          _tc["low_52w"] < _tc["high_52w"])
    check("ratios are measured against the stock's own year",
          _tc["volatility_ratio"] > 0 and _tc["volume_ratio"] > 0)
    check("no composite score is produced",
          not any(k in _tc for k in ("score", "index", "sentiment")))
# Their bands describe rather than judge, so none of them earns an arrow.
check("condition bands carry no direction",
      all(nq.band_direction(b) == 0 for b in
          (nq.range_band(90), nq.range_band(10), nq.activity_band(2.0),
           nq.turbulence_band(2.0), nq.turbulence_band(0.5))))

check("app runs with NO API key", not at.exception, str(at.exception)[:300] if at.exception else "")
check("app made zero API calls", CALLS["n"] == 0, f"{CALLS['n']}")
check("cached mode default", radio(at, "Data source").value == "Cached snapshot",
      str(radio(at, "Data source").value))
succ = " ".join(s.value for s in at.success)
# The sidebar now tells the reader when the figures are from rather than what
# they cost to fetch; the credit accounting is the operator's concern.
check("states the data vintage", "Figures as of" in succ, succ[:100])

at.session_state["ticker"] = TICKERS[0]; at.session_state["analysed"] = TICKERS[0]
at.run()
check("single stock renders", not at.exception, str(at.exception)[:300] if at.exception else "")
labels = {m.label: m.value for m in at.metric}
_risk = "6M High Volatility Probability"
_swing = "Typical swing in a year"
check("volatility forecast shown",
      labels.get(_risk, "Not available") != "Not available", str(labels.get(_risk)))
check("volatility forecast leads the page", _risk in labels)
check("both volatility horizons appear",
      "12M High Volatility Probability" in labels, ", ".join(sorted(labels)))
check("historical risk still shown", "Worst drop from a peak" in labels)
check("both risk classes lead the page",
      {"6M Risk Class", "12M Risk Class"} <= set(labels), ", ".join(sorted(labels)))

# A volatility is a distance and can never be a minus. Printed bare next to a
# drawdown it reads as a return, which is the confusion the sign exists to end.
_swings = [v for k, v in labels.items()
           if k in (_swing, "Swing on down days") and v not in ("—", "Not available")]
check("volatilities are written as plus-minus",
      _swings and all(v.startswith("±") for v in _swings), str(_swings))
check("the drop is not written as plus-minus",
      not str(labels.get("Worst drop from a peak", "")).startswith("±"),
      str(labels.get("Worst drop from a peak")))

# The return estimates are demoted, not deleted: a test that found nothing is
# still a result, and removing it would leave nothing to judge the rest by.
check("both return horizons kept on the page",
      {"6M Positive Return Probability",
       "12M Positive Return Probability"} <= set(labels),
      ", ".join(sorted(labels)))
# Return and volatility are laid out the same way, so neither gets more of the
# page than its evidence supports.
check("return and volatility read as a matched pair",
      len([l for l in labels if l.endswith("Positive Return Probability")])
      == len([l for l in labels if l.endswith("High Volatility Probability")]),
      ", ".join(sorted(labels)))
# The feature table is hand-rolled HTML, not st.dataframe: st.dataframe draws
# onto a canvas whose cells clip long text, and this table is read rather than
# sorted, so the wrapping matters more than the interactivity.
# Selected by its own class: the projected-range table shares nq-table.
feat = [str(m.value) for m in at.markdown if "nq-metrics" in str(m.value)]
check("feature table rendered", len(feat) == 1, f"{len(feat)} tables")

check("STILL zero API calls", CALLS["n"] == 0, f"{CALLS['n']}")

# The previous version of this block was guarded by `if btn:`, so when the
# button was renamed the whole thing skipped and reported nothing. That is the
# same failure that once let a renamed mode label silently open the wrong page.
# Find the button by asserting it exists.
radio(at, "Analysis").set_value(appmod.MODE_PICKS).run()
_screen = [b for b in at.button if "screen" in (b.label or "").lower()]
check("screening view offers its button", _screen,
      ", ".join(b.label or "" for b in at.button))
if _screen:
    _screen[0].click().run()
    check("screening runs offline", not at.exception,
          str(at.exception)[:300] if at.exception else "")
    _tables = [d.value for d in at.dataframe
               if "6M High Volatility Probability" in list(d.value.columns)]
    check("screening table produced", len(_tables) > 0,
          str([list(d.value.columns) for d in at.dataframe])[:200])
    if _tables:
        _table = _tables[0]
        # Percentages held as text sort 9% above 53%, and a sortable table that
        # sorts wrongly is worse than one that does not sort at all.
        _numeric = ["6M High Volatility Probability",
                    "Realised Volatility (1Y)", "Data Quality"]
        check("sortable columns are numeric, not text",
              all(str(_table[c].dtype).startswith("float") for c in _numeric
                  if c in _table),
              str({c: str(_table[c].dtype) for c in _numeric if c in _table}))
        # Every column a reader could mistake for a return must say what the
        # number is. "6M" alone reads as a return; "Probability" says it is not.
        check("probability columns name themselves",
              all("Probability" in c for c in _table.columns
                  if c.startswith(("6M", "12M"))
                  and "Risk Class" not in c),
              ", ".join(_table.columns))
        # The ranking must rest on the estimate that passed its test, not on
        # the one that did not.
        check("ranked ascending by the validated forecast",
              list(_table["Rank"]) == sorted(_table["Rank"])
              and _table["6M High Volatility Probability"].is_monotonic_increasing,
              str(list(_table["6M High Volatility Probability"])[:5]))
        _return_column = next((c for c in _table.columns
                               if "Positive Return" in c), None)
        if _return_column and len(_table) > 2:
            _values = [v for v in _table[_return_column] if v == v]
            check("not ranked by the return estimate",
                  _values != sorted(_values, reverse=True), str(_values[:5]))
        # The horizon control has to actually control something. A loop
        # variable named `horizon` once overwrote the reader's choice, so the
        # radio moved and the table never did.
        _hz = [r for r in at.radio if r.label == "Return horizon"]
        if _hz:
            _before = next(c for c in _table.columns if "Positive Return" in c)
            _hz[0].set_value("12 Months").run()
            [b for b in at.button if "screen" in (b.label or "").lower()][0].click().run()
            _after = next(c for c in at.dataframe[0].value.columns
                          if "Positive Return" in c)
            check("the horizon control changes the return column",
                  _before.startswith("6M") and _after.startswith("12M"),
                  f"{_before} then {_after}")
            check("risk classes are banded",
              set(_table["6M Risk Class"]) <= {"High", "Medium", "Low", "Unknown"},
              str(set(_table["6M Risk Class"])))

# ── Portfolio analysis: entering, removing, and measuring together ──────
radio(at, "Analysis").set_value(appmod.MODE_PORTFOLIO).run()
check("portfolio view opens", not at.exception,
      str(at.exception)[:300] if at.exception else "")

_added = []
for _want in TICKERS[:3]:
    _options = [o for o in at.selectbox[0].options if o.startswith(_want)]
    if not _options:
        continue
    at.selectbox[0].set_value(_options[0]).run()
    at.number_input[0].set_value(10).run()
    [b for b in at.button if b.label == "Add"][0].click().run()
    _added.append(_want)
check("holdings can be entered", len(_added) >= 2, str(_added))
def _portfolio(app):
    """AppTest's session_state maps attributes to keys, so .get is not a method."""
    try:
        return dict(app.session_state["portfolio"])
    except (KeyError, AttributeError):
        return {}

check("holdings are remembered",
      set(_portfolio(at)) == set(_added), str(_portfolio(at)))

# A wrong entry has to be removable, or the only fix is reloading the page.
_removable = [b for b in at.button if b.label == "Remove"]
check("every holding offers a way to remove it",
      len(_removable) == len(_added), f"{len(_removable)} for {len(_added)}")
if _removable:
    _removable[-1].click().run()
    check("removing a holding takes it out",
          len(_portfolio(at)) == len(_added) - 1, str(_portfolio(at)))

_analyse = [b for b in at.button if "analyse" in (b.label or "").lower()]
check("portfolio offers an explicit analyse button", _analyse,
      ", ".join(b.label or "" for b in at.button))
if _analyse:
    _analyse[0].click().run()
    check("portfolio analyses offline", not at.exception,
          str(at.exception)[:300] if at.exception else "")
    _labels = {m.label: m.value for m in at.metric}
    check("portfolio reports a value", "Total value" in _labels, str(list(_labels)))
    check("portfolio swing is written as plus-minus",
          str(_labels.get("Typical swing in a year", "")).startswith("\u00b1"),
          str(_labels.get("Typical swing in a year")))

check("ENTIRE DEMO = 0 API CALLS", CALLS["n"] == 0, f"{CALLS['n']}")

os.chdir(HOME)
print(f"\n{'='*62}\n" + ("ALL PASSED" if not fails else f"FAILURES ({len(fails)}): " + ", ".join(fails)) + f"\n{'='*62}")
shutil.rmtree(WORK, ignore_errors=True)
sys.exit(1 if fails else 0)
