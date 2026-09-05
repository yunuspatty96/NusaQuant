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
for _mode in (appmod.MODE_SINGLE, appmod.MODE_PICKS, appmod.MODE_SECTOR):
    _probe = AppTest.from_file(str(WORK / "app.py"), default_timeout=240)
    _probe.run()
    radio(_probe, "Analysis").set_value(_mode).run()
    _heads = [str(h.value) for h in _probe.markdown if "nq-sec" in str(h.value)]
    _opened = any(_mode in h for h in _heads) and not _probe.exception
    check(f"view opens: {_mode[:34]}", _opened,
          "" if _opened else (str(_probe.exception)[:120] if _probe.exception
                              else "the heading for this view never rendered"))

check("app runs with NO API key", not at.exception, str(at.exception)[:300] if at.exception else "")
check("app made zero API calls", CALLS["n"] == 0, f"{CALLS['n']}")
check("cached mode default", radio(at, "Data source").value == "Cached snapshot",
      str(radio(at, "Data source").value))
succ = " ".join(s.value for s in at.success)
check("states 0 credits", "0 API credits" in succ, succ[:100])

at.session_state["ticker"] = TICKERS[0]; at.session_state["analysed"] = TICKERS[0]
at.run()
check("single stock renders", not at.exception, str(at.exception)[:300] if at.exception else "")
labels = {m.label: m.value for m in at.metric}
check("6M probability shown", labels.get("6M probability", "—") != "—", str(labels.get("6M probability")))
check("12M probability shown", labels.get("12M probability", "—") != "—")
check("reliability shown", "Machine learning model reliability" in labels)
check("risk shown", "Annualised volatility" in labels)
# The feature table is hand-rolled HTML, not st.dataframe: st.dataframe draws
# onto a canvas whose cells clip long text, and this table is read rather than
# sorted, so the wrapping matters more than the interactivity.
feat = [str(m.value) for m in at.markdown if "class='nq-table'" in str(m.value)]
check("feature table rendered", len(feat) == 1, f"{len(feat)} tables")
if feat:
    # Group headers are <tr class='grp'>, which does not match "<tr>",
    # so only the plain metric rows and the one header row are counted.
    rows = feat[0].count("<tr>") - 1
    check("metric table lists every metric", rows == len(nq.METRIC_NAMES),
          f"{rows} of {len(nq.METRIC_NAMES)}")
    check("metric table is grouped by category",
          feat[0].count("<tr class='grp'>") == len(nq.CATEGORY_ORDER))
    check("acronyms are expanded", all(x in feat[0] for x in
          ("Return on Equity", "Return on Asset", "Price to Earnings",
           "Price to Book Value", "Price to Sales", "Price to Cash Flow",
           "Enterprise Value to EBITDA", "Debt to Equity")))
    check("an absent metric still says why",
          "Not available." in feat[0])
check("STILL zero API calls", CALLS["n"] == 0, f"{CALLS['n']}")

radio(at, "Analysis").set_value(appmod.MODE_PICKS).run()
btn = [b for b in at.button if "top picks" in (b.label or "").lower()]
if btn:
    btn[0].click().run()
    check("Top picks run offline", not at.exception, str(at.exception)[:300] if at.exception else "")
    rank = [d.value for d in at.dataframe if "Probability up" in list(d.value.columns)]
    check("ranking produced", len(rank) > 0)
    if rank:
        pv = [float(s.rstrip('%')) for s in rank[0]["Probability up"]]
        check("ranked descending", pv == sorted(pv, reverse=True), str(pv))
check("ENTIRE DEMO = 0 API CALLS", CALLS["n"] == 0, f"{CALLS['n']}")

os.chdir(HOME)
print(f"\n{'='*62}\n" + ("ALL PASSED" if not fails else f"FAILURES ({len(fails)}): " + ", ".join(fails)) + f"\n{'='*62}")
shutil.rmtree(WORK, ignore_errors=True)
sys.exit(1 if fails else 0)
