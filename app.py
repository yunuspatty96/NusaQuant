"""
NusaQuant — IDX Machine Learning Market Intelligence
====================================================

    streamlit run app.py

Two modes:

* **Cached snapshot** (default) — runs the whole dashboard on the real Sectors
  data collected during training. Costs zero API credits. Labelled with the
  date the snapshot was taken, because serving month-old figures as though
  they were live would be a quiet lie.
* **Live Sectors API** — today's figures, for companies outside the snapshot.
  Costs credits, and the app says how many before you spend them.

No API key is needed for cached mode. In live mode the key is typed into the
sidebar, held in session only, and never logged or written to disk.

Research and decision support only. Probabilities are model estimates, not
guarantees and not financial advice. No LLM anywhere — every sentence comes
from a template in nusaquant.py.
"""

from __future__ import annotations

import colorsys
import datetime as dt
import math
import json
from html import escape
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

import nusaquant as nq

MODELS_DIR = Path("models")
CACHE_TTL = 6 * 60 * 60

ACCENT, POSITIVE, NEGATIVE, MUTED, GRID = "#1D4E6F", "#1B7F4B", "#B3341F", "#5C6672", "#E4E7EB"
#: Cost sits beside revenue in the income chart, so it needs a colour that
#: reads as the opposite of revenue without shouting like the loss red.
COST = "#E8A0A0"
PRICE_WINDOWS = {"1Y": 1, "3Y": 3, "5Y": 5}

#: The trend gradient runs from the palette's red to its green by rotating the
#: hue rather than blending the two hex values. A straight RGB interpolation
#: between them passes through a muddy khaki at the midpoint — "Recovering"
#: came out #958237 — because it cuts across the colour wheel instead of
#: travelling around it. Rotating hue keeps every intermediate step as
#: saturated as the endpoints.
TREND_HUE = (8, 148)        # degrees: the palette's red and green
TREND_SATURATION = 0.68
TREND_LIGHTNESS = (0.42, 0.30)   # red sits a little lighter than green
#: The chip carries white text, and yellow is the lightest hue on the path, so
#: the middle of the gradient is darkened to keep it legible. Without this dip
#: "Weakening" landed at 2.9:1 against white — well under the 4.5:1 WCAG AA
#: wants for text this size. At 0.12 the worst state on the scale is 4.9:1.
TREND_MIDTONE_DIP = 0.12


def trend_colour(position: float) -> str:
    """Red at 0, green at 1, through amber and olive. Grey when unknown."""
    if position is None or not np.isfinite(position):
        return MUTED
    position = float(np.clip(position, 0.0, 1.0))
    hue = TREND_HUE[0] + (TREND_HUE[1] - TREND_HUE[0]) * position
    light = (TREND_LIGHTNESS[0]
             + (TREND_LIGHTNESS[1] - TREND_LIGHTNESS[0]) * position
             - TREND_MIDTONE_DIP * math.sin(math.pi * position))
    red, green, blue = colorsys.hls_to_rgb(hue / 360.0, light, TREND_SATURATION)
    return "#%02X%02X%02X" % (round(red * 255), round(green * 255), round(blue * 255))


#: Shown beside every universe-size control. The list is ordered largest first
#: on both paths — the screener sorts by -market_cap, and the cached side is
#: sorted by the market cap in its own price files — so the wording is a
#: description of what happens rather than a promise about it.
UNIVERSE_SIZE_NOTE = ("Taken from the largest companies by market capitalisation, "
                      "largest first.")
UNIVERSE_SIZE_HELP = ("How many companies to include, counting down from the "
                      "largest by market capitalisation. {total} are available.")


#: The three views, named once. These strings are the sidebar labels, the
#: headings on the pages they open, and the values main() dispatches on, so a
#: reworded label used to have to be changed in three places by hand — and
#: twice it was not: the heading disagreed with the sidebar, and renaming the
#: sector view left the dispatch matching a string nothing produced any more,
#: which quietly opened Top Picks instead.
MODE_SINGLE = "Single Stock Analysis"
MODE_PICKS = "Machine Learning Top Picks (Ranked)"
MODE_SECTOR = "Sector Ranking (Compare Ratios)"

#: Plotly options shared by every chart. The mode bar is left at its default,
#: which reveals it on hover rather than parking it permanently over the plot,
#: and the buttons that do not apply to a time series are dropped. Drag to box
#: zoom, scroll to zoom, double-click to reset.
CHART_CONFIG = {
    "displaylogo": False,
    "scrollZoom": True,
    "doubleClick": "reset",
    "modeBarButtonsToRemove": ["select2d", "lasso2d", "toggleSpikelines",
                               "hoverClosestCartesian", "hoverCompareCartesian"],
}

#: Live-mode cost for one company: 8 quarters (the minimum for TTM plus
#: one-year growth) plus one report section.
QUARTERS_FOR_INFERENCE = 8
CREDITS_PER_COMPANY = QUARTERS_FOR_INFERENCE + 1


# ══════════════════════════════════════════════════════════════════════
# MODELS
# ══════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner=False)
def load_models() -> tuple[dict[str, Any], dict[str, Any]]:
    """Load the two artifacts once per session, not once per rerun."""
    models: dict[str, Any] = {}
    for horizon in nq.HORIZON_TRADING_DAYS:
        path = MODELS_DIR / f"model_{horizon}_xgb.joblib"
        if path.exists():
            try:
                models[horizon] = joblib.load(path)
            except Exception:
                continue
    metadata = {}
    meta_path = MODELS_DIR / "metadata.json"
    if meta_path.exists():
        try:
            metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        except ValueError:
            metadata = {}
    return models, metadata


def diagnose_models() -> dict[str, Any]:
    """Explain *why* no models loaded, not merely that none did.

    Three very different causes share one symptom, and telling them apart from
    inside the app saves a long guessing game.
    """
    report = {"dir": str(MODELS_DIR.resolve()), "exists": MODELS_DIR.exists(),
              "files": [], "loaded": [], "errors": []}
    if MODELS_DIR.exists():
        report["files"] = sorted(p.name for p in MODELS_DIR.iterdir() if p.is_file())
    for horizon in nq.HORIZON_TRADING_DAYS:
        path = MODELS_DIR / f"model_{horizon}_xgb.joblib"
        if not path.exists():
            continue
        try:
            joblib.load(path); report["loaded"].append(path.name)
        except Exception as exc:
            report["errors"].append(f"{path.name}: {type(exc).__name__}: {exc}"[:300])
    if report["loaded"]:
        report["diagnosis"] = "ok"
    elif report["errors"]:
        report["diagnosis"] = "unreadable"
    else:
        report["diagnosis"] = "absent"
    return report


def predict(features: pd.DataFrame, artifact: dict[str, Any] | None,
            horizon: str) -> dict[str, Any]:
    """Score one company, or refuse and say why.

    Refuses rather than filling absent inputs with zeros to force a number out.
    A confident-looking probability built on missing data is worse than none.
    """
    if not artifact:
        return {"available": False, "reason": f"No trained machine learning model for the {horizon} horizon."}
    model_features = list(artifact.get("feature_names", []))
    missing_columns = [c for c in model_features if c not in features.columns]
    if missing_columns:
        return {"available": False,
                "reason": "Some machine learning model inputs are unavailable for this company.",
                "missing": missing_columns}

    frame = features[model_features]
    completeness = nq.data_quality(frame, model_features)
    if completeness < nq.MIN_DATA_COMPLETENESS:
        absent = [c for c in model_features if pd.isna(frame.iloc[0][c])]
        return {"available": False,
                "reason": "Too many machine learning model inputs are missing for a reliable prediction.",
                "missing": absent, "data_quality": completeness}
    try:
        probability = float(artifact["pipeline"].predict_proba(frame)[0, 1])
    except Exception:
        return {"available": False, "reason": "The machine learning model could not score this company."}
    return {"available": True, "probability": probability, "data_quality": completeness}


# ══════════════════════════════════════════════════════════════════════
# DATA ACCESS
# ══════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def snapshot_tickers() -> list[str]:
    return nq.cached_tickers("quarterly")


@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def snapshot_as_of() -> str:
    return nq.cache_as_of()


@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def snapshot_data(ticker: str, kind: str) -> pd.DataFrame:
    return nq.load_from_cache(ticker, kind)


@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def snapshot_market_caps() -> dict[str, float]:
    """Latest cached market cap per ticker, for ordering the universe.

    The live screener returns companies largest-first, but the cached side had
    no ordering at all: cached_tickers() is alphabetical, and the merge that
    added the classification re-sorted universe.parquet by symbol as well. A
    "universe size" of 10 therefore took AMMN through BYAN rather than the ten
    largest, which is not what the control claims to do. The cached price files
    already carry a market cap, so the order costs nothing to get right.
    """
    caps = {}
    for ticker in nq.cached_tickers():
        prices = nq.load_from_cache(ticker, "prices")
        if prices.empty or "market_cap" not in prices.columns:
            continue
        latest = prices.sort_values("date").iloc[-1]
        caps[ticker] = nq._to_float(latest.get("market_cap"))
    return caps


def by_market_cap(frame: pd.DataFrame) -> pd.DataFrame:
    """Largest first, so "the top N" means what it says."""
    caps = snapshot_market_caps()
    ordered = frame.copy()
    ordered["_cap"] = ordered["symbol"].map(caps)
    return (ordered.sort_values("_cap", ascending=False, na_position="last")
            .drop(columns="_cap").reset_index(drop=True))


# The API key is part of these cache keys, which is correct — two keys must
# not share cached responses. Nothing is persisted (no persist= argument), so
# the key never reaches disk.

@st.cache_data(ttl=24 * 3600, show_spinner=False)
def check_key(api_key: str) -> bool:
    return nq.validate_api_key(api_key)


@st.cache_data(ttl=24 * 3600, show_spinner=False)
def live_universe(api_key: str, where: str, limit: int) -> pd.DataFrame:
    return nq.get_companies(api_key, where=where, limit=limit)


@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def live_report(ticker: str, api_key: str) -> dict[str, Any]:
    return nq.get_company_report(api_key, ticker, ("overview",))


@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def live_quarterly(ticker: str, api_key: str, n: int) -> pd.DataFrame:
    return nq.get_quarterly_financials(api_key, ticker, n)


@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def live_prices(ticker: str, start: str, end: str, api_key: str) -> pd.DataFrame:
    return nq.get_daily_history(api_key, ticker, start, end)


def _with_dividends(features: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Fill the dividend columns from the cached screen.

    They are written here rather than in compute_features on purpose. The
    training path goes through compute_features and must never see them: a
    trailing figure taken today is not what was known in 2022, and the whole
    point-in-time discipline of this project depends on that line holding.
    """
    for metric, value in nq.company_dividends(ticker).items():
        if metric in features.columns:
            features.loc[features.index[0], metric] = value
    return features


def load_company(ticker: str, api_key: str, offline: bool) -> dict[str, Any]:
    """Assemble one company's features, from the snapshot or from the API.

    Both paths end in nq.features_frame, the same function the training script
    used at every historical observation, so a feature cannot mean one thing
    in training and another here.
    """
    if offline:
        quarterly = snapshot_data(ticker, "quarterly")
        prices = snapshot_data(ticker, "prices")
        if quarterly.empty:
            raise nq.SectorsAPIError(f"{ticker} is not in the cached snapshot.")
        market_cap = close = np.nan
        if not prices.empty:
            latest = prices.sort_values("date").iloc[-1]
            market_cap = nq._to_float(latest.get("market_cap"))
            close = nq._to_float(latest.get("close"))
        # The screener result is already cached, so the name and the IDX
        # classification both cost nothing here.
        overview = {"market_cap": market_cap, "last_close_price": close,
                    **nq.company_classification(ticker)}
        name = nq.company_name(ticker)
    else:
        report = live_report(ticker, api_key)
        quarterly = live_quarterly(ticker, api_key, QUARTERS_FOR_INFERENCE)
        overview = (report or {}).get("overview") or {}
        market_cap = overview.get("market_cap")
        # The per-share metrics need a share count, and the share count is
        # market cap over close. Both come from the overview in live mode.
        close = overview.get("last_close_price")
        prices = pd.DataFrame()
        name = (report or {}).get("company_name") or ticker

    return {
        "ticker": nq.normalise_symbol(ticker),
        "name": name,
        "overview": overview,
        "quarterly": quarterly,
        "prices": prices,
        "features": _with_dividends(
            nq.features_frame(quarterly, market_cap, close), ticker),
        "latest_period": quarterly["report_date"].max() if not quarterly.empty else pd.NaT,
        "n_quarters": len(quarterly),
    }


# ══════════════════════════════════════════════════════════════════════
# UI HELPERS
# ══════════════════════════════════════════════════════════════════════

def configure_page() -> None:
    st.set_page_config(page_title="NusaQuant — IDX Machine Learning Market Intelligence",
                       page_icon="◧", layout="wide")
    # Tabular figures: this dashboard is mostly numbers, and a column of
    # prices that does not align is harder to scan.
    st.markdown(f"""<style>
      html, body, [class*="css"] {{ font-feature-settings: "tnum" 1; }}
      /* Metric values must wrap. A metric can hold a number four characters
         wide or the words "No measurable edge", and at the old 1.5rem the
         second one ran past its column and was clipped. Sized for the longest
         label this dashboard can produce, not the shortest. */
      [data-testid="stMetricValue"] {{
        font-size:1.15rem; font-weight:600; line-height:1.3;
        font-variant-numeric:tabular-nums;
        white-space:normal; overflow-wrap:anywhere;
      }}
      [data-testid="stMetricLabel"] {{ color:{MUTED}; }}
      /* Streamlit lays the label out as a grid: text in one cell, the tooltip
         icon in a 22px cell beside it, and the <p> inside set to nowrap. A
         label as long as "12M probability of positive return" then runs past
         its 168px cell and is cut. Letting it wrap is the whole fix. */
      [data-testid="stMetricLabel"] p {{ font-size:.8rem; line-height:1.3;
                                        white-space:normal;
                                        overflow-wrap:break-word; }}
      [data-testid="stMetricLabel"] {{ align-items:start; }}
      [data-testid="stMetric"] {{ overflow:visible; }}
      .nq-title {{ font-size:1.6rem; font-weight:650; letter-spacing:-.01em; margin-bottom:.1rem; }}
      .nq-sub {{ color:{MUTED}; font-size:.9rem; }}
      .nq-sec {{ font-size:1.05rem; font-weight:600; margin:1.4rem 0 .5rem;
                 padding-bottom:.3rem; border-bottom:1px solid {GRID}; }}
      .nq-note {{ color:{MUTED}; font-size:.84rem; line-height:1.55; }}
      .nq-tile {{ padding:.1rem 0 .3rem; }}
      .nq-tile-label {{ color:{MUTED}; font-size:.8rem; line-height:1.3;
                        margin-bottom:.15rem; }}
      .nq-tile-value {{ font-size:1.15rem; font-weight:600; line-height:1.3;
                        font-variant-numeric:tabular-nums;
                        overflow-wrap:anywhere; }}
      .nq-tile-band {{ font-size:.8rem; font-weight:600; margin-top:.1rem; }}
      .nq-tile-help {{ display:inline-block; margin-left:.3rem; width:.95rem;
                       height:.95rem; line-height:.95rem; text-align:center;
                       border:1px solid {GRID}; border-radius:50%;
                       font-size:.65rem; cursor:help; }}
      .nq-foot {{ margin:2.5rem 0 .5rem; padding-top:1rem;
                  border-top:1px solid {GRID}; color:{MUTED}; font-size:.8rem;
                  line-height:1.6; }}
      .nq-foot strong {{ color:{MUTED}; font-weight:600; }}
      .nq-name {{ font-size:1.3rem; font-weight:650; letter-spacing:-.01em;
                  line-height:1.3; margin:.2rem 0 .1rem; overflow-wrap:anywhere; }}
      .nq-name span {{ color:{MUTED}; font-weight:450; }}
      .nq-chip {{ display:inline-block; padding:.12rem .5rem; border-radius:10px;
                  font-size:.76rem; font-weight:600; border:1px solid {GRID};
                  color:{MUTED}; margin-right:.35rem; }}
      .nq-trend {{ display:inline-block; padding:.2rem .7rem; border-radius:12px;
                   font-size:.8rem; font-weight:600; color:#FFFFFF;
                   letter-spacing:.01em; }}
      .nq-trend-note {{ color:{MUTED}; font-size:.78rem; margin-left:.5rem; }}
      div[data-testid="stDataFrame"] {{ font-variant-numeric:tabular-nums;
                                        font-size:.86rem; }}
      /* st.dataframe draws onto a canvas, so its cells clip long text and no
         stylesheet can reach inside to wrap them. Reference tables that are
         read rather than sorted are plain HTML for exactly that reason. */
      .nq-table {{ width:100%; border-collapse:collapse; font-size:.86rem;
                   table-layout:fixed; margin:.2rem 0 .4rem; }}
      .nq-table thead th {{ text-align:left; font-weight:600; color:#FFFFFF;
                            background:{ACCENT}; padding:.5rem .6rem;
                            font-size:.78rem; letter-spacing:.03em;
                            text-transform:uppercase; border-bottom:none; }}
      .nq-table thead th:first-child {{ border-top-left-radius:4px; }}
      .nq-table thead th:last-child {{ border-top-right-radius:4px; }}
      .nq-table td {{ padding:.45rem .6rem; border-bottom:1px solid {GRID};
                      vertical-align:top; line-height:1.45; }}
      .nq-table tr:last-child td {{ border-bottom:none; }}
      .nq-table .num {{ font-variant-numeric:tabular-nums; white-space:nowrap; }}
      .nq-table .na {{ color:{MUTED}; }}
      .nq-table .sub {{ display:block; color:{MUTED}; font-size:.78rem;
                        font-weight:400; }}
      .nq-table tr.grp td {{ background:#F7F8FA; font-weight:600;
                             font-size:.78rem; letter-spacing:.04em;
                             text-transform:uppercase; color:{MUTED};
                             padding:.4rem .6rem; }}
    </style>""", unsafe_allow_html=True)


def footer() -> None:
    st.markdown(
        "<div class='nq-foot'>"
        "<strong>NusaQuant &copy; 2026 Patty Kyoudai</strong><br>"
        "Developed by Patty Kyoudai &middot; Yunus Patty &middot; Lukas Patty"
        "</div>", unsafe_allow_html=True)


def section(title: str) -> None:
    st.markdown(f'<div class="nq-sec">{title}</div>', unsafe_allow_html=True)


def note(text: str) -> None:
    st.markdown(f'<div class="nq-note">{text}</div>', unsafe_allow_html=True)


def probability_bar(probability: float) -> None:
    """A flat bar. No gauges, no gradients."""
    if not np.isfinite(nq._to_float(probability)):
        st.write("—"); return
    pct = max(0.0, min(1.0, probability)) * 100
    colour = POSITIVE if probability >= .5 else MUTED
    st.markdown(f"""<div style="margin:.2rem 0 .6rem;">
      <div style="background:{GRID};border-radius:3px;height:10px;width:100%;">
        <div style="width:{pct:.1f}%;background:{colour};height:10px;border-radius:3px;"></div>
      </div></div>""", unsafe_allow_html=True)


def show_error(error: nq.SectorsAPIError) -> None:
    st.error(f"{error.message}\n\n{nq.API_HELP}")
    if error.detail:
        with st.expander("Technical details"):
            st.code(error.detail, language="text")


def render_missing_models() -> None:
    """Never invent a prediction to fill the gap — explain the actual cause."""
    report = diagnose_models()
    if report["diagnosis"] == "unreadable":
        st.error("**Machine learning model files exist but could not be loaded.**\n\n"
                 "Almost always a library version mismatch: a pickled pipeline "
                 "is not portable across major scikit-learn or xgboost versions. "
                 "Re-run `python train.py`, or match the versions in "
                 "`requirements.txt`.")
    else:
        st.warning("**No trained models found.**\n\n"
                   "NusaQuant will not show a probability it has not trained.")
        st.markdown("""
Train locally, then commit the result:

```bash
export SECTORS_API_KEY=your-key-here
python train.py

git add -f models/ data/cache/
git commit -m "Add trained models and snapshot"
git push
```

`train.py` cannot run on Streamlit Cloud, so `models/` must be committed.
See **DEPLOY.md**.
        """)
    with st.expander("Diagnostics"):
        st.write(f"**Looking in:** `{report['dir']}`")
        st.write(f"**Exists:** {report['exists']}")
        st.code("\n".join(report["files"]) or "(no files)", language="text")
        for error in report["errors"]:
            st.code(error, language="text")


# ══════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════

def render_sidebar(metadata: dict[str, Any]) -> dict[str, Any]:
    snapshot = snapshot_tickers()
    with st.sidebar:
        st.markdown("### NusaQuant")

        if snapshot:
            source = st.radio("Data source", ["Cached snapshot", "Live Sectors API"],
                              help="The snapshot is real Sectors data collected "
                                   "during training. It costs no credits, but it "
                                   "is a snapshot rather than today's figures.")
        else:
            source = "Live Sectors API"
        offline = source == "Cached snapshot"

        api_key = ""
        if offline:
            st.success(f"Cached mode — 0 API credits.\n\n"
                       f"Snapshot as of {snapshot_as_of()} · "
                       f"{len(snapshot)} companies")
        else:
            api_key = st.text_input(
                "Sectors API key", value=st.session_state.get("api_key", ""),
                type="password",
                help="Held in this session only. Never logged or saved.")
            if st.button("Connect", width="stretch", type="primary"):
                st.session_state["api_key"] = api_key.strip()
            api_key = st.session_state.get("api_key", "")
            if api_key:
                if check_key(api_key):
                    st.success("Sectors API connected")
                else:
                    st.error("Could not authenticate with that key.")
                    api_key = ""
            st.caption(f"About {CREDITS_PER_COMPANY} credits per company analysed.")

        st.divider()
        mode = st.radio("Analysis", [MODE_SINGLE, MODE_PICKS, MODE_SECTOR],
                        label_visibility="collapsed")

        st.divider()
        if metadata:
            st.caption(f"XGBoost v{metadata.get('version','—')} · "
                       f"{len(metadata.get('feature_set', []))} features · "
                       f"trained to {metadata.get('training_end_date','—')}")
        else:
            st.caption("No trained models loaded.")

    return {"offline": offline, "api_key": api_key, "snapshot": snapshot,
            "mode": mode}


# ══════════════════════════════════════════════════════════════════════
# SINGLE STOCK
# ══════════════════════════════════════════════════════════════════════

def render_profile(company: dict, predictions: dict) -> None:
    ticker, name = company["ticker"], company["name"]
    heading = ticker if name == ticker else f"{ticker} <span>— {name}</span>"
    st.markdown(f"<div class='nq-name'>{heading}</div>", unsafe_allow_html=True)

    overview = company["overview"]
    # The trend moved to the price chart, which is what it describes; the
    # classification stays here, which is what the company is.
    chips = [c for c in (overview.get("sector"), overview.get("sub_sector")) if c]
    if chips:
        st.markdown("".join(f"<span class='nq-chip'>{c}</span>" for c in chips),
                    unsafe_allow_html=True)

    columns = st.columns(4)
    columns[0].metric("Latest close",
                      nq.format_rupiah(overview.get("last_close_price"), compact=False),
                      help=nq.TOOLTIPS["latest_close"])
    columns[1].metric("Market cap", nq.format_rupiah(overview.get("market_cap")),
                      help=nq.TOOLTIPS["market_cap"])
    for column, horizon in ((columns[2], "6m"), (columns[3], "12m")):
        result = predictions.get(horizon, {})
        # Spelled out rather than left as "6M probability": the horizon alone
        # never said probability of what, and the answer is the one thing a
        # reader must not guess at.
        label = f"{'6' if horizon == '6m' else '12'}M probability of positive return"
        if result.get("available"):
            column.metric(label, f"{result['probability'] * 100:.0f}%",
                          help=nq.TOOLTIPS["probability"])
        else:
            # Never a bare dash: an unavailable probability has a cause, and
            # the cause is more useful than the punctuation.
            column.metric(label, "Not available",
                          help=result.get("reason", "Prediction unavailable."))

    period = company["latest_period"]
    period_text = ("not reported" if pd.isna(period)
                   else f"Q{pd.Timestamp(period).quarter} {pd.Timestamp(period).year}")
    st.caption(f"Latest financial period: {period_text} · "
               f"{company['n_quarters']} quarters on file · "
               f"Generated {dt.datetime.now():%Y-%m-%d %H:%M}")


def render_chart(company: dict, api_key: str,
                 offline: bool) -> tuple[pd.DataFrame, str, dict]:
    section("Price history")
    # The window and the style sit with the chart they change rather than in
    # the sidebar: a control three sections away from its effect is one the
    # reader has to go looking for.
    left, middle, right = st.columns([1, 1, 1])
    window = left.radio("Window", list(PRICE_WINDOWS), horizontal=True,
                        key="price_window")
    style = middle.radio("Chart style", ["Line", "Candlestick"], horizontal=True,
                         key="chart_style")
    # Off by default is tempting, but the projection is the point of the
    # feature; a reader who wants a clean price line can clear it, and the
    # y-axis it stretches is the only cost.
    right.markdown("<div style='height:1.9rem'></div>", unsafe_allow_html=True)
    project = right.checkbox("Show 6/12-month projection", value=True,
                             key="show_cone", help=nq.TOOLTIPS["cone_range"])
    # Reserved above the chart and filled once the prices are in hand: in live
    # mode the series does not exist until the fetch below has run.
    trend_slot = st.empty()
    years = PRICE_WINDOWS[window]

    if offline:
        prices = company["prices"]
        if not prices.empty:
            prices = prices[prices["date"] >= prices["date"].max() - pd.DateOffset(years=years)]
    else:
        # Lazy: only the selected window is ever requested.
        end = dt.date.today()
        start = end - dt.timedelta(days=365 * years + 5)
        try:
            prices = live_prices(company["ticker"], start.isoformat(), end.isoformat(), api_key)
        except nq.SectorsAPIError as error:
            show_error(error); return pd.DataFrame(), window, {}

    # Indicators read the full cached series where there is one, so a 1Y window
    # does not blank the 12-month return; live mode has only what it fetched.
    source = company["prices"] if not company["prices"].empty else prices
    technical = nq.technical_state(source)
    technical["cone"] = nq.volatility_cone(source)
    if technical.get("available"):
        trend = technical["trend"]
        colour = trend_colour(nq.trend_position(trend))
        trend_slot.markdown(
            f"<span class='nq-trend' style='background:{colour}'>{escape(trend)}</span>"
            f"<span class='nq-trend-note'>price against its own 50- and "
            f"200-day averages</span>", unsafe_allow_html=True)

    if prices.empty:
        st.info("No price history available for this window.")
        return prices, window, technical

    history = prices.sort_values("date").reset_index(drop=True)
    close = history["close"].astype(float)
    first, last = float(close.iloc[0]), float(close.iloc[-1])
    gained = last >= first
    line_colour = POSITIVE if gained else NEGATIVE
    fill = "rgba(31,122,90,.10)" if gained else "rgba(179,52,31,.10)"

    ohlc = [c for c in ("open", "high", "low") if c in history.columns]
    # A candlestick needs the whole bar. Roughly half the cached companies
    # carry open/high/low and the rest carry only a close, so the choice is
    # offered and then honestly withdrawn where it cannot be drawn.
    can_candle = (len(ohlc) == 3
                  and history[ohlc].notna().all(axis=1).sum() >= 10)
    candle = style == "Candlestick" and can_candle
    if style == "Candlestick" and not can_candle:
        st.info("This company's price history carries only a closing price, "
                "so there is no open, high and low to build a candle from. "
                "Showing the line instead.")

    has_volume = "volume" in history.columns and history["volume"].notna().any()
    figure = make_subplots(
        rows=2 if has_volume else 1, cols=1, shared_xaxes=True,
        row_heights=[0.76, 0.24] if has_volume else [1.0], vertical_spacing=0.04)

    if candle:
        figure.add_trace(go.Candlestick(
            x=history["date"], open=history["open"], high=history["high"],
            low=history["low"], close=history["close"], name="Price",
            increasing={"line": {"color": POSITIVE}, "fillcolor": POSITIVE},
            decreasing={"line": {"color": NEGATIVE}, "fillcolor": NEGATIVE},
            showlegend=False), row=1, col=1)
    else:
        figure.add_trace(go.Scatter(
            x=history["date"], y=close, mode="lines", name="Close",
            line={"color": line_colour, "width": 1.8},
            fill="tozeroy", fillcolor=fill,
            hovertemplate="Rp %{y:,.0f}<extra>Close</extra>"), row=1, col=1)

    # Moving averages are drawn only where they are fully formed. A 200-day
    # average seeded from 30 days of data is not a 200-day average.
    averages = nq.moving_averages(history)
    # `period`, not `window`: this loop used to shadow the selected price
    # window, leaving it set to 200 by the time the function returned.
    for period, colour, dash in ((50, ACCENT, "solid"), (200, MUTED, "dot")):
        column = f"ma{period}"
        if column in averages and averages[column].notna().any():
            figure.add_trace(go.Scatter(
                x=averages["date"], y=averages[column], mode="lines",
                name=f"MA{period}",
                line={"color": colour, "width": 1.1, "dash": dash},
                hovertemplate="Rp %{y:,.0f}<extra>MA" + str(period) + "</extra>"),
                row=1, col=1)

    if has_volume:
        figure.add_trace(go.Bar(
            x=history["date"], y=history["volume"].astype(float), name="Volume",
            marker={"color": GRID}, hovertemplate="%{y:,.0f}<extra>Volume</extra>"),
            row=2, col=1)
        figure.update_yaxes(title_text="Volume", row=2, col=1, gridcolor=GRID,
                            showticklabels=False)

    # The area fill runs to zero, but the axis must not: a stock trading
    # between 4,000 and 8,000 drawn on a 0-8,000 axis loses half its range to
    # empty space and every move in it looks flat. Clip the view to the data,
    # padded, and let the fill disappear off the bottom.
    span = [float(np.nanmin(close)), float(np.nanmax(close))]
    if candle:
        span = [float(np.nanmin(history["low"])), float(np.nanmax(history["high"]))]

    # The cone is drawn from the FULL cached series, not the windowed slice: a
    # one-year window leaves barely enough returns to estimate volatility from,
    # and the estimate should not change because the reader changed the zoom.
    cone = nq.volatility_cone(company["prices"] if not company["prices"].empty
                              else prices) if project else {"available": False}
    if cone.get("available"):
        last_date = history["date"].iloc[-1]
        future = [last_date + pd.Timedelta(days=int(d * 365 / 252))
                  for d in (0, 126, 252)]
        for level, shade in ((80, "rgba(29,78,111,.09)"),
                             (50, "rgba(29,78,111,.20)")):
            upper = [cone["last"]] + [cone["bands"][h][level][1] for h in (126, 252)]
            lower = [cone["last"]] + [cone["bands"][h][level][0] for h in (126, 252)]
            figure.add_trace(go.Scatter(
                x=future + future[::-1], y=upper + lower[::-1], fill="toself",
                fillcolor=shade, line={"width": 0}, hoverinfo="skip",
                name=f"{level}% projected range"), row=1, col=1)
            span = [min(span[0], *lower), max(span[1], *upper)]
    for column in ("ma50", "ma200"):
        if column in averages and averages[column].notna().any():
            span[0] = min(span[0], float(averages[column].min()))
            span[1] = max(span[1], float(averages[column].max()))
    pad = max((span[1] - span[0]) * 0.08, span[1] * 0.02, 1.0)

    figure.update_layout(
        height=420 if has_volume else 340,
        margin={"l": 0, "r": 0, "t": 30, "b": 0},
        plot_bgcolor="white", paper_bgcolor="white", bargap=0.1,
        hovermode="x unified",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02,
                "xanchor": "left", "x": 0, "font": {"size": 11}},
        xaxis={"showgrid": False, "linecolor": GRID},
        xaxis2={"showgrid": False, "linecolor": GRID} if has_volume else None,
        yaxis={"gridcolor": GRID, "title": "Close (IDR)",
               "range": [span[0] - pad, span[1] + pad]})
    figure.update_xaxes(rangeslider_visible=False, showspikes=True,
                        spikemode="across", spikethickness=1,
                        spikecolor=GRID, spikedash="dot")
    st.plotly_chart(figure, width="stretch", config=CHART_CONFIG)

    change = (last / first - 1.0) if first > 0 else np.nan
    note(f"Over this window the close moved from "
         f"{nq.format_rupiah(first, compact=False)} to "
         f"{nq.format_rupiah(last, compact=False)}, a change of "
         f"{nq.format_percent(change, 1)}. MA50 and MA200 are drawn only where "
         f"there is enough history to form them. Drag across the chart to "
         f"zoom in, scroll to zoom, double-click to reset.")
    if cone.get("available"):
        note(nq.EXPLANATIONS["cone"])
    elif project:
        note("A projected range needs a year of daily prices behind it, and "
             "this company does not have one yet in the snapshot.")
    return prices, window, technical


def render_technical(technical: dict, window: str) -> None:
    """Descriptive trend state. Deliberately kept apart from the model."""
    section("Technical state")
    if not technical.get("available"):
        st.info("Not enough price history in this window to describe a trend.")
        return

    rsi = nq._to_float(technical.get("rsi14"))
    histogram = nq._to_float(technical.get("macd_histogram"))
    columns = st.columns(6)

    tile(columns[0], "Trend", technical["trend"], technical["trend"],
         nq.TOOLTIPS["trend"])
    tile(columns[1], "RSI (14)",
         f"{rsi:.0f}" if np.isfinite(rsi) else "Insufficient history",
         nq.rsi_band(rsi) if np.isfinite(rsi) else None, nq.TOOLTIPS["rsi"])
    tile(columns[2], "MACD (12, 26, 9)",
         f"{histogram:+,.1f}" if np.isfinite(histogram) else "Insufficient history",
         nq.macd_band(histogram) if np.isfinite(histogram) else None,
         nq.TOOLTIPS["macd"])
    tile(columns[3], "From 52-week high",
         nq.format_percent(technical.get("from_52w_high"), 0), None,
         nq.TOOLTIPS["from_52w_high"])
    tile(columns[4], "6-month return",
         nq.format_percent(technical.get("return_6m"), 0), None,
         nq.TOOLTIPS["return_6m"])
    tile(columns[5], "12-month return",
         nq.format_percent(technical.get("return_12m"), 0), None,
         nq.TOOLTIPS["return_12m"])

    cone = technical.get("cone") or {}
    if cone.get("available"):
        st.markdown("##### Projected range")
        rows = []
        for horizon, label in ((126, "6 months"), (252, "12 months")):
            for level in (50, 80):
                low, high = cone["bands"][horizon][level]
                rows.append(
                    "<tr>"
                    f"<td><strong>{label}</strong></td>"
                    f"<td>{level}% of the time</td>"
                    f"<td class='num'>{escape(nq.format_rupiah(low, compact=False))}"
                    f" &ndash; {escape(nq.format_rupiah(high, compact=False))}</td>"
                    f"<td class='num'>{(high / cone['last'] - 1) * 100:+.0f}% / "
                    f"{(low / cone['last'] - 1) * 100:+.0f}%</td>"
                    "</tr>")
        st.markdown(
            "<table class='nq-table nq-cone'>"
            "<colgroup><col style='width:16%'><col style='width:22%'>"
            "<col style='width:38%'><col style='width:24%'></colgroup>"
            "<thead><tr><th>Horizon</th><th>Lands inside</th><th>Price range</th>"
            "<th>Versus today</th></tr></thead>"
            f"<tbody>{''.join(rows)}</tbody></table>", unsafe_allow_html=True)

    ma50, ma200 = nq._to_float(technical.get("ma50")), nq._to_float(technical.get("ma200"))
    if np.isfinite(ma50) and np.isfinite(ma200):
        cross = ("above" if ma50 > ma200 else "below")
        note(f"MA50 is {cross} MA200 "
             f"({nq.format_rupiah(ma50, compact=False)} versus "
             f"{nq.format_rupiah(ma200, compact=False)}). "
             + nq.EXPLANATIONS["technical"])
    else:
        note(nq.EXPLANATIONS["technical"])


def render_income_chart(company: dict) -> None:
    """Revenue against cost against net income, quarter by quarter."""
    section("Revenue vs Cost vs Net Income")
    frame = nq.income_statement_series(company["quarterly"])
    if frame.empty or frame[["revenue", "net_income"]].notna().sum().sum() == 0:
        st.info("The quarterly filings for this company do not carry enough "
                "income-statement detail to chart.")
        return

    cost_label = frame.attrs.get("cost_label", "Cost")
    figure = go.Figure()
    figure.add_trace(go.Bar(
        x=frame["report_date"], y=frame["revenue"], name="Revenue",
        marker={"color": ACCENT},
        hovertemplate="Rp %{y:,.0f}<extra>Revenue</extra>"))
    if frame["cost"].notna().any():
        figure.add_trace(go.Bar(
            x=frame["report_date"], y=frame["cost"], name=cost_label,
            marker={"color": COST},
            hovertemplate="Rp %{y:,.0f}<extra>" + cost_label + "</extra>"))
    figure.add_trace(go.Scatter(
        x=frame["report_date"], y=frame["net_income"], name="Net income",
        mode="lines+markers", line={"color": POSITIVE, "width": 2},
        marker={"size": 5},
        hovertemplate="Rp %{y:,.0f}<extra>Net income</extra>"))

    figure.update_layout(
        height=340, margin={"l": 0, "r": 0, "t": 30, "b": 0},
        plot_bgcolor="white", paper_bgcolor="white", barmode="group", bargap=0.25,
        hovermode="x unified",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02,
                "xanchor": "left", "x": 0, "font": {"size": 11}},
        xaxis={"showgrid": False, "linecolor": GRID},
        yaxis={"gridcolor": GRID, "title": "IDR per quarter", "zerolinecolor": GRID})
    st.plotly_chart(figure, width="stretch", config=CHART_CONFIG)

    detail = ("Cost of revenue" if cost_label == "Cost of revenue"
              else "Operating expense")
    note(f"Per quarter, not cumulative: filings that report year-to-date are "
         f"de-cumulated first, so a fourth quarter is one quarter and not the "
         f"whole year. Cost is shown as <strong>{escape(detail.lower())}</strong> "
         f"for this company — issuers that do not file a cost of revenue, banks "
         f"among them, are charted on operating expense instead, and the label "
         f"says which.")


def render_momentum_charts(prices: pd.DataFrame) -> None:
    """RSI and MACD as charts. The tiles below give the latest reading only."""
    if prices is None or prices.empty or "close" not in prices.columns:
        return
    history = prices.sort_values("date").reset_index(drop=True)
    close = history["close"].astype(float)
    if len(close) < 60:
        st.info("Not enough price history in this window to chart RSI or MACD.")
        return

    rsi = nq.rsi_series(close)
    macd = nq.macd_series(close)

    figure = make_subplots(rows=2, cols=1, shared_xaxes=True,
                           row_heights=[0.45, 0.55], vertical_spacing=0.10,
                           subplot_titles=("RSI (14)", "MACD (12, 26, 9)"))

    figure.add_trace(go.Scatter(
        x=history["date"], y=rsi, mode="lines", name="RSI",
        line={"color": ACCENT, "width": 1.4},
        hovertemplate="%{y:.0f}<extra>RSI</extra>"), row=1, col=1)
    # 70 and 30 are the conventional bands, drawn so the line has something to
    # be read against rather than floating on an empty axis.
    for level, colour in ((70, NEGATIVE), (30, POSITIVE)):
        figure.add_hline(y=level, line={"color": colour, "width": 1, "dash": "dot"},
                         opacity=.5, row=1, col=1)
    figure.update_yaxes(range=[0, 100], gridcolor=GRID, row=1, col=1,
                        tickvals=[30, 50, 70])

    colours = [POSITIVE if v >= 0 else NEGATIVE for v in macd["histogram"].fillna(0)]
    figure.add_trace(go.Bar(
        x=history["date"], y=macd["histogram"], name="Histogram",
        marker={"color": colours},
        hovertemplate="%{y:,.1f}<extra>Histogram</extra>"), row=2, col=1)
    figure.add_trace(go.Scatter(
        x=history["date"], y=macd["macd"], mode="lines", name="MACD",
        line={"color": ACCENT, "width": 1.4},
        hovertemplate="%{y:,.1f}<extra>MACD</extra>"), row=2, col=1)
    figure.add_trace(go.Scatter(
        x=history["date"], y=macd["signal"], mode="lines", name="Signal",
        line={"color": COST, "width": 1.4},
        hovertemplate="%{y:,.1f}<extra>Signal</extra>"), row=2, col=1)
    figure.update_yaxes(gridcolor=GRID, zerolinecolor=GRID, row=2, col=1)

    figure.update_layout(
        height=430, margin={"l": 0, "r": 0, "t": 40, "b": 0},
        plot_bgcolor="white", paper_bgcolor="white", bargap=0.1,
        hovermode="x unified", showlegend=False,
        xaxis={"showgrid": False, "linecolor": GRID},
        xaxis2={"showgrid": False, "linecolor": GRID})
    figure.update_annotations(font_size=12, x=0, xanchor="left")
    st.plotly_chart(figure, width="stretch", config=CHART_CONFIG)


def tile(column, label: str, value: str, band: str | None = None,
         help_text: str | None = None) -> None:
    """A metric tile whose arrow and colour follow the band's own direction.

    st.metric cannot do this: it picks the arrow from whether the delta string
    starts with a minus, so a text band like "Bearish" always came out green
    and rising — the opposite of what the number underneath it said.
    """
    direction = nq.band_direction(band) if band else 0
    arrow, colour = {1: ("\u25b2", POSITIVE), -1: ("\u25bc", NEGATIVE),
                     0: ("", MUTED)}[direction]
    hint = (f"<span class='nq-tile-help' title='{escape(help_text)}'>?</span>"
            if help_text else "")
    band_html = (f"<div class='nq-tile-band' style='color:{colour}'>"
                 f"{arrow} {escape(band)}</div>" if band else "")
    column.markdown(
        f"<div class='nq-tile'><div class='nq-tile-label'>{escape(label)}{hint}</div>"
        f"<div class='nq-tile-value'>{escape(value)}</div>{band_html}</div>",
        unsafe_allow_html=True)


def render_features(company: dict, model_features: list[str]) -> None:
    section("Fundamental metrics")
    row = company["features"].iloc[0]
    used = set(model_features or [])

    body = []
    for category in nq.CATEGORY_ORDER:
        specs = [f for f in nq.FEATURE_SCHEMA if f.category == category]
        if not specs:
            continue
        body.append(f"<tr class='grp'><td colspan='5'>{escape(category)}</td></tr>")
        for spec in specs:
            value = row.get(spec.name)
            present = np.isfinite(nq._to_float(value))
            # Three states, not two. "Reference" is for the rupiah amounts and
            # the per-share figures: they are shown because a reader wants them,
            # but a level cannot be a cross-sectional input — a bank with IDR
            # 1,600T of assets and a small cap with IDR 2T are not on one scale.
            if not spec.point_in_time:
                in_model = "Snapshot"
            elif not spec.modelled:
                in_model = "Reference"
            elif spec.name in used:
                in_model = "Yes"
            else:
                in_model = "No (dropped)" if used else "Unknown"
            meaning = (spec.meaning if present
                       else f"Not available. {nq.feature_absence_reason(spec.name)}")
            expansion = (f"<span class='sub'>{escape(spec.expansion)}</span>"
                         if spec.expansion else "")
            muted = "" if present else " na"
            body.append(
                "<tr>"
                f"<td><strong>{escape(spec.label)}</strong>{expansion}</td>"
                f"<td class='num{muted}'>"
                f"{escape(nq.format_feature(spec.name, value))}</td>"
                f"<td>{escape(spec.unit.title())}</td>"
                f"<td>{escape(in_model)}</td>"
                f"<td class='{muted.strip()}'>{escape(meaning)}</td>"
                "</tr>")

    st.markdown(
        "<table class='nq-table nq-metrics'>"
        "<colgroup><col style='width:20%'><col style='width:13%'>"
        "<col style='width:11%'><col style='width:13%'><col style='width:43%'>"
        "</colgroup>"
        "<thead><tr><th>Metric</th><th>Value</th><th>Unit</th>"
        "<th>In model</th><th>Meaning</th></tr></thead>"
        f"<tbody>{''.join(body)}</tbody></table>",
        unsafe_allow_html=True)

    modelled = sum(1 for f in nq.FEATURE_SCHEMA if f.modelled)
    as_of = nq.screen_as_of()
    screened = f" taken on {as_of}" if as_of else ""
    note(f"Values are as reported for the latest available financial period. A "
         f"high or low reading is not automatically good or bad — the machine "
         f"learning model "
         f"weighs these together rather than applying a rule to any single one."
         f"<br><br><strong>In model</strong> has three states. "
         f"<em>Yes</em> and <em>No (dropped)</em> apply to the {modelled} "
         f"scale-free ratios the machine learning model may read; dropped means the "
         f"ratio was missing for more than {nq.MAX_FEATURE_MISSINGNESS:.0%} of "
         f"the training panel, so training left it out rather than impute its "
         f"way around the gap. <em>Reference</em> is everything measured in "
         f"rupiah: shown because a reader wants it, never modelled, because a "
         f"level would let the machine learning model split on company size rather "
         f"than on value."
         f"<br><br><em>Snapshot</em> marks the dividend figures, which come "
         f"from the screener as trailing values{screened} rather than "
         f"reconstructed at each historical date. They are shown and never "
         f"modelled: feeding today's yield to a 2022 observation would be "
         f"look-ahead of exactly the kind the leakage audit exists to catch."
         f"<br><br>A dash is a metric this company did not file for the period, "
         f"or one that would be economically meaningless — a P/E on a loss "
         f"is a category error, not a cheap stock — and the Meaning column "
         f"says which. NPL, LDR, NIM and the dividend ratios are not listed at "
         f"all: the quarterly endpoint returns none of the fields they need.")


def render_prediction(result: dict, artifact: dict | None, horizon: str) -> None:
    months = "6" if horizon == "6m" else "12"
    section(f"{months}-month outlook")

    if not result.get("available"):
        st.warning(result.get("reason", "Prediction unavailable."))
        if result.get("missing"):
            with st.expander("Technical details"):
                st.code(", ".join(result["missing"]), language="text")
        return

    probability = result["probability"]
    reliability = (artifact or {}).get("reliability", {})
    metrics = (artifact or {}).get("validation_metrics", {})
    has_edge = bool((artifact or {}).get("has_edge", reliability.get("has_edge", True)))

    left, right = st.columns([1, 2])
    with left:
        st.metric("Probability of positive return", f"{probability * 100:.0f}%",
                  help=nq.TOOLTIPS["probability"])
        st.caption(nq.probability_band(probability, has_edge))
    with right:
        probability_bar(probability)
        note(nq.explain_probability(probability, horizon, has_edge))

    columns = st.columns(4)
    columns[0].metric("Machine learning model reliability",
                      reliability.get("label", "Unknown"),
                      help=nq.TOOLTIPS["reliability"])
    columns[1].metric("Out-of-sample ROC-AUC",
                      f"{metrics.get('roc_auc', float('nan')):.3f}"
                      if np.isfinite(nq._to_float(metrics.get("roc_auc"))) else "—",
                      help=nq.TOOLTIPS["roc_auc"])
    columns[2].metric("Validation folds", (artifact or {}).get("validation_folds", "—"),
                      help=nq.TOOLTIPS["folds"])
    columns[3].metric("Data quality", f"{result.get('data_quality', 0) * 100:.0f}%",
                      help=nq.TOOLTIPS["data_quality"])

    if not has_edge:
        # The panel size is read from the artifact rather than written into the
        # sentence: it was hard-coded at 15 and silently went stale the first
        # time the universe grew.
        tickers = (artifact or {}).get("n_training_tickers")
        panel = f" — {tickers} tickers" if tickers else ""
        st.warning(
            f"**This machine learning model has no measurable edge.** Across purged "
            "walk-forward "
            f"folds it did not rank winners above losers by more than chance, so "
            f"its probabilities are deliberately shrunk toward the historical "
            f"base rate. Read the number as *how often stocks in this universe "
            f"rose over this horizon*, not as a view on this company. The limit "
            f"is the size of the training panel{panel} — not the algorithm.")
    elif reliability.get("label") == "Weak":
        st.warning("This machine learning model provides limited predictive separation on "
                   "out-of-sample data. Treat the probability as weak evidence, "
                   "not a signal.")

    # The validation expander used to sit here: leaderboard, per-fold metrics,
    # feature importances. It was removed because it read as a wall of numbers
    # to anyone who had not just trained the model. Nothing is hidden — the
    # figures above still name the reliability, the out-of-sample AUC and the
    # fold count, and the full record lives in the artifact and in train.py's
    # output for anyone auditing the repository.


def render_risk(prices: pd.DataFrame, window: str) -> None:
    section("Historical risk")
    if prices.empty:
        st.info("Risk cannot be measured without price history."); return
    years = PRICE_WINDOWS[window]
    metrics = nq.risk_metrics(prices, years)
    risk = nq.risk_score(metrics)

    columns = st.columns(5)
    columns[0].metric("Risk", risk["band"], help=nq.TOOLTIPS["risk_band"])
    columns[1].metric("Annualised volatility",
                      nq.format_percent(metrics["volatility"], 0),
                      help=nq.TOOLTIPS["volatility"])
    columns[2].metric("Maximum drawdown",
                      nq.format_percent(metrics["max_drawdown"], 0),
                      help=nq.TOOLTIPS["max_drawdown"])
    columns[3].metric("Downside volatility",
                      nq.format_percent(metrics["downside_volatility"], 0),
                      help=nq.TOOLTIPS["downside_volatility"])
    # Liquidity carries 10% of the risk score and was computed all along, but
    # the panel showed four of its five inputs and left this one invisible.
    turnover = nq._to_float(metrics.get("turnover"))
    columns[4].metric("Daily turnover",
                      nq.format_percent(turnover, 2) if np.isfinite(turnover)
                      else "Volume not reported",
                      help=nq.TOOLTIPS["turnover"])
    note(f"Measured over {years} year{'s' if years > 1 else ''}. " + nq.EXPLANATIONS["risk"])

    with st.expander("Drawdown"):
        history = prices.sort_values("date")
        close = history["close"].astype(float)
        figure = go.Figure(go.Scatter(
            x=history["date"], y=close / close.cummax() - 1.0, mode="lines",
            line={"color": NEGATIVE, "width": 1.2}, fill="tozeroy",
            fillcolor="rgba(179,52,31,.10)",
            hovertemplate="%{x|%d %b %Y}<br>%{y:.1%}<extra></extra>"))
        figure.update_layout(height=240, margin={"l": 0, "r": 0, "t": 10, "b": 0},
                             plot_bgcolor="white", paper_bgcolor="white", showlegend=False,
                             xaxis={"showgrid": False, "linecolor": GRID},
                             yaxis={"gridcolor": GRID, "tickformat": ".0%"})
        st.plotly_chart(figure, width="stretch", config=CHART_CONFIG)


def render_single_stock(companies: pd.DataFrame, models: dict, controls: dict) -> None:
    section(MODE_SINGLE)
    if companies.empty:
        st.error("No companies available."); return

    labels = {f"{r.symbol} — {r.company_name}": r.symbol for r in companies.itertuples()}
    # Alphabetical here, deliberately against the market-cap order the frame
    # arrives in. The size filters need largest-first because they take the top
    # N; this list is read by someone hunting for one name they already have in
    # mind, and for that the alphabet beats any ranking.
    keys = sorted(labels)
    chosen = st.session_state.get("ticker")
    index = next((i for i, k in enumerate(keys) if labels[k] == chosen), 0)

    left, right = st.columns([3, 1])
    selection = left.selectbox(
        "Select stock", keys, index=index,
        help="Type to filter by ticker or company name.")
    if right.button("Analyze", width="stretch", type="primary"):
        st.session_state["ticker"] = labels[selection]
        st.session_state["analysed"] = labels[selection]
    ticker = labels[selection]
    st.caption(f"{len(keys)} companies, listed A–Z. Click the box and type to "
               f"search by ticker or by name.")

    if st.session_state.get("analysed") != ticker:
        st.info("Select a stock and press **Analyze**.")
        st.caption("Cached mode — 0 credits." if controls["offline"]
                   else f"About {CREDITS_PER_COMPANY} credits per analysis.")
        return

    try:
        with st.spinner(f"Loading {ticker}…"):
            company = load_company(ticker, controls["api_key"], controls["offline"])
    except nq.SectorsAPIError as error:
        show_error(error); return

    if company["quarterly"].empty:
        st.error(f"No quarterly financial data available for {ticker}."); return

    predictions = {h: predict(company["features"], models.get(h), h)
                   for h in nq.HORIZON_TRADING_DAYS}
    render_profile(company, predictions)

    if company["n_quarters"] < nq.MIN_QUARTERS_FOR_PREDICTION:
        st.warning(f"Only {company['n_quarters']} quarterly reports available. "
                   f"NusaQuant wants at least {nq.MIN_QUARTERS_FOR_PREDICTION} "
                   f"before treating a fundamental prediction as meaningful.")

    prices, window, technical = render_chart(
        company, controls["api_key"], controls["offline"])
    render_momentum_charts(prices)
    render_technical(technical, window)
    render_income_chart(company)
    model_features = list((models.get("6m") or {}).get("feature_names", []))
    render_features(company, model_features)

    if not models:
        render_missing_models()
    else:
        render_prediction(predictions["6m"], models.get("6m"), "6m")
        render_prediction(predictions["12m"], models.get("12m"), "12m")

    render_risk(prices, window)
    st.caption(nq.DISCLAIMER)


# ══════════════════════════════════════════════════════════════════════
# BEST 10
# ══════════════════════════════════════════════════════════════════════

def render_best_10(companies: pd.DataFrame, models: dict, controls: dict) -> None:
    section(MODE_PICKS)
    if not models:
        render_missing_models(); return

    horizon = "6m" if st.radio("Horizon", ["6 Months", "12 Months"], horizontal=True,
                               label_visibility="collapsed") == "6 Months" else "12m"
    artifact = models.get(horizon)
    if not artifact:
        st.info("No machine learning model for this horizon."); return

    if not artifact.get("has_edge", True):
        st.warning(
            "**This ranking is not evidence.** The machine learning model for this "
            "horizon showed "
            "no measurable out-of-sample edge, so the order below reflects how it "
            "sorts fundamentals in training, not a validated ability to pick "
            "winners. It is shown for inspection of the pipeline, not as a "
            "shortlist to act on.")

    # step=1, not 5. The universe holds 19 companies, and a step of 5 made 19
    # unreachable: the slider stopped at 15 and quietly excluded four of them
    # from every ranking, with nothing on screen to say so.
    largest = max(5, len(companies))
    # A bordered container rather than st.form, which is what Sector Ranking
    # uses: a form holds its widget values back until submit, and the live-mode
    # cost estimate below has to move with the slider. The border is the same.
    with st.container(border=True):
        size = st.slider("Universe size", 5, largest, min(len(companies), largest),
                         step=1,
                         help=UNIVERSE_SIZE_HELP.format(total=len(companies)))
        st.caption(UNIVERSE_SIZE_NOTE)
        if controls["offline"]:
            st.caption("Cached mode — this ranking costs 0 API credits.")
        else:
            st.caption(f"Estimated cost: about "
                       f"{size * CREDITS_PER_COMPANY:,} API credits.")
        go = st.button("Show top picks", type="primary")

    if not go:
        st.info("Press the button to rank the universe."); return

    universe = companies.head(size)
    rows, metrics_by_ticker = [], {}
    progress = st.progress(0.0, text="Scoring…")

    for index, company_row in enumerate(universe.itertuples(), start=1):
        progress.progress(index / len(universe), text=f"Scoring {company_row.symbol}")
        record = {"ticker": company_row.symbol, "company_name": company_row.company_name,
                  "probability": np.nan, "risk": "Not measured",
                  "trend": "Not measured", "data_quality": 0.0,
                  "eligible": False, "reason": ""}
        try:
            company = load_company(company_row.symbol, controls["api_key"], controls["offline"])
        except nq.SectorsAPIError as error:
            record["reason"] = error.message; rows.append(record); continue

        if company["n_quarters"] < nq.MIN_QUARTERS_FOR_PREDICTION:
            record["reason"] = (f"Only {company['n_quarters']} quarterly reports "
                                f"(minimum {nq.MIN_QUARTERS_FOR_PREDICTION})")
            rows.append(record); continue

        result = predict(company["features"], artifact, horizon)
        if not result.get("available"):
            record["reason"] = result.get("reason", "unavailable")
            record["data_quality"] = result.get("data_quality", 0.0)
            rows.append(record); continue

        record.update({"probability": result["probability"], "eligible": True,
                       "data_quality": result["data_quality"]})
        if not company["prices"].empty:
            metrics_by_ticker[company_row.symbol] = nq.risk_metrics(company["prices"], 1)
            record["trend"] = nq.technical_state(company["prices"]).get("trend",
                                                                       "Not measured")
        rows.append(record)
    progress.empty()

    ranked = pd.DataFrame(rows).sort_values("probability", ascending=False,
                                            na_position="last").reset_index(drop=True)

    # Risk is ranked across the shortlist, so "High" means high relative to
    # these peers rather than against an arbitrary absolute threshold.
    peers = list(metrics_by_ticker.values())
    for ticker, metrics in metrics_by_ticker.items():
        ranked.loc[ranked.ticker == ticker, "risk"] = nq.risk_score(metrics, peers)["band"]

    qualified = ranked[ranked.eligible]
    if qualified.empty:
        st.error("No company met NusaQuant's minimum data-quality criteria."); return
    if len(qualified) < 10:
        st.info(f"Only {len(qualified)} stocks meet the minimum data-quality "
                f"criteria, so the table is shorter than ten.")

    top = qualified.head(10)
    months = "6" if horizon == "6m" else "12"
    reliability = artifact.get("reliability", {}).get("label", "Unknown")
    inputs = len(artifact.get("feature_names", []))
    st.markdown(f"#### Top {len(top)} — {months} month outlook")
    st.dataframe(pd.DataFrame({
        "Rank": range(1, len(top) + 1),
        "Ticker": top.ticker.to_numpy(),
        "Company": top.company_name.to_numpy(),
        "Probability up": [f"{v * 100:.0f}%" for v in top.probability],
        "Risk": top.risk.to_numpy(),
        "Trend": top.trend.to_numpy(),
        "Data quality": [f"{v * 100:.0f}%" for v in top.data_quality],
    }), width="stretch", hide_index=True, column_config={
        "Rank": st.column_config.NumberColumn(width="small",
                                              help=nq.TOOLTIPS["rank"]),
        "Ticker": st.column_config.TextColumn(width="small",
                                              help=nq.TOOLTIPS["ticker"]),
        "Company": st.column_config.TextColumn(width="large",
                                               help=nq.TOOLTIPS["company"]),
        "Probability up": st.column_config.TextColumn(
            help=nq.TOOLTIPS["probability"]),
        "Risk": st.column_config.TextColumn(help=nq.TOOLTIPS["risk_column"]),
        "Trend": st.column_config.TextColumn(help=nq.TOOLTIPS["trend"]),
        "Data quality": st.column_config.TextColumn(
            help=nq.TOOLTIPS["data_quality"])})

    # Reliability is a property of the model, not of a row, so repeating it
    # down every line of the table only made the columns narrower.
    note(f"<strong>How to read this table.</strong> Stocks are ranked by the "
         f"machine learning model's estimated probability of a positive return "
         f"over the horizon. "
         f"Machine learning model reliability for this horizon is "
         f"<strong>{reliability}</strong>, "
         f"and it applies to every row equally. Risk and trend are measured "
         f"from price history alone, independently of the machine learning model, "
         f"because a high "
         f"probability does not automatically mean low risk."
         f"<br><br><strong>Data quality</strong> is the share of the "
         f"{inputs} inputs the machine learning model reads that "
         f"are actually present for that company this period — {inputs} "
         f"of {inputs} is 100%. It measures the company's filing, "
         f"not the model: a company can score 100% and still sit under a model "
         f"with no measurable edge, which is the case here. It is shown because "
         f"a probability built on half the inputs deserves less weight than one "
         f"built on all of them, and NusaQuant refuses to score a company at all "
         f"below {nq.MIN_DATA_COMPLETENESS:.0%} — those appear under "
         f"<em>Excluded by quality gates</em> rather than in the ranking.")

    excluded = ranked[~ranked.eligible]
    if not excluded.empty:
        with st.expander(f"Excluded by quality gates ({len(excluded)})"):
            st.dataframe(excluded[["ticker", "company_name", "reason"]],
                         width="stretch", hide_index=True)
    st.caption(nq.DISCLAIMER)


# ══════════════════════════════════════════════════════════════════════
# SECTOR RANKING
# ══════════════════════════════════════════════════════════════════════

RANKABLE = ["pe", "ps", "pbv", "pcf", "ev_ebitda", "der", "roa", "roe",
            "gpm", "opm", "npm", "dividend_yield", "dpr"]


def cached_sector_table(tickers: list[str]) -> pd.DataFrame:
    """NusaQuant's own ratios for cached companies, one row each."""
    rows = []
    for ticker in tickers:
        quarterly = snapshot_data(ticker, "quarterly")
        prices = snapshot_data(ticker, "prices")
        if quarterly.empty or prices.empty:
            continue
        latest = prices.sort_values("date").iloc[-1]
        metrics = nq.features_frame(quarterly,
                                    nq._to_float(latest.get("market_cap")),
                                    nq._to_float(latest.get("close"))).iloc[0]
        # The dividend figures are a screener snapshot rather than something
        # compute_features can reconstruct, so they are merged in here.
        rows.append({"symbol": ticker, "company_name": nq.company_name(ticker),
                     "market_cap": nq._to_float(latest.get("market_cap")),
                     **{m: nq._to_float(metrics.get(m)) for m in RANKABLE},
                     **nq.company_dividends(ticker)})
    return pd.DataFrame(rows)


def render_sector_ranking(controls: dict) -> None:
    section(MODE_SECTOR)
    universe = nq.load_universe()
    if universe.empty or "sector" not in universe.columns:
        st.warning("The cached universe carries no IDX classification yet. "
                   "Run `python train.py --screen` once — it costs 1 credit "
                   "and covers every company in the screen.")
        return

    cached = [t for t in by_market_cap(
        pd.DataFrame({"symbol": sorted(nq.cached_tickers())}))["symbol"]]
    available = len(cached) if controls["offline"] else len(universe)

    # The size is chosen and submitted before any grouping is offered: the
    # sector counts depend on it, so showing them first would mean showing
    # numbers that change the moment the slider moves.
    with st.form("sector_universe"):
        size = st.slider("Universe size", 5, max(5, available),
                         min(available, max(5, available)), step=1,
                         help=UNIVERSE_SIZE_HELP.format(total=available))
        st.caption(UNIVERSE_SIZE_NOTE)
        submitted = st.form_submit_button("Submit", type="primary")
    if submitted:
        st.session_state["sector_size"] = size
    if "sector_size" not in st.session_state:
        st.info("Choose a universe size and press **Submit**.")
        return
    size = st.session_state["sector_size"]

    level_label = st.radio("Group by", ["Sector", "Sub-sector"], horizontal=True)
    level = "sector" if level_label == "Sector" else "sub_sector"

    if controls["offline"]:
        # Only companies with collected data can be compared on NusaQuant's own
        # ratios, so the picker counts those rather than the whole screen.
        scope = cached[:size]
        pool = universe[universe["symbol"].isin(scope)]
        counts = pool[level].value_counts()
        if counts.empty:
            st.info("No cached company carries a classification yet.")
            return
        st.caption(f"{len(scope)} of {len(cached)} cached companies in scope, "
                   f"the largest by market capitalisation.")
        options = [f"{name} ({n})" for name, n in counts.items()]
        chosen = st.selectbox(f"{level_label} — the number in brackets is how "
                              f"many companies it holds within this universe",
                              options)
        name = chosen.rsplit(" (", 1)[0]
        members = sorted(pool.loc[pool[level] == name, "symbol"])
        with st.spinner(f"Scoring {len(members)} companies…"):
            table = cached_sector_table(members)
        source = ("NusaQuant's own point-in-time ratios, computed from the "
                  "cached snapshot at zero credits")
    else:
        names = sorted(universe[level].dropna().unique())
        name = st.selectbox(level_label, names)
        st.caption(f"Live mode — this screen costs 1 API credit and returns up "
                   f"to {size} companies in the group, largest by market "
                   f"capitalisation first, not only the cached ones.")
        if not st.button("Screen this group", type="primary"):
            st.info("Press the button to run the screen."); return
        try:
            table = nq.get_sector_peers(controls["api_key"], level=level,
                                        name=name, limit=size)
        except nq.SectorsAPIError as error:
            show_error(error); return
        source = ("Sectors' own screener ratios (pe_ttm, pb_mrq, ps_ttm, "
                  "der_mrq, roa_ttm, roe_ttm), as of the screen date")

    if table.empty:
        st.info("No company came back for this group."); return

    present = [m for m in RANKABLE if m in table.columns
               and table[m].notna().sum() >= 1]
    if not present:
        st.info("No ratio is available for this group."); return

    rankable = [m for m in present if table[m].notna().sum() >= 2]
    if not rankable:
        st.warning(f"Only one company in {name} has a usable ratio, so there "
                   f"is nothing to rank it against. The figures are shown, but "
                   f"a peer comparison needs at least two.")

    metric = st.selectbox(
        "Rank by",
        rankable or present,
        format_func=lambda m: (f"{nq.FEATURE_BY_NAME[m].title} — "
                               f"{'lowest first' if nq.rank_ascending(m) else 'highest first'}"))

    ordered = table.sort_values(metric, ascending=nq.rank_ascending(metric),
                                na_position="last").reset_index(drop=True)
    display = pd.DataFrame({
        "Rank": range(1, len(ordered) + 1),
        "Ticker": ordered["symbol"].to_numpy(),
        "Company": ordered["company_name"].to_numpy(),
        "Market cap": [nq.format_rupiah(v) for v in ordered["market_cap"]],
    })
    for m in present:
        display[nq.FEATURE_BY_NAME[m].label] = [
            nq.format_feature(m, v) for v in ordered[m]]
    st.dataframe(display, width="stretch", hide_index=True, column_config={
        "Rank": st.column_config.NumberColumn(width="small"),
        "Ticker": st.column_config.TextColumn(width="small"),
        "Company": st.column_config.TextColumn(width="medium")})

    # The median is the comparison that matters: a P/E of 14 means nothing
    # until you know the sector trades at 9.
    # A median needs a middle. With one reporting company the "median" is just
    # that company's own number wearing a peer-group label, which is worse than
    # showing nothing: it invites a comparison against itself.
    comparable = [m for m in present if ordered[m].notna().sum() >= 2]
    if comparable:
        st.markdown(f"##### {name} — peer median "
                    f"({len(ordered)} companies)")
        columns = st.columns(min(len(comparable), 6))
        for index, m in enumerate(comparable):
            reporting = int(ordered[m].notna().sum())
            columns[index % len(columns)].metric(
                nq.FEATURE_BY_NAME[m].label,
                nq.format_feature(m, ordered[m].median(skipna=True)),
                help=f"Median of the {reporting} companies in {name} that "
                     f"report it.")

    if len(ordered) >= 2:
        with st.expander("Percentile against peers"):
            percentile = pd.DataFrame({"Ticker": ordered["symbol"].to_numpy()})
            for m in present:
                percentile[nq.FEATURE_BY_NAME[m].label] = (
                    nq.peer_percentile(ordered[m], m).round(0).to_numpy())
            st.dataframe(percentile, width="stretch", hide_index=True)
            note("100 is the best reading in the group and 0 the worst. For a "
                 "multiple that means cheapest or least indebted; for a "
                 "percentage, most profitable. A better ratio is not the same "
                 "thing as a better investment.")

    if any(m in present for m in ("dividend_yield", "dpr")):
        as_of = nq.screen_as_of()
        note(f"Dividend yield and payout are trailing figures from the Sectors "
             f"screener{f', taken on {as_of}' if as_of else ''}, not "
             f"point-in-time history. A company with no reading either pays "
             f"nothing or was not covered by the screen — the two are not "
             f"distinguished, so an absent yield is shown as a dash rather "
             f"than as zero.")

    note(f"<strong>Source.</strong> {source}. Comparing within a sector is the "
         f"point: a bank's balance sheet and a miner's are not alike, and a "
         f"ratio that looks extreme across the whole market is often ordinary "
         f"beside its own peers. This view is descriptive and has no machine "
         f"learning model in it.")
    st.caption(nq.DISCLAIMER)


# ══════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════

def main() -> None:
    configure_page()
    st.markdown('<div class="nq-title">NusaQuant</div>', unsafe_allow_html=True)
    st.markdown('<div class="nq-sub">IDX Machine Learning Market Intelligence · XGBoost probability '
                'estimates for positive 6-month and 12-month returns.</div>',
                unsafe_allow_html=True)

    models, metadata = load_models()
    controls = render_sidebar(metadata)

    if controls["offline"]:
        names = nq.company_names()
        companies = by_market_cap(pd.DataFrame({
            "symbol": controls["snapshot"],
            "company_name": [names.get(t, t) for t in controls["snapshot"]]}))
        st.info(f"Cached mode — no API credits are being spent. Figures are a "
                f"real Sectors snapshot taken on {snapshot_as_of()}, not today's "
                f"market.")
    else:
        if not controls["api_key"]:
            st.info(nq.WELCOME)
            st.caption(nq.DISCLAIMER)
            footer()
            return
        try:
            companies = live_universe(
                controls["api_key"],
                metadata.get("universe_filter") or "market_cap > 1000000000000", 50)
        except nq.SectorsAPIError as error:
            show_error(error); return
        if companies.empty:
            st.error("The Sectors universe came back empty."); return

    if controls["mode"] == MODE_SINGLE:
        render_single_stock(companies, models, controls)
    elif controls["mode"] == MODE_SECTOR:
        render_sector_ranking(controls)
    else:
        render_best_10(companies, models, controls)
    footer()


if __name__ == "__main__":
    main()
