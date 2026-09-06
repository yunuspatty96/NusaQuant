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
from typing import Any, Sequence

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
MODE_PICKS = "Machine Learning Screening"
MODE_PORTFOLIO = "Portfolio Analysis"

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

#: Three models now, and the third is the only one with a measurable edge.
#: "risk" is keyed like a horizon because everything downstream — predict(),
#: the reliability panels, the artifact layout — already speaks that language.
MODEL_FILES: dict[str, str] = {"6m": "model_6m_xgb.joblib",
                               "12m": "model_12m_xgb.joblib",
                               "risk_6m": "model_risk_6m_xgb.joblib",
                               "risk_12m": "model_risk_12m_xgb.joblib"}


@st.cache_resource(show_spinner=False)
def load_models() -> tuple[dict[str, Any], dict[str, Any]]:
    """Load the exported artifacts once per session, not once per rerun."""
    models: dict[str, Any] = {}
    for horizon, filename in MODEL_FILES.items():
        path = MODELS_DIR / filename
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
    for filename in MODEL_FILES.values():
        path = MODELS_DIR / filename
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
        missing = ("No trained volatility model was found."
                   if horizon.startswith("risk")
                   else f"No trained machine learning model for the {horizon} horizon.")
        return {"available": False, "reason": missing}
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


def with_price_features(features: pd.DataFrame,
                        prices: pd.DataFrame) -> pd.DataFrame:
    """Add the price-derived inputs the risk model reads.

    Computed at the last bar of whatever series is on screen, through the same
    nq.price_features the training script called at every historical
    observation — so "volatility" cannot quietly mean a 63-day window in
    training and a 90-day one here.
    """
    frame = features.copy()
    if prices is None or prices.empty:
        for name in ("vol_3m", "dist_52w_high", "reversal_1m"):
            frame[name] = np.nan
        return frame
    ordered = prices.sort_values("date").reset_index(drop=True)
    for name, value in nq.price_features(ordered, len(ordered) - 1).items():
        frame[name] = value
    return frame


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
      .nq-note {{ color:{MUTED}; font-size:.84rem; line-height:1.55;
                  margin:.9rem 0 .2rem; }}
      .nq-note + .nq-note {{ margin-top:1.1rem; }}
      .nq-note p {{ margin:0 0 .7rem; }}
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
      .nq-disc {{ border:1.5px solid {NEGATIVE}; border-radius:6px;
                  padding:.75rem .9rem; margin:1.6rem 0 .5rem;
                  background:rgba(179,52,31,.04); }}
      .nq-disc-title {{ color:{NEGATIVE}; font-weight:700; font-size:.82rem;
                        letter-spacing:.06em; margin-bottom:.35rem; }}
      .nq-disc-body {{ color:{MUTED}; font-size:.82rem; line-height:1.6; }}
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


def disclaimer() -> None:
    """The standing warning, boxed so it is not read as a caption."""
    st.markdown(
        "<div class='nq-disc'><div class='nq-disc-title'>DISCLAIMER!</div>"
        f"<div class='nq-disc-body'>{escape(nq.DISCLAIMER)}</div></div>",
        unsafe_allow_html=True)


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
        st.error("**The estimates could not be loaded.**\n\n"
                 "The saved files are present but unreadable in this "
                 "environment, so probabilities cannot be shown. Everything "
                 "measured from price history still works.")
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
            st.success(f"Figures as of {snapshot_as_of()}\n\n"
                       f"{len(snapshot)} companies covered")
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
        mode = st.radio("Analysis", [MODE_SINGLE, MODE_PICKS, MODE_PORTFOLIO],
                        label_visibility="collapsed")

        st.divider()
        if metadata:
            st.caption(f"Estimates built from company filings up to "
                       f"{metadata.get('training_end_date','—')}, across "
                       f"{metadata.get('n_tickers','—')} companies.")
        else:
            st.caption("No trained models loaded.")

    return {"offline": offline, "api_key": api_key, "snapshot": snapshot,
            "mode": mode}


# ══════════════════════════════════════════════════════════════════════
# SINGLE STOCK
# ══════════════════════════════════════════════════════════════════════

def render_profile(company: dict, risks: dict, peers: dict) -> None:
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
    # The two return probabilities used to sit here, at the top of the page,
    # in the position a reader treats as the answer. They were never able to
    # carry that: neither horizon beat chance out of sample. What replaced them
    # is the one estimate that did. The return figures still appear, lower
    # down, with the label the testing supports.
    #
    # Only the forecast is repeated from the Risk Analysis section, and only
    # because the top of a page is where a reader looks first. The measured
    # volatility is NOT repeated here: two identical numbers in two places is
    # how a reader ends up wondering which one to believe.
    # The band, not the probability. Risk Analysis further down carries the
    # figure itself for both horizons, and printing the same number in two
    # places is how a reader ends up asking which one to believe. What the top
    # of a page is good for is the one-glance answer.
    for column, horizon in zip(columns[2:], nq.RISK_HORIZONS):
        months = 6 if horizon.endswith("6m") else 12
        result = risks.get(horizon, {})
        if result.get("available"):
            column.metric(f"{months}M Risk Class",
                          nq.volatility_class(result["probability"],
                                              peers.get(horizon, [])),
                          help=nq.TOOLTIPS["risk_class"])
        else:
            # Never a bare dash: an unavailable estimate has a cause, and the
            # cause is more useful than the punctuation.
            column.metric(f"{months}M Risk Class", "Not available",
                          help=result.get("reason",
                                          "Volatility forecast unavailable."))

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

    # Moving averages are drawn only where they are fully formed. A 50-day
    # average seeded from 30 days of data is not a 50-day average.
    averages = nq.moving_averages(history)
    # `period`, not `window`: this loop used to shadow the selected price
    # window, leaving it set to the last average by the time it returned.
    for period, colour, dash in ((20, ACCENT, "solid"), (50, MUTED, "dot")):
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
        # Only the 50% band is drawn. The 80% band is correctly calibrated —
        # for the most volatile observations it caught 74.9% at six months
        # against a target of 80%, so if anything it is narrow — but half this
        # panel's 12-month 80% ranges span more than five times bottom to top,
        # and MORA's spans sixty-five. Shading that would stretch the axis
        # until the price line became a flat scratch. It is tabulated instead.
        # A stepped path rather than three anchors: the width grows with the
        # square root of time, so straight lines between today, +6m and +12m
        # understate it in between — and three points give hover nothing to
        # report at any date the reader actually points at.
        path = nq.cone_path(cone, level=50)
        dates = [last_date + pd.Timedelta(days=int(d * 365 / 252))
                 for d in path["days"]]
        figure.add_trace(go.Scatter(
            x=dates + dates[::-1],
            y=path["high"].tolist() + path["low"].tolist()[::-1],
            fill="toself", fillcolor="rgba(29,78,111,.18)", line={"width": 0},
            hoverinfo="skip", name="50% projected range"), row=1, col=1)
        for column, label in (("high", "50% range high"),
                              ("low", "50% range low")):
            figure.add_trace(go.Scatter(
                x=dates, y=path[column], mode="lines", name=label,
                line={"color": ACCENT, "width": 1, "dash": "dot"},
                opacity=.55, showlegend=False,
                hovertemplate="Rp %{y:,.0f}<extra>" + label + "</extra>"),
                row=1, col=1)
        span = [min(span[0], float(path["low"].min())),
                max(span[1], float(path["high"].max()))]

    # Support and resistance: horizontal levels the price has turned at, drawn
    # across the history only. They describe the past and are not projected.
    levels = nq.support_resistance(company["prices"]
                                   if not company["prices"].empty else prices)
    for kind, colour in (("resistance", NEGATIVE), ("support", POSITIVE)):
        for level in levels.get(kind, []):
            price = level["price"]
            if not (span[0] * 0.75 <= price <= span[1] * 1.25):
                continue
            figure.add_trace(go.Scatter(
                x=[history["date"].iloc[0], history["date"].iloc[-1]],
                y=[price, price], mode="lines", showlegend=False,
                line={"color": colour, "width": 1, "dash": "dash"},
                opacity=.45,
                hovertemplate=(f"Rp {price:,.0f}<extra>{kind.title()} · "
                               f"{level['touches']} touches</extra>")),
                row=1, col=1)
            span = [min(span[0], price), max(span[1], price)]

    # The axis is padded after every trace is in, so the cone and the levels
    # both fit rather than being clipped by a range computed before them.
    pad = max((span[1] - span[0]) * 0.06, span[1] * 0.02, 1.0)
    figure.update_layout(
        height=440 if has_volume else 360,
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

    if cone.get("available"):
        st.markdown("##### Projected range")
        rows = []
        for horizon, label in ((126, "6 months"), (252, "12 months")):
            low, high = cone["bands"][horizon][50]
            span_ratio = high / low if low > 0 else float("inf")
            wide = cone.get("too_wide", {}).get(horizon)
            caveat = (f"<span class='sub'>{span_ratio:.0f}x wide &mdash; too "
                      f"broad to act on</span>" if wide else "")
            rows.append(
                f"<tr{' class=' + chr(39) + 'na' + chr(39) if wide else ''}>"
                f"<td><strong>{label}</strong></td>"
                f"<td>half the time{caveat}</td>"
                f"<td class='num'>{escape(nq.format_rupiah(low, compact=False))}"
                f" &ndash; {escape(nq.format_rupiah(high, compact=False))}</td>"
                f"<td class='num'>{(high / cone['last'] - 1) * 100:+.0f}% / "
                f"{(low / cone['last'] - 1) * 100:+.0f}%</td>"
                "</tr>")
        st.markdown(
            "<table class='nq-table nq-cone'>"
            "<colgroup><col style='width:18%'><col style='width:24%'>"
            "<col style='width:34%'><col style='width:24%'></colgroup>"
            "<thead><tr><th>Horizon</th><th>Lands inside</th>"
            "<th>Price range</th><th>Versus today</th></tr></thead>"
            f"<tbody>{''.join(rows)}</tbody></table>", unsafe_allow_html=True)
        if any(cone.get("too_wide", {}).values()):
            note("Volatility here is high enough that the 12M range covers "
                 "almost any outcome. The 6M figure is the usable one.")

    change = (last / first - 1.0) if first > 0 else np.nan
    note(f"{nq.format_rupiah(first, compact=False)} → "
         f"{nq.format_rupiah(last, compact=False)} "
         f"({nq.format_percent(change, 1)}) over this window. "
         f"Drag to zoom, double-click to reset.")
    if any(levels.get(k) for k in ("support", "resistance")):
        note("Dashed lines are support (green) and resistance (red).")
    if cone.get("available"):
        note(nq.EXPLANATIONS["cone"])
    elif project:
        note("A projected range needs a year of daily prices. "
             "This company does not have one yet.")
    return prices, window, technical


def render_technical(technical: dict, window: str) -> None:
    """Descriptive trend state. Deliberately kept apart from the model."""
    section("Trend & Momentum")
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

    ma20, ma50 = nq._to_float(technical.get("ma20")), nq._to_float(technical.get("ma50"))
    if np.isfinite(ma20) and np.isfinite(ma50):
        cross = "above" if ma20 > ma50 else "below"
        note(f"MA20 {nq.format_rupiah(ma20, compact=False)} is {cross} MA50 "
             f"{nq.format_rupiah(ma50, compact=False)}. "
             + nq.EXPLANATIONS["technical"])
    else:
        note(nq.EXPLANATIONS["technical"])


def render_range_strip(conditions: dict, position: float) -> None:
    """Where the price sits between its 52-week low and high.

    The number was already on the page as "62% of the range", which is a
    sentence a reader has to convert into a picture before it means anything.
    The picture is the thing they wanted, and it is one bar.
    """
    low = nq._to_float(conditions.get("low_52w"))
    high = nq._to_float(conditions.get("high_52w"))
    last = nq._to_float(conditions.get("last"))
    if not all(np.isfinite(v) for v in (low, high, last)) or high <= low:
        return

    figure = go.Figure()
    figure.add_shape(type="rect", x0=low, x1=high, y0=-0.18, y1=0.18,
                     fillcolor=GRID, line={"width": 0}, layer="below")
    # The marker, not a bar from the low: a bar reads as a quantity, and this
    # is a location.
    figure.add_trace(go.Scatter(
        x=[last], y=[0], mode="markers",
        marker={"size": 15, "color": ACCENT, "symbol": "diamond",
                "line": {"width": 2, "color": "white"}},
        hovertemplate=f"Now Rp {last:,.0f}<extra></extra>"))
    for value, text, anchor in ((low, f"52w low<br>{nq.format_rupiah(low, compact=False)}", "left"),
                                (high, f"52w high<br>{nq.format_rupiah(high, compact=False)}", "right")):
        figure.add_annotation(x=value, y=-0.55, text=text, showarrow=False,
                              xanchor=anchor, font={"size": 11, "color": MUTED})
    figure.add_annotation(x=last, y=0.62,
                          text=f"<b>{nq.format_rupiah(last, compact=False)}</b>"
                               f"<br>{position:.0f}% of range",
                          showarrow=False, font={"size": 12, "color": ACCENT})
    span = high - low
    figure.update_layout(
        height=130, showlegend=False,
        margin={"l": 8, "r": 8, "t": 26, "b": 26},
        plot_bgcolor="white", paper_bgcolor="white",
        xaxis={"range": [low - span * 0.12, high + span * 0.12],
               "visible": False},
        yaxis={"range": [-1.0, 1.1], "visible": False})
    st.plotly_chart(figure, width="stretch",
                    config={**CHART_CONFIG, "staticPlot": True})


def render_trading_conditions(prices: pd.DataFrame) -> None:
    """Three descriptive gauges. Deliberately not combined into one score.

    A single dial would be read the way a fear-and-greed dial is read, and on
    this panel that reading does not hold: bucketed by such a score the
    apparent pattern comes from one stock supplying a third of the extreme
    readings, with a rank correlation against the forward six-month return of
    +0.017. Three separate readings say what is happening without implying
    what happens next.
    """
    section("Market Conditions")
    conditions = nq.trading_conditions(prices)
    if not conditions.get("available"):
        st.info("A year of daily prices is needed to compare this stock "
                "against its own normal, and the snapshot does not have one "
                "for this company.")
        return

    position = nq._to_float(conditions.get("range_position"))
    volume = nq._to_float(conditions.get("volume_ratio"))
    turbulence = nq._to_float(conditions.get("volatility_ratio"))
    # The strip carries the range position, so the tile that used to state it
    # as a percentage is gone: the picture and the number said the same thing
    # a centimetre apart, and the low and high were printed a third time in
    # the note underneath.
    render_range_strip(conditions, position)
    st.caption(nq.range_band(position))
    columns = st.columns(2)
    tile(columns[0], "Volume vs its own normal",
         f"{volume:.2f}x" if np.isfinite(volume) else "Unavailable",
         nq.activity_band(volume), nq.TOOLTIPS["volume_ratio"])
    tile(columns[1], "Movement vs its own normal",
         f"{turbulence:.2f}x" if np.isfinite(turbulence) else "Unavailable",
         nq.turbulence_band(turbulence), nq.TOOLTIPS["volatility_ratio"])

    note("Both readings compare this stock against its own past year. Neither "
         "is good or bad on its own.")


def render_income_chart(company: dict) -> None:
    """Revenue against cost against net income, quarter by quarter."""
    section("Income Statement")
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
    section("Technical Indicators")
    if prices is None or prices.empty or "close" not in prices.columns:
        st.info("No price history to chart indicators from.")
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
    section("Fundamentals")
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
    note(f"As reported for the latest financial period. <em>In model</em> "
         f"marks which of the {modelled} ratios the estimate reads. A dash "
         f"means not reported, or not meaningful for this company \u2014 the "
         f"Meaning column says which.")


def render_return_forecast(predictions: dict, models: dict) -> None:
    """Both horizons side by side, shaped like the volatility forecast above.

    Two stacked full-width sections, each with its own four-metric row, gave
    the weaker of the two estimates more of the page than the stronger one.
    Same shape, same density, and the difference between them is left to the
    numbers rather than to the layout.
    """
    section("Return Forecast")
    note("Tested on past periods, these fundamentals sorted risers above "
         "fallers <strong>no better than chance</strong>. Kept on the page "
         "because a negative result is still a result. Read each figure as the "
         "historical frequency of a rise, not as a view on this company.")

    for column, horizon in zip(st.columns(2), nq.HORIZON_TRADING_DAYS):
        months = "6" if horizon == "6m" else "12"
        result, artifact = predictions.get(horizon, {}), models.get(horizon)
        label = f"{months}M Positive Return Probability"
        with column:
            if not result.get("available"):
                st.metric(label, "Not available",
                          help=result.get("reason", "Prediction unavailable."))
                continue
            probability = result["probability"]
            reliability = (artifact or {}).get("reliability", {})
            metrics = (artifact or {}).get("validation_metrics", {})
            has_edge = bool((artifact or {}).get(
                "has_edge", reliability.get("has_edge", True)))

            st.metric(label, f"{probability * 100:.0f}%",
                      help=nq.TOOLTIPS["probability"])
            st.caption(nq.probability_band(probability, has_edge))
            probability_bar(probability)
            auc = nq._to_float(metrics.get("roc_auc"))
            folds = (artifact or {}).get("validation_folds", "\u2014")
            st.caption(f"Tested {auc:.0%} accurate over {folds} periods \u00b7 "
                       f"{reliability.get('label', 'Unknown')}"
                       if np.isfinite(auc) else "Not validated")



def risk_band(probability: float) -> str:
    """Words for the forecast, pitched at what a weak edge can support."""
    if not np.isfinite(probability):
        return "Unavailable"
    if probability >= 0.65: return "Wider swings than most"
    if probability >= 0.55: return "Somewhat wider than most"
    if probability > 0.45:  return "About average"
    if probability > 0.35:  return "Somewhat calmer than most"
    return "Calmer than most"


def render_risk_analysis(prices: pd.DataFrame, window: str,
                         risks: dict, models: dict, peers: dict) -> None:
    """One section for risk, forecast first and history underneath.

    These used to be two things in two places: a descriptive "Historical risk"
    panel and, briefly, a proposal for a separate machine learning risk panel.
    Two risk sections would have been a reasonable thing for a reader to find
    confusing, and the second one would have been mostly a re-presentation of
    the first — the forecast's dominant input is trailing volatility, which the
    history panel already shows. So they are one section: what is expected,
    then what has happened.
    """
    section("Risk Analysis")
    if prices.empty:
        st.info("Risk cannot be measured without price history."); return

    st.markdown("**Volatility Forecast**")
    note("Probability this stock is more volatile than the median company over "
         "the horizon. <strong>Volatility is not direction</strong> — a stock "
         "can be turbulent while rising.")

    shown = False
    for column, horizon in zip(st.columns(2), ("risk_6m", "risk_12m")):
        artifact = models.get(horizon)
        result = risks.get(horizon, {})
        months = (artifact or {}).get("horizon_months",
                                      6 if horizon.endswith("6m") else 12)
        with column:
            if not result.get("available"):
                st.metric(f"{months}M High Volatility Probability", "Not available",
                          help=result.get("reason", "Forecast unavailable."))
                continue
            shown = True
            probability = result["probability"]
            klass = nq.volatility_class(probability, peers.get(horizon))
            st.metric(f"{months}M High Volatility Probability",
                      f"{probability * 100:.0f}%",
                      help=nq.TOOLTIPS["risk_probability"])
            st.caption(f"{klass} risk versus the {len(peers.get(horizon, []))} "
                       f"companies on file")
            probability_bar(probability)
            metrics = (artifact or {}).get("validation_metrics", {})
            auc = nq._to_float(metrics.get("roc_auc"))
            folds = (artifact or {}).get("validation_folds", "—")
            reliability = (artifact or {}).get("reliability", {})
            st.caption(f"Tested {auc:.0%} accurate over {folds} periods · "
                       f"{reliability.get('label', 'Unknown')}"
                       if np.isfinite(auc) else "Not validated")
            if not (artifact or {}).get("has_edge", False):
                st.warning("Did not beat chance out of sample.")
    if not shown:
        st.info("The volatility forecast is unavailable for this company.")

    st.markdown("**Historical Risk**")
    years = PRICE_WINDOWS[window]
    measures = nq.risk_metrics(prices, years)
    score = nq.risk_score(measures)

    columns = st.columns(5)
    # Not "Risk": the forecast band above is also a risk, and two things
    # under the same word on one page is a question the reader should not
    # have to answer. This one is a composite of what already happened.
    columns[0].metric("Composite Risk", score["band"],
                      help=nq.TOOLTIPS["risk_band"])
    columns[1].metric("Typical swing in a year",
                      nq.format_swing(measures["volatility"]),
                      help=nq.TOOLTIPS["swing_year"])
    columns[2].metric("Worst drop from a peak",
                      nq.format_percent(measures["max_drawdown"], 0),
                      help=nq.TOOLTIPS["max_drawdown"])
    columns[3].metric("Swing on down days",
                      nq.format_swing(measures["downside_volatility"]),
                      help=nq.TOOLTIPS["downside_volatility"])
    turnover = nq._to_float(measures.get("turnover"))
    columns[4].metric("Daily turnover",
                      nq.format_percent(turnover, 2) if np.isfinite(turnover)
                      else "Volume not reported",
                      help=nq.TOOLTIPS["turnover"])
    note(f"Measured over {years} year{'s' if years > 1 else ''}. "
         f"± figures are sizes, not gains or losses.")

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
        st.caption("" if controls["offline"]
                   else f"About {CREDITS_PER_COMPANY} API credits per analysis.")
        return

    try:
        with st.spinner(f"Loading {ticker}…"):
            company = load_company(ticker, controls["api_key"], controls["offline"])
    except nq.SectorsAPIError as error:
        show_error(error); return

    if company["quarterly"].empty:
        st.error(f"No quarterly financial data available for {ticker}."); return

    # The profile belongs at the top of the page but needs figures that only
    # exist once the price series has loaded. Claiming the space first and
    # filling it afterwards keeps the reading order and the data order apart.
    profile_slot = st.container()
    quarters_slot = st.container()

    prices, window, technical = render_chart(
        company, controls["api_key"], controls["offline"])
    features = with_price_features(company["features"], prices)
    risks = {h: predict(features, models.get(h), h) for h in nq.RISK_HORIZONS}
    risk = risks.get("risk_6m", {})
    peers = universe_volatility(models)
    predictions = {h: predict(features, models.get(h), h)
                   for h in nq.HORIZON_TRADING_DAYS}
    with profile_slot:
        render_profile(company, risks, peers)
    if company["n_quarters"] < nq.MIN_QUARTERS_FOR_PREDICTION:
        with quarters_slot:
            st.warning(f"Only {company['n_quarters']} quarterly reports available. "
                       f"NusaQuant wants at least {nq.MIN_QUARTERS_FOR_PREDICTION} "
                       f"before treating a fundamental prediction as meaningful.")

    render_momentum_charts(prices)
    render_technical(technical, window)
    render_trading_conditions(prices if company["prices"].empty
                              else company["prices"])
    render_income_chart(company)
    model_features = list((models.get("6m") or {}).get("feature_names", []))
    render_features(company, model_features)

    render_risk_analysis(prices, window, risks, models, peers)

    if not models:
        render_missing_models()
    else:
        render_return_forecast(predictions, models)

    disclaimer()


# ══════════════════════════════════════════════════════════════════════
# BEST 10
# ══════════════════════════════════════════════════════════════════════

def render_screening(companies: pd.DataFrame, models: dict, controls: dict) -> None:
    """Every company in the universe, both estimates, sorted by the reader.

    This view used to be "Top Picks", ordered by the return probability and cut
    to ten. Both halves of that were a claim the testing does not support: the
    return model has no measurable out-of-sample edge, so ranking by it put an
    ordering on the page that meant nothing, and calling the first ten "picks"
    invited exactly the reading it could not bear.

    So nothing is ranked for you. The table opens in market-cap order, which
    asserts nothing, and every column sorts on click. The figures are numeric
    rather than pre-formatted strings for that reason \u2014 a column of "53%"
    text sorts 9% above 53%, which is the kind of bug that stays invisible
    until someone acts on it.
    """
    section(MODE_PICKS)
    if not models:
        render_missing_models(); return

    horizon = "6m" if st.radio("Return horizon", ["6 Months", "12 Months"],
                               horizontal=True,
                               label_visibility="collapsed") == "6 Months" else "12m"
    artifact = models.get(horizon)
    risk_models = {h: models.get(h) for h in nq.RISK_HORIZONS if models.get(h)}
    if not artifact and not risk_models:
        st.info("No machine learning model is available."); return

    largest = max(5, len(companies))
    with st.container(border=True):
        size = st.slider("Universe size", 5, largest, min(len(companies), largest),
                         step=1,
                         help=UNIVERSE_SIZE_HELP.format(total=len(companies)))
        st.caption(UNIVERSE_SIZE_NOTE)
        if controls["offline"]:
            st.caption(f"Screening the companies recorded on {snapshot_as_of()}.")
        else:
            st.caption(f"Estimated cost: about "
                       f"{size * CREDITS_PER_COMPANY:,} API credits.")
        go = st.button("Screen the universe", type="primary")

    if not go:
        st.info("Press the button to screen the universe."); return

    universe = companies.head(size)
    rows = []
    progress = st.progress(0.0, text="Scoring\u2026")

    for index, company_row in enumerate(universe.itertuples(), start=1):
        progress.progress(index / len(universe), text=f"Scoring {company_row.symbol}")
        record = {"ticker": company_row.symbol, "company_name": company_row.company_name,
                  "probability": np.nan, "volatility": np.nan,
                  "trend": "Not measured", "data_quality": 0.0,
                  "eligible": False, "reason": ""}
        # NOT `for horizon in ...`: that name holds the return horizon the
        # reader picked, and rebinding it here left it stuck on the last key of
        # RISK_HORIZONS, so the radio silently did nothing and the return
        # column was always 12M.
        for risk_horizon in nq.RISK_HORIZONS:
            record[risk_horizon] = np.nan
        try:
            company = load_company(company_row.symbol, controls["api_key"],
                                   controls["offline"])
        except nq.SectorsAPIError as error:
            record["reason"] = error.message; rows.append(record); continue

        if company["n_quarters"] < nq.MIN_QUARTERS_FOR_PREDICTION:
            record["reason"] = (f"Only {company['n_quarters']} quarterly reports "
                                f"(minimum {nq.MIN_QUARTERS_FOR_PREDICTION})")
            rows.append(record); continue

        features = with_price_features(company["features"], company["prices"])
        result = predict(features, artifact, horizon)
        if result.get("available"):
            record.update({"probability": result["probability"], "eligible": True,
                           "data_quality": result["data_quality"]})
        else:
            record["reason"] = result.get("reason", "unavailable")
            record["data_quality"] = result.get("data_quality", 0.0)

        for name, model in risk_models.items():
            forecast = predict(features, model, name)
            if forecast.get("available"):
                record[name] = forecast["probability"]
                record["eligible"] = True
        if not company["prices"].empty:
            record["volatility"] = nq.risk_metrics(company["prices"], 1)["volatility"]
            record["trend"] = nq.technical_state(company["prices"]).get(
                "trend", "Not measured")
        rows.append(record)
    progress.empty()

    screened = pd.DataFrame(rows)
    qualified = screened[screened.eligible].copy()
    if qualified.empty:
        st.error("No company met NusaQuant's minimum data-quality criteria."); return

    months = "6" if horizon == "6m" else "12"
    return_column = f"{months}M Positive Return Probability"
    selected_risk = f"risk_{horizon}"

    # Ranked on the 6-month volatility forecast, ascending, and that choice is
    # the honest one rather than the flattering one. It is the only estimate on
    # this page that beat chance out of sample, so it is the only ordering that
    # rests on something tested. Ranking by the return probability would look
    # more like a stock tip and would be ordering the table by a model that was
    # measured and found not to work.
    rank_on = selected_risk if selected_risk in qualified else None
    if rank_on and qualified[rank_on].notna().any():
        qualified = qualified.sort_values(rank_on, ascending=True,
                                          na_position="last")
    qualified = qualified.reset_index(drop=True)
    qualified["Rank"] = np.arange(1, len(qualified) + 1)

    reference = universe_volatility(models)
    table = {"Rank": qualified["Rank"].to_numpy(),
             "Ticker": qualified.ticker.to_numpy(),
             "Company": qualified.company_name.to_numpy()}
    config = {
        "Rank": st.column_config.NumberColumn(width="small",
                                              help=nq.TOOLTIPS["rank"]),
        "Ticker": st.column_config.TextColumn(width="small",
                                              help=nq.TOOLTIPS["ticker"]),
        "Company": st.column_config.TextColumn(width="medium",
                                               help=nq.TOOLTIPS["company"])}
    # Only the horizon the reader picked. Showing both put two volatility
    # columns and two class columns beside a single return column, so the
    # control at the top of the page governed a third of the table.
    for name in (selected_risk,) if selected_risk in qualified else ():
        label = 6 if name.endswith("6m") else 12
        # Against the whole cached universe, not the rows on screen. Banding
        # within the selection makes "High" mean something different at a
        # universe size of 5 than at 31, and with a handful of rows the
        # terciles are decided by the count rather than by the companies.
        peers = reference.get(name) or qualified[name].dropna().tolist()
        probability = f"{label}M High Volatility Probability"
        klass = f"{label}M Risk Class"
        table[probability] = qualified[name].to_numpy() * 100
        table[klass] = [nq.volatility_class(v, peers) for v in qualified[name]]
        config[probability] = st.column_config.NumberColumn(
            format="%.0f%%", help=nq.TOOLTIPS["risk_probability"])
        config[klass] = st.column_config.TextColumn(
            help=nq.TOOLTIPS["risk_class"])
    table["Realised Volatility (1Y)"] = qualified.volatility.to_numpy() * 100
    table[return_column] = qualified.probability.to_numpy() * 100
    table["Trend"] = qualified.trend.to_numpy()
    table["Data Quality"] = qualified.data_quality.to_numpy() * 100
    config.update({
        "Realised Volatility (1Y)": st.column_config.NumberColumn(
            format="\u00b1%.1f%%", help=nq.TOOLTIPS["swing_year"]),
        return_column: st.column_config.NumberColumn(
            format="%.0f%%", help=nq.TOOLTIPS["probability"]),
        "Trend": st.column_config.TextColumn(help=nq.TOOLTIPS["trend"]),
        "Data Quality": st.column_config.NumberColumn(
            format="%.0f%%", help=nq.TOOLTIPS["data_quality"])})

    st.markdown(f"#### {len(qualified)} companies ranked")
    st.dataframe(pd.DataFrame(table), width="stretch", hide_index=True,
                 column_config=config)

    risk_edge = any((m or {}).get("has_edge") for m in risk_models.values())
    return_edge = bool((artifact or {}).get("has_edge", False))
    note(f"<strong>Ranked by the {months}M volatility forecast, calmest "
         f"first</strong> "
         + ("\u2014 the only estimate here that beat chance out of sample. "
            if risk_edge else "\u2014 no estimate here beat chance out of "
            "sample, so the order is for inspection only. ")
         + ("The return probability "
            + ("also passed its test. " if return_edge else
               "<strong>did not</strong>: it sorted risers above fallers no "
               "better than a coin toss, so it is shown but not ranked on. ")
            if artifact else "")
         + "Risk Class is the company's position among all companies on "
           "file, not an absolute scale. Click any heading to re-sort.")
    if not controls["offline"]:
        st.caption("Volatility columns need daily price history, which live "
                   "mode does not fetch here. Switch to the cached snapshot "
                   "to see them filled in.")

    excluded = screened[~screened.eligible]
    if not excluded.empty:
        with st.expander(f"Excluded by quality gates ({len(excluded)})"):
            st.dataframe(excluded[["ticker", "company_name", "reason"]],
                         width="stretch", hide_index=True)
    disclaimer()


# ══════════════════════════════════════════════════════════════════════
# PORTFOLIO ANALYSIS
# ══════════════════════════════════════════════════════════════════════
#: A year of daily bars, fetched in 90-day chunks because the endpoint clamps
#: anything wider without saying so. Five calls covers 365 days.
PORTFOLIO_PRICE_CREDITS = 5


def portfolio_prices(tickers: list[str], api_key: str,
                     offline: bool) -> dict[str, pd.DataFrame]:
    """A year of daily bars for each holding, from the snapshot or the API."""
    out: dict[str, pd.DataFrame] = {}
    for ticker in tickers:
        if offline:
            frame = snapshot_data(ticker, "prices")
        else:
            end = dt.date.today()
            frame = live_prices(ticker, (end - dt.timedelta(days=400)).isoformat(),
                                end.isoformat(), api_key)
        if frame is not None and not frame.empty:
            out[ticker] = frame
    return out


@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def _universe_volatility(horizons: tuple[str, ...]) -> dict[str, list[float]]:
    """Forecast volatility for every cached company, for the class bands.

    The bands are positions within a peer group, so they need a peer group. The
    cached snapshot is the natural one: it is the same universe the model was
    trained on, it costs nothing, and it does not shift when a reader changes
    which company they are looking at.
    """
    models, _ = load_models()
    out: dict[str, list[float]] = {h: [] for h in horizons}
    for ticker in snapshot_tickers():
        prices = snapshot_data(ticker, "prices")
        quarterly = snapshot_data(ticker, "quarterly")
        if prices.empty or quarterly.empty:
            continue
        latest = prices.sort_values("date").iloc[-1]
        frame = nq.features_frame(quarterly, nq._to_float(latest.get("market_cap")),
                                  nq._to_float(latest.get("close")))
        frame = with_price_features(frame, prices)
        for horizon in horizons:
            result = predict(frame, models.get(horizon), horizon)
            if result.get("available"):
                out[horizon].append(result["probability"])
    return out


def universe_volatility(models: dict) -> dict[str, list[float]]:
    available = tuple(h for h in nq.RISK_HORIZONS if models.get(h))
    return _universe_volatility(available) if available else {}


def render_holdings_editor(companies: pd.DataFrame, controls: dict) -> dict[str, float]:
    """Enter positions in lots, and take them out again.

    Lots rather than shares because that is the unit an IDX order is placed
    in — one lot is a hundred shares, and a reader who owns "twenty lots" of
    BBRI should not have to multiply before they can use this page.
    """
    holdings: dict[str, float] = st.session_state.setdefault("portfolio", {})

    with st.container(border=True):
        left, middle, right = st.columns([3, 1, 1])
        if controls["offline"]:
            labels = {f"{r.symbol} — {r.company_name}": r.symbol
                      for r in companies.itertuples()}
            choice = left.selectbox("Stock", sorted(labels),
                                    label_visibility="collapsed")
            ticker = labels.get(choice, "")
        else:
            ticker = left.text_input(
                "Stock", placeholder="Ticker, for example BBRI",
                label_visibility="collapsed").strip().upper()
        lots = middle.number_input("Lots", min_value=1, value=10, step=1,
                                   label_visibility="collapsed")
        if right.button("Add", width="stretch"):
            if not ticker:
                st.warning("Enter a ticker before adding it.")
            else:
                holdings[ticker] = float(lots)
                st.session_state["portfolio"] = holdings
                st.rerun()
        st.caption(
            f"Enter lots. 1 lot = {nq.LOT_SIZE} shares, so 20 lots = "
            f"{20 * nq.LOT_SIZE:,} shares. "
            + ("In this mode you can pick from the companies already stored. "
               "Switch to Live Sectors API to enter any listed company."
               if controls["offline"] else
               "Any ticker Sectors covers. Adding one the API does not know "
               "will be reported when you press Analyse."))

    if not holdings:
        return holdings

    st.markdown("**Your holdings**")
    for ticker in list(holdings):
        row = st.columns([3, 1, 1])
        row[0].markdown(f"`{ticker}` {nq.company_name(ticker) or ''}")
        row[1].markdown(f"{holdings[ticker]:,.0f} lots")
        if row[2].button("Remove", key=f"drop_{ticker}", width="stretch"):
            holdings.pop(ticker, None)
            st.session_state["portfolio"] = holdings
            st.rerun()
    return holdings


def render_portfolio(companies: pd.DataFrame, models: dict, controls: dict) -> None:
    """What a set of holdings does together, which no single-stock page can say.

    Sector Ranking used to live here and was removed: it listed the same ratios
    the stock page already shows, one sector at a time, so a reader who had
    seen one view had seen the other. This view earns its place on the
    covariance. Diversification is not a property any individual holding has —
    it only exists between them.
    """
    section(MODE_PORTFOLIO)
    note("Enter your holdings in lots. NusaQuant measures the portfolio as "
         "one position: volatility, drawdown, projected range and "
         "correlation.")

    holdings = render_holdings_editor(companies, controls)
    if not holdings:
        st.info("Add at least two holdings to analyse."); return

    per_holding = (0 if controls["offline"]
                   else CREDITS_PER_COMPANY + PORTFOLIO_PRICE_CREDITS)
    left, right = st.columns([1, 3])
    pressed = left.button("Analyse portfolio", type="primary", width="stretch")
    right.caption(
        f"{len(holdings)} holding{'s' if len(holdings) != 1 else ''} · "
        + ("free in this mode, nothing is fetched." if controls["offline"] else
           f"about {len(holdings) * per_holding:,} API credits "
           f"({per_holding} per holding)."))

    if pressed:
        st.session_state["portfolio_analysed"] = dict(holdings)
    analysed = st.session_state.get("portfolio_analysed")
    if not analysed:
        st.info("Press **Analyse portfolio** when your holdings are complete.")
        return
    if analysed != holdings:
        # Silently redrawing stale figures under an edited portfolio is the
        # failure mode worth guarding: the numbers look current and are not.
        st.warning("Your holdings have changed since this was last analysed. "
                   "The figures below describe the earlier portfolio — press "
                   "**Analyse portfolio** again to bring them up to date.")

    with st.spinner("Measuring the portfolio…"):
        try:
            prices = portfolio_prices(list(analysed), controls["api_key"],
                                      controls["offline"])
        except nq.SectorsAPIError as error:
            show_error(error); return
        analysis = nq.portfolio_analysis(prices, analysed)

    if not analysis.get("available"):
        if analysis.get("reason"):
            st.error(f"These holdings share only {analysis.get('overlap', 0)} "
                     f"trading days of overlapping price history, which is too "
                     f"little to measure them together. Remove the most "
                     f"recently listed holding and try again.")
        else:
            st.error("No price history was found for these holdings.")
        return
    if analysis["dropped"]:
        st.warning(f"No price history for {', '.join(analysis['dropped'])} — "
                   f"left out of everything below.")

    render_portfolio_summary(analysis)
    render_portfolio_projection(analysis, prices)
    render_portfolio_risk(analysis, models, controls)
    render_portfolio_mix(analysis)
    disclaimer()


def render_portfolio_summary(analysis: dict) -> None:
    columns = st.columns(3)
    columns[0].metric("Total value", nq.format_rupiah(analysis["total"]),
                      help=nq.TOOLTIPS["portfolio_value"])
    columns[0].caption(f"{sum(analysis['lots'].values()):,.0f} lots across "
                       f"{len(analysis['tickers'])} stocks")
    columns[1].metric("Typical swing in a year",
                      nq.format_swing(analysis["volatility"]),
                      help=nq.TOOLTIPS["portfolio_swing"])
    columns[1].caption("the whole portfolio, not the average holding")
    columns[2].metric("Worst drop from a peak",
                      nq.format_percent(analysis["max_drawdown"], 1),
                      help=nq.TOOLTIPS["portfolio_drawdown"])
    columns[2].caption("if you had never changed the mix")

    saved = analysis["diversification_benefit"]
    with st.container(border=True):
        st.markdown("**Diversification**")
        note(f"Weighted average of the parts is "
             f"{nq.format_swing(analysis['undiversified_volatility'])}; the "
             f"portfolio is {nq.format_swing(analysis['volatility'])}. The "
             f"{abs(saved) * 100:.1f}-point gap is the diversification "
             f"benefit.")
        for label, value, colour in (
                ("Each one alone", analysis["undiversified_volatility"], NEGATIVE),
                ("Held together", analysis["volatility"], POSITIVE)):
            widest = max(analysis["undiversified_volatility"],
                         analysis["volatility"], 1e-9)
            bar, figure = st.columns([5, 1])
            bar.markdown(
                f"<div style='font-size:.75rem;color:{MUTED};margin-bottom:2px'>"
                f"{label}</div>"
                f"<div style='background:{GRID};border-radius:4px;height:16px'>"
                f"<div style='width:{100 * value / widest:.1f}%;background:{colour};"
                f"height:16px;border-radius:4px'></div></div>",
                unsafe_allow_html=True)
            figure.markdown(f"<div style='padding-top:14px'>"
                            f"{nq.format_swing(value)}</div>",
                            unsafe_allow_html=True)
    st.caption(f"Measured over {analysis['observations']:,} trading days that "
               f"every holding traded, {analysis['from']:%b %Y} to "
               f"{analysis['to']:%b %Y}.")


def render_portfolio_projection(analysis: dict, prices: dict) -> None:
    section("Projected Range")
    months = st.radio("Horizon", ["6 months", "12 months"], horizontal=True,
                      label_visibility="collapsed")
    horizon = 126 if months == "6 months" else 252
    projection = nq.portfolio_projection(analysis, prices, horizon)
    if not projection.get("available"):
        st.info("A year of price history is needed before a range can be drawn.")
        return

    note("A calibrated volatility cone: trailing volatility scaled to the "
         "horizon. It bounds how far the value may travel, not which way.")

    rows = []
    for position in projection["positions"]:
        ticker = position["ticker"]
        if not position.get("available"):
            rows.append({"Stock": ticker, "Change": "—",
                         "Projected Value": "not enough history"})
            continue
        rows.append({
            "Stock": ticker,
            "Change": f"{position['low_change']:+.0%} to {position['high_change']:+.0%}",
            "Projected Value": f"{nq.format_rupiah(position['low'])} – "
                               f"{nq.format_rupiah(position['high'])}"})
    rows.append({"Stock": "Your portfolio",
                 "Change": f"{projection['low_change']:+.0%} to "
                           f"{projection['high_change']:+.0%}",
                 "Projected Value": f"{nq.format_rupiah(projection['low'])} – "
                                    f"{nq.format_rupiah(projection['high'])}"})
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True,
                 column_config={
                     "Stock": st.column_config.TextColumn(width="small",
                                                          help=nq.TOOLTIPS["ticker"]),
                     "Change": st.column_config.TextColumn(
                         help=nq.TOOLTIPS["position_range"]),
                     "Projected Value": st.column_config.TextColumn(
                         width="large", help=nq.TOOLTIPS["position_range"])})




def render_portfolio_risk(analysis: dict, models: dict, controls: dict) -> None:
    section("Volatility Forecast")
    horizons = {h: models.get(h) for h in nq.RISK_HORIZONS if models.get(h)}
    rows = []
    for ticker in analysis["tickers"]:
        record = {"Ticker": ticker,
                  "Realised Volatility (1Y)": analysis["each_volatility"][ticker] * 100}
        for name in horizons:
            record[name] = np.nan
        if horizons:
            try:
                company = load_company(ticker, controls["api_key"], controls["offline"])
                features = with_price_features(company["features"], company["prices"])
                for name, artifact in horizons.items():
                    result = predict(features, artifact, name)
                    if result.get("available"):
                        record[name] = result["probability"] * 100
            except nq.SectorsAPIError:
                pass
        rows.append(record)

    frame = pd.DataFrame(rows)
    reference = universe_volatility(models)
    table = {"Ticker": frame["Ticker"].to_numpy()}
    config = {"Ticker": st.column_config.TextColumn(width="small",
                                                    help=nq.TOOLTIPS["ticker"])}
    for name in horizons:
        months = 6 if name.endswith("6m") else 12
        probability = f"{months}M High Volatility Probability"
        klass = f"{months}M Risk Class"
        peers = reference.get(name) or (frame[name].dropna() / 100).tolist()
        table[probability] = frame[name].to_numpy()
        table[klass] = [nq.volatility_class(v / 100 if v == v else v, peers)
                        for v in frame[name]]
        config[probability] = st.column_config.NumberColumn(
            format="%.0f%%", help=nq.TOOLTIPS["risk_probability"])
        config[klass] = st.column_config.TextColumn(help=nq.TOOLTIPS["risk_class"])
    table["Realised Volatility (1Y)"] = frame["Realised Volatility (1Y)"].to_numpy()
    config["Realised Volatility (1Y)"] = st.column_config.NumberColumn(
        format="\u00b1%.1f%%", help=nq.TOOLTIPS["swing_year"])

    ordered = pd.DataFrame(table)
    lead = next((c for c in ordered.columns if "High Volatility" in c), None)
    if lead:
        ordered = ordered.sort_values(lead, ascending=False, na_position="last")
    note("Probability each holding is more volatile than the median company. "
         "Risk Class is its position among all companies on file. "
         "<strong>Volatility is not direction.</strong>")
    st.dataframe(ordered, width="stretch", hide_index=True, column_config=config)


def render_portfolio_mix(analysis: dict) -> None:
    section("Sector Proportion")
    sectors: dict[str, float] = {}
    for ticker, weight in analysis["weight"].items():
        name = nq.company_classification(ticker).get("sector") or "Not classified"
        sectors[name] = sectors.get(name, 0.0) + weight
    ordered = sorted(sectors.items(), key=lambda kv: -kv[1])

    # A horizontal bar per sector rather than one stacked bar. Stacking put
    # every label inside its own slice, and a slice worth 3% is narrower than
    # the text it holds, so the percentages were clipped by whatever sat next
    # to them. One row each gives every label the full width to sit in.
    figure = go.Figure(go.Bar(
        x=[weight * 100 for _, weight in ordered],
        y=[name for name, _ in ordered], orientation="h",
        marker={"color": ACCENT},
        text=[f"{weight:.1%}" for _, weight in ordered],
        textposition="outside", cliponaxis=False,
        hovertemplate="%{y}: %{x:.1f}%<extra></extra>"))
    figure.update_layout(
        height=max(150, 40 * len(ordered)), showlegend=False,
        margin={"l": 0, "r": 60, "t": 8, "b": 8},
        plot_bgcolor="white", paper_bgcolor="white",
        xaxis={"range": [0, max(w for _, w in ordered) * 118],
               "ticksuffix": "%", "gridcolor": GRID},
        yaxis={"autorange": "reversed", "ticksuffix": "  "})
    st.plotly_chart(figure, width="stretch", config=CHART_CONFIG)

    top_two = sum(w for _, w in ordered[:2])
    largest_ticker = max(analysis["weight"], key=analysis["weight"].get)
    note(f"Top two sectors: {top_two:.0%}. Largest position: "
         f"{largest_ticker} at {analysis['weight'][largest_ticker]:.0%}.")

    with st.expander("Correlation Matrix"):
        correlation = analysis["correlation"]
        note("1.00 is perfect lockstep, 0 is unrelated. High pairs "
             "diversify you less than their count suggests.")
        pairs = [(a, b, float(correlation.loc[a, b]))
                 for i, a in enumerate(correlation.columns)
                 for b in correlation.columns[i + 1:]]
        if pairs:
            closest = max(pairs, key=lambda p: p[2])
            loosest = min(pairs, key=lambda p: p[2])
            st.caption(f"Closest pair: {closest[0]} and {closest[1]} at "
                       f"{closest[2]:.2f}. Loosest: {loosest[0]} and "
                       f"{loosest[1]} at {loosest[2]:.2f}, the pair doing the "
                       f"most work for you.")
        # Red at 0, green at 1, through the same hue rotation the trend badge
        # uses. Note what the colour means here: green marks pairs that move
        # TOGETHER, which is the direction that diversifies you least. The
        # scale reads as magnitude, not as good and bad, so the caption says
        # which is which rather than leaving red and green to imply it.
        def shade(value: float) -> str:
            number = nq._to_float(value)
            if not np.isfinite(number):
                return ""
            return (f"background-color:{trend_colour(np.clip(number, 0.0, 1.0))};"
                    f"color:#FFFFFF")

        st.caption("Green: the pair moves together, which diversifies you "
                   "least. Red: they move independently.")
        st.dataframe(correlation.style.map(shade).format("{:.2f}"),
                     width="stretch")


def main() -> None:
    configure_page()
    st.markdown('<div class="nq-title">NusaQuant</div>', unsafe_allow_html=True)
    # The subtitle used to promise probability estimates of positive returns,
    # which is the one thing the testing says this cannot do. Leading with it
    # made the honest labels further down read as a retraction.
    st.markdown('<div class="nq-sub">IDX Machine Learning Market Intelligence · '
                'Fundamentals, price history and a tested forecast of how much '
                'a share is likely to move.</div>',
                unsafe_allow_html=True)

    models, metadata = load_models()
    controls = render_sidebar(metadata)

    if controls["offline"]:
        names = nq.company_names()
        companies = by_market_cap(pd.DataFrame({
            "symbol": controls["snapshot"],
            "company_name": [names.get(t, t) for t in controls["snapshot"]]}))
        st.info(f"Everything on this page is real market data recorded on "
                f"{snapshot_as_of()}. It is not today's market — prices and "
                f"figures will have moved since.")
    else:
        if not controls["api_key"]:
            st.info(nq.WELCOME)
            disclaimer()
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
    elif controls["mode"] == MODE_PORTFOLIO:
        render_portfolio(companies, models, controls)
    else:
        render_screening(companies, models, controls)
    footer()


if __name__ == "__main__":
    main()
