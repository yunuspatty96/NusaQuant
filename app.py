"""
NusaQuant — IDX Market Intelligence
===================================

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

import datetime as dt
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
PRICE_WINDOWS = {"1Y": 1, "3Y": 3, "5Y": 5}

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
        return {"available": False, "reason": f"No trained model for the {horizon} horizon."}
    model_features = list(artifact.get("feature_names", []))
    missing_columns = [c for c in model_features if c not in features.columns]
    if missing_columns:
        return {"available": False,
                "reason": "Some model inputs are unavailable for this company.",
                "missing": missing_columns}

    frame = features[model_features]
    completeness = nq.data_quality(frame, model_features)
    if completeness < nq.MIN_DATA_COMPLETENESS:
        absent = [c for c in model_features if pd.isna(frame.iloc[0][c])]
        return {"available": False,
                "reason": "Too many model inputs are missing for a reliable prediction.",
                "missing": absent, "data_quality": completeness}
    try:
        probability = float(artifact["pipeline"].predict_proba(frame)[0, 1])
    except Exception:
        return {"available": False, "reason": "The model could not score this company."}
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
        overview = {"market_cap": market_cap, "last_close_price": close}
        # The screener result is already cached, so the real name costs nothing.
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
        "features": nq.features_frame(quarterly, market_cap, close),
        "latest_period": quarterly["report_date"].max() if not quarterly.empty else pd.NaT,
        "n_quarters": len(quarterly),
    }


# ══════════════════════════════════════════════════════════════════════
# UI HELPERS
# ══════════════════════════════════════════════════════════════════════

def configure_page() -> None:
    st.set_page_config(page_title="NusaQuant — IDX Market Intelligence",
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
      [data-testid="stMetricLabel"] p {{ font-size:.8rem; line-height:1.3; }}
      [data-testid="stMetric"] {{ overflow:visible; }}
      .nq-title {{ font-size:1.6rem; font-weight:650; letter-spacing:-.01em; margin-bottom:.1rem; }}
      .nq-sub {{ color:{MUTED}; font-size:.9rem; }}
      .nq-sec {{ font-size:1.05rem; font-weight:600; margin:1.4rem 0 .5rem;
                 padding-bottom:.3rem; border-bottom:1px solid {GRID}; }}
      .nq-note {{ color:{MUTED}; font-size:.84rem; line-height:1.55; }}
      .nq-name {{ font-size:1.3rem; font-weight:650; letter-spacing:-.01em;
                  line-height:1.3; margin:.2rem 0 .1rem; overflow-wrap:anywhere; }}
      .nq-name span {{ color:{MUTED}; font-weight:450; }}
      .nq-chip {{ display:inline-block; padding:.12rem .5rem; border-radius:10px;
                  font-size:.76rem; font-weight:600; border:1px solid {GRID};
                  color:{MUTED}; margin-right:.35rem; }}
      div[data-testid="stDataFrame"] {{ font-variant-numeric:tabular-nums;
                                        font-size:.86rem; }}
      /* st.dataframe draws onto a canvas, so its cells clip long text and no
         stylesheet can reach inside to wrap them. Reference tables that are
         read rather than sorted are plain HTML for exactly that reason. */
      .nq-table {{ width:100%; border-collapse:collapse; font-size:.86rem;
                   table-layout:fixed; margin:.2rem 0 .4rem; }}
      .nq-table th {{ text-align:left; font-weight:600; color:{MUTED};
                      border-bottom:1px solid {GRID}; padding:.45rem .6rem;
                      font-size:.8rem; }}
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
        st.error("**Model files exist but could not be loaded.**\n\n"
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
        mode = st.radio("Analysis", ["Single Stock", "Best 10"],
                        label_visibility="collapsed")
        window = st.radio("Price history", list(PRICE_WINDOWS), horizontal=True)

        st.divider()
        if metadata:
            st.caption(f"XGBoost v{metadata.get('version','—')} · "
                       f"{len(metadata.get('feature_set', []))} features · "
                       f"trained to {metadata.get('training_end_date','—')}")
        else:
            st.caption("No trained models loaded.")

    return {"offline": offline, "api_key": api_key, "snapshot": snapshot,
            "mode": mode, "window": window}


# ══════════════════════════════════════════════════════════════════════
# SINGLE STOCK
# ══════════════════════════════════════════════════════════════════════

def render_profile(company: dict, predictions: dict, technical: dict) -> None:
    ticker, name = company["ticker"], company["name"]
    heading = ticker if name == ticker else f"{ticker} <span>— {name}</span>"
    st.markdown(f"<div class='nq-name'>{heading}</div>", unsafe_allow_html=True)

    overview = company["overview"]
    chips = [c for c in (overview.get("sector"), overview.get("sub_sector"),
                         technical.get("trend") if technical.get("available") else None)
             if c]
    if chips:
        st.markdown("".join(f"<span class='nq-chip'>{c}</span>" for c in chips),
                    unsafe_allow_html=True)

    columns = st.columns(4)
    columns[0].metric("Latest close",
                      nq.format_rupiah(overview.get("last_close_price"), compact=False))
    columns[1].metric("Market cap", nq.format_rupiah(overview.get("market_cap")))
    for column, horizon in ((columns[2], "6m"), (columns[3], "12m")):
        result = predictions.get(horizon, {})
        label = f"{'6' if horizon == '6m' else '12'}M probability"
        if result.get("available"):
            column.metric(label, f"{result['probability'] * 100:.0f}%")
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


def render_chart(company: dict, api_key: str, window: str, offline: bool) -> pd.DataFrame:
    section(f"Price history — {window}")
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
            show_error(error); return pd.DataFrame()

    if prices.empty:
        st.info("No price history available for this window.")
        return prices

    history = prices.sort_values("date").reset_index(drop=True)
    close = history["close"].astype(float)
    first, last = float(close.iloc[0]), float(close.iloc[-1])
    gained = last >= first
    line_colour = POSITIVE if gained else NEGATIVE
    fill = "rgba(31,122,90,.10)" if gained else "rgba(179,52,31,.10)"

    has_volume = "volume" in history.columns and history["volume"].notna().any()
    figure = make_subplots(
        rows=2 if has_volume else 1, cols=1, shared_xaxes=True,
        row_heights=[0.76, 0.24] if has_volume else [1.0], vertical_spacing=0.04)

    figure.add_trace(go.Scatter(
        x=history["date"], y=close, mode="lines", name="Close",
        line={"color": line_colour, "width": 1.8},
        fill="tozeroy", fillcolor=fill,
        hovertemplate="Rp %{y:,.0f}<extra>Close</extra>"), row=1, col=1)

    # Moving averages are drawn only where they are fully formed. A 200-day
    # average seeded from 30 days of data is not a 200-day average.
    averages = nq.moving_averages(history)
    for window, colour, dash in ((50, ACCENT, "solid"), (200, MUTED, "dot")):
        column = f"ma{window}"
        if column in averages and averages[column].notna().any():
            figure.add_trace(go.Scatter(
                x=averages["date"], y=averages[column], mode="lines",
                name=f"MA{window}",
                line={"color": colour, "width": 1.1, "dash": dash},
                hovertemplate="Rp %{y:,.0f}<extra>MA" + str(window) + "</extra>"),
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
    st.plotly_chart(figure, width="stretch", config={"displayModeBar": False})

    change = (last / first - 1.0) if first > 0 else np.nan
    note(f"Over this window the close moved from "
         f"{nq.format_rupiah(first, compact=False)} to "
         f"{nq.format_rupiah(last, compact=False)}, a change of "
         f"{nq.format_percent(change, 1)}. MA50 and MA200 are drawn only where "
         f"there is enough history to form them.")
    return prices


def render_technical(technical: dict, window: str) -> None:
    """Descriptive trend state. Deliberately kept apart from the model."""
    section("Technical state")
    if not technical.get("available"):
        st.info("Not enough price history in this window to describe a trend.")
        return

    columns = st.columns(5)
    columns[0].metric("Trend", technical["trend"],
                      help="Price against its own 50- and 200-day averages. "
                           "Above both is an uptrend and below both a "
                           "downtrend; above the 50 but under the 200 is "
                           "recovering, and the reverse is weakening.")
    rsi = nq._to_float(technical.get("rsi14"))
    columns[1].metric("RSI (14)",
                      f"{rsi:.0f}" if np.isfinite(rsi) else "Insufficient history",
                      nq.rsi_band(rsi) if np.isfinite(rsi) else None,
                      help="Relative Strength Index over 14 days. Above 70 is "
                           "conventionally read as overbought and below 30 as "
                           "oversold; between them is neutral.")
    columns[2].metric("From 52-week high",
                      nq.format_percent(technical.get("from_52w_high"), 0))
    columns[3].metric("6-month return", nq.format_percent(technical.get("return_6m"), 0))
    columns[4].metric("12-month return", nq.format_percent(technical.get("return_12m"), 0))

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
            marker={"color": MUTED},
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
    st.plotly_chart(figure, width="stretch", config={"displayModeBar": False})

    detail = ("Cost of revenue" if cost_label == "Cost of revenue"
              else "Operating expense")
    note(f"Per quarter, not cumulative: filings that report year-to-date are "
         f"de-cumulated first, so a fourth quarter is one quarter and not the "
         f"whole year. Cost is shown as <strong>{escape(detail.lower())}</strong> "
         f"for this company — issuers that do not file a cost of revenue, banks "
         f"among them, are charted on operating expense instead, and the label "
         f"says which.")


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
            if spec.unavailable:
                # Neither modelled nor available: saying "Reference" would
                # imply the number is there to look at, and it is not.
                in_model = "—"
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
        "<table class='nq-table'>"
        "<colgroup><col style='width:20%'><col style='width:13%'>"
        "<col style='width:11%'><col style='width:13%'><col style='width:43%'>"
        "</colgroup>"
        "<thead><tr><th>Metric</th><th>Value</th><th>Unit</th>"
        "<th>In model</th><th>Meaning</th></tr></thead>"
        f"<tbody>{''.join(body)}</tbody></table>",
        unsafe_allow_html=True)

    modelled = sum(1 for f in nq.FEATURE_SCHEMA if f.modelled)
    note(f"Values are as reported for the latest available financial period. A "
         f"high or low reading is not automatically good or bad — the model "
         f"weighs these together rather than applying a rule to any single one."
         f"<br><br><strong>In model</strong> has three states. "
         f"<em>Yes</em> and <em>No (dropped)</em> apply to the {modelled} "
         f"scale-free ratios the model is allowed to read; dropped means the "
         f"ratio was missing for more than {nq.MAX_FEATURE_MISSINGNESS:.0%} of "
         f"the training panel, so training left it out rather than impute its "
         f"way around the gap. <em>Reference</em> is everything measured in "
         f"rupiah: shown because a reader wants it, never modelled, because a "
         f"level would let the model split on company size rather than on value."
         f"<br><br>A dash is a metric that does not apply or was not filed — "
         f"NPL and LDR outside a bank, a dividend history the endpoint does not "
         f"carry — and the Meaning column says which.")


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
        st.metric("Probability of positive return", f"{probability * 100:.0f}%")
        st.caption(nq.probability_band(probability, has_edge))
    with right:
        probability_bar(probability)
        note(nq.explain_probability(probability, horizon, has_edge))

    columns = st.columns(4)
    columns[0].metric("Model reliability", reliability.get("label", "Unknown"))
    columns[1].metric("Out-of-sample ROC-AUC",
                      f"{metrics.get('roc_auc', float('nan')):.3f}"
                      if np.isfinite(nq._to_float(metrics.get("roc_auc"))) else "—")
    columns[2].metric("Validation folds", (artifact or {}).get("validation_folds", "—"))
    columns[3].metric("Data quality", f"{result.get('data_quality', 0) * 100:.0f}%")

    if not has_edge:
        # The panel size is read from the artifact rather than written into the
        # sentence: it was hard-coded at 15 and silently went stale the first
        # time the universe grew.
        tickers = (artifact or {}).get("n_training_tickers")
        panel = f" — {tickers} tickers" if tickers else ""
        st.warning(
            f"**This model has no measurable edge.** Across purged walk-forward "
            f"folds it did not rank winners above losers by more than chance, so "
            f"its probabilities are deliberately shrunk toward the historical "
            f"base rate. Read the number as *how often stocks in this universe "
            f"rose over this horizon*, not as a view on this company. The limit "
            f"is the size of the training panel{panel} — not the algorithm.")
    elif reliability.get("label") == "Weak":
        st.warning("This model provides limited predictive separation on "
                   "out-of-sample data. Treat the probability as weak evidence, "
                   "not a signal.")

    with st.expander("How was this validated?"):
        st.write(nq.explain_reliability(reliability.get("label", "Unknown"), horizon))
        st.write(nq.EXPLANATIONS["probability"])
        baseline = (artifact or {}).get("baseline_roc_auc")
        if baseline:
            st.write(f"**Versus a baseline.** A model that always predicts the "
                     f"class prior scored {baseline:.3f} on the same folds. "
                     f"ML is only worth using if it clearly beats that.")
        weight = nq._to_float((artifact or {}).get("shrinkage_weight"))
        if np.isfinite(weight):
            unshrunk = (artifact or {}).get("validation_metrics_unshrunk", {})
            st.write(
                f"**Shrinkage {weight:.2f}.** The served probability is "
                f"`{weight:.2f} x model + {1 - weight:.2f} x base rate`, with the "
                f"weight fitted leave-one-fold-out on out-of-sample log loss. "
                f"Blending is monotone, so the ranking is untouched; only the "
                f"spread of the numbers changes. Before shrinkage the same "
                f"model scored ROC-AUC "
                f"{nq._to_float(unshrunk.get('roc_auc')):.3f}.")
        leaderboard = (artifact or {}).get("leaderboard", [])
        if leaderboard:
            st.write("**Candidates considered** (ranked on out-of-sample log loss):")
            st.dataframe(pd.DataFrame(leaderboard).round(4),
                         width="stretch", hide_index=True)
        rows = [{k: v for k, v in metrics.items() if k in
                 ("roc_auc", "pr_auc", "brier", "log_loss", "balanced_accuracy",
                  "precision", "recall", "base_rate", "roc_auc_std", "n")}]
        st.dataframe(pd.DataFrame(rows).round(4), width="stretch", hide_index=True)
        folds = (artifact or {}).get("fold_metrics", [])
        if folds:
            st.write("**Per validation fold** (purged walk-forward, one fold per "
                     "quarterly rebalance):")
            frame = pd.DataFrame(folds)
            keep = [c for c in ("validation_year", "n_train", "n_validation",
                                "roc_auc", "brier", "base_rate") if c in frame.columns]
            st.dataframe(frame[keep].round(4), width="stretch", hide_index=True)
        importance = (artifact or {}).get("feature_importance", [])
        if importance:
            st.write("**Feature importance** (a diagnostic — importance is not causality):")
            st.dataframe(pd.DataFrame(importance).round(4), width="stretch", hide_index=True)


def render_risk(prices: pd.DataFrame, window: str) -> None:
    section("Historical risk")
    if prices.empty:
        st.info("Risk cannot be measured without price history."); return
    years = PRICE_WINDOWS[window]
    metrics = nq.risk_metrics(prices, years)
    risk = nq.risk_score(metrics)

    columns = st.columns(5)
    columns[0].metric("Risk", risk["band"])
    columns[1].metric("Annualised volatility", nq.format_percent(metrics["volatility"], 0))
    columns[2].metric("Maximum drawdown", nq.format_percent(metrics["max_drawdown"], 0))
    columns[3].metric("Downside volatility", nq.format_percent(metrics["downside_volatility"], 0))
    # Liquidity carries 10% of the risk score and was computed all along, but
    # the panel showed four of its five inputs and left this one invisible.
    turnover = nq._to_float(metrics.get("turnover"))
    columns[4].metric("Daily turnover",
                      nq.format_percent(turnover, 2) if np.isfinite(turnover)
                      else "Volume not reported",
                      help="Median daily traded value as a share of market cap. "
                           "Thin trading is itself a risk.")
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
        st.plotly_chart(figure, width="stretch", config={"displayModeBar": False})


def render_single_stock(companies: pd.DataFrame, models: dict, controls: dict) -> None:
    section("Stock analysis")
    if companies.empty:
        st.error("No companies available."); return

    labels = {f"{r.symbol} — {r.company_name}": r.symbol for r in companies.itertuples()}
    keys = list(labels)
    chosen = st.session_state.get("ticker")
    index = next((i for i, k in enumerate(keys) if labels[k] == chosen), 0)

    left, right = st.columns([3, 1])
    selection = left.selectbox("Select stock", keys, index=index)
    if right.button("Analyze", width="stretch", type="primary"):
        st.session_state["ticker"] = labels[selection]
        st.session_state["analysed"] = labels[selection]
    ticker = labels[selection]

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
    technical = nq.technical_state(company["prices"])
    render_profile(company, predictions, technical)

    if company["n_quarters"] < nq.MIN_QUARTERS_FOR_PREDICTION:
        st.warning(f"Only {company['n_quarters']} quarterly reports available. "
                   f"NusaQuant wants at least {nq.MIN_QUARTERS_FOR_PREDICTION} "
                   f"before treating a fundamental prediction as meaningful.")

    prices = render_chart(company, controls["api_key"], controls["window"], controls["offline"])
    # In live mode the profile has no cached series, so the trend is described
    # from whatever the chart just fetched rather than not at all.
    if not technical.get("available"):
        technical = nq.technical_state(prices)
    render_income_chart(company)
    render_technical(technical, controls["window"])
    model_features = list((models.get("6m") or {}).get("feature_names", []))
    render_features(company, model_features)

    if not models:
        render_missing_models()
    else:
        render_prediction(predictions["6m"], models.get("6m"), "6m")
        render_prediction(predictions["12m"], models.get("12m"), "12m")

    render_risk(prices, controls["window"])
    st.caption(nq.DISCLAIMER)


# ══════════════════════════════════════════════════════════════════════
# BEST 10
# ══════════════════════════════════════════════════════════════════════

def render_best_10(companies: pd.DataFrame, models: dict, controls: dict) -> None:
    section("Best 10 stocks")
    if not models:
        render_missing_models(); return

    horizon = "6m" if st.radio("Horizon", ["6 Months", "12 Months"], horizontal=True,
                               label_visibility="collapsed") == "6 Months" else "12m"
    artifact = models.get(horizon)
    if not artifact:
        st.info("No model for this horizon."); return

    if not artifact.get("has_edge", True):
        st.warning(
            "**This ranking is not evidence.** The model for this horizon showed "
            "no measurable out-of-sample edge, so the order below reflects how it "
            "sorts fundamentals in training, not a validated ability to pick "
            "winners. It is shown for inspection of the pipeline, not as a "
            "shortlist to act on.")

    # step=1, not 5. The universe holds 19 companies, and a step of 5 made 19
    # unreachable: the slider stopped at 15 and quietly excluded four of them
    # from every ranking, with nothing on screen to say so.
    largest = max(5, len(companies))
    size = st.slider("Universe size", 5, largest, min(len(companies), largest),
                     step=1, help=f"How many companies to score and rank. "
                                  f"{len(companies)} are available.")
    if controls["offline"]:
        st.caption("Cached mode — this ranking costs 0 API credits.")
    else:
        st.caption(f"Estimated cost: about {size * CREDITS_PER_COMPANY:,} API credits.")

    if not st.button("Show best 10", type="primary"):
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
    st.markdown(f"#### Best {len(top)} — {months} month outlook")
    st.dataframe(pd.DataFrame({
        "Rank": range(1, len(top) + 1),
        "Ticker": top.ticker.to_numpy(),
        "Company": top.company_name.to_numpy(),
        "Probability up": [f"{v * 100:.0f}%" for v in top.probability],
        "Risk": top.risk.to_numpy(),
        "Trend": top.trend.to_numpy(),
        "Data quality": [f"{v * 100:.0f}%" for v in top.data_quality],
    }), width="stretch", hide_index=True, column_config={
        "Rank": st.column_config.NumberColumn(width="small"),
        "Ticker": st.column_config.TextColumn(width="small"),
        "Company": st.column_config.TextColumn(width="large")})

    # Reliability is a property of the model, not of a row, so repeating it
    # down every line of the table only made the columns narrower.
    note(f"<strong>How to read this table.</strong> Stocks are ranked by the "
         f"model's estimated probability of a positive return over the horizon. "
         f"Model reliability for this horizon is <strong>{reliability}</strong>, "
         f"and it applies to every row equally. Risk and trend are measured "
         f"from price history alone, independently of the model, because a high "
         f"probability does not automatically mean low risk.")

    excluded = ranked[~ranked.eligible]
    if not excluded.empty:
        with st.expander(f"Excluded by quality gates ({len(excluded)})"):
            st.dataframe(excluded[["ticker", "company_name", "reason"]],
                         width="stretch", hide_index=True)
    st.caption(nq.DISCLAIMER)


# ══════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════

def main() -> None:
    configure_page()
    st.markdown('<div class="nq-title">NusaQuant</div>', unsafe_allow_html=True)
    st.markdown('<div class="nq-sub">IDX Market Intelligence · XGBoost probability '
                'estimates for positive 6-month and 12-month returns.</div>',
                unsafe_allow_html=True)

    models, metadata = load_models()
    controls = render_sidebar(metadata)

    if controls["offline"]:
        names = nq.company_names()
        companies = pd.DataFrame({
            "symbol": controls["snapshot"],
            "company_name": [names.get(t, t) for t in controls["snapshot"]]})
        st.info(f"Cached mode — no API credits are being spent. Figures are a "
                f"real Sectors snapshot taken on {snapshot_as_of()}, not today's "
                f"market.")
    else:
        if not controls["api_key"]:
            st.info(nq.WELCOME)
            st.caption(nq.DISCLAIMER)
            return
        try:
            companies = live_universe(
                controls["api_key"],
                metadata.get("universe_filter") or "market_cap > 1000000000000", 50)
        except nq.SectorsAPIError as error:
            show_error(error); return
        if companies.empty:
            st.error("The Sectors universe came back empty."); return

    if controls["mode"] == "Single Stock":
        render_single_stock(companies, models, controls)
    else:
        render_best_10(companies, models, controls)


if __name__ == "__main__":
    main()
