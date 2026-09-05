# NusaQuant © 2026 Patty Kyoudai

**IDX market-intelligence dashboard that ranks Indonesian stocks by the
model-estimated probability of a positive return over 6 and 12 months.**

Data: [Sectors Financial API v2](https://docs.sectors.app/). Model: XGBoost.

> Research and decision support only. Probabilities are model estimates, not
> guarantees and not financial advice.

**Deployment steps: see [DEPLOY.md](DEPLOY.md).**

---

## What it costs

| | Credits |
|---|---:|
| Training (one time, ~13 companies) | ~495 |
| Re-training (`--offline`), deployment, demo | **0** |
| Live analysis of a company outside the snapshot | ~9 |

The dashboard ships in **Cached snapshot** mode: it runs the whole flow on the
real Sectors data collected during training, at zero credits, labelled with the
date the snapshot was taken.

---

## Files

```
app.py            Streamlit dashboard
nusaquant.py      API client, cache, features, targets, risk, explanations
train.py          CLI: collect -> validate -> train -> export
requirements.txt
DEPLOY.md         deployment steps + credit costs
models/           model_6m_xgb.joblib, model_12m_xgb.joblib, metadata.json
data/cache/       one parquet pair per company (the snapshot)
```

Three Python files. The shared logic lives in `nusaquant.py` because both
`train.py` and `app.py` must compute a feature *identically* — otherwise the
model is served inputs that mean something different from what it learned.

---

## Quick start

```bash
pip install -r requirements.txt
export SECTORS_API_KEY=your-key-here

python train.py --dry-run      # see the plan and cost, spend nothing
python train.py                # ~495 credits, once
python train.py --offline      # re-train from data/cache/ — no key, no network
streamlit run app.py           # 0 credits
```

The API key is read from the environment. It appears in no source file, is
never written to disk, never logged, and is sent only in a request header.

---

## The ten features

All ten are reconstructible point-in-time from the v2 API.

| Feature | Category | Notes |
|---|---|---|
| PE | Valuation | Market cap ÷ TTM earnings; NaN when earnings ≤ 0 |
| PB | Valuation | Market cap ÷ book equity |
| PS | Valuation | Market cap ÷ TTM revenue |
| ROE | Profitability | TTM earnings ÷ 4-quarter average equity |
| ROA | Profitability | TTM earnings ÷ 4-quarter average assets |
| Net Profit Margin | Profitability | |
| Debt-to-Equity | Leverage | (assets − equity) ÷ equity, i.e. **total liabilities** to equity — the API does not expose interest-bearing debt separately |
| Earnings Growth 1Y | Growth | TTM vs TTM, four quarters apart |
| Revenue Growth 1Y | Growth | As above |
| Accruals | Earnings quality | (earnings − operating cash flow) ÷ assets |

A ratio that is economically meaningless becomes `NaN`, never zero: a PE built
on negative earnings is a category error, not a cheap stock. A negative ROE
*is* meaningful and is kept.

The growth pair needs eight quarters of warm-up, so at `--quarters 16` they are
mostly missing and the missingness gate drops them, leaving eight features.
`--quarters 24` keeps all ten at the cost of a narrower universe. `train.py`
prints which features survived and why.

---

## Method

**Classification, not price forecasting.** The product question is directional,
so the model estimates `P(forward return > 0)`. A claim that can be checked
against what happened, rather than a price target that mostly cannot.

**Point-in-time alignment.** A statement dated 31 March was not public on
31 March, so fundamentals are held back 90 days before becoming eligible:

```
report date ──+90d──▶ available ──▶ next trading day ──▶ observation
                                                             │
                                                    +126 / +252 trading days
                                                             ▼
                                                        6M / 12M target
```

Targets use trading-day offsets into each ticker's own price series, never
calendar arithmetic, because IDX closes for weekends and many national
holidays.

**Cumulative filings are detected and corrected.** Indonesian issuers commonly
file cumulative year-to-date income statements — Q4 is the full year. Summing
four of those as standalone quarters overstates TTM revenue by roughly 2.5x.
The basis is detected per company by comparing the level of the last quarter
against Q1 (cumulative predicts Qn ≈ n × Q1, discrete predicts ≈ Q1).

**Purged walk-forward validation, one fold per rebalance.** Expanding folds,
each validating a single quarterly observation date. A 12-month target observed
in June 2022 does not resolve until June 2023, so a row is admitted to training
only once its own forward window has closed. Imputation and the rank transform
sit inside the pipeline and refit per fold.

Folding by rebalance rather than by calendar year is what makes the numbers
mean anything here: yearly folds gave two validation folds at 6M and one at 12M,
and a stability score computed across two numbers is not a measurement. Per
rebalance there are nine and five.

**Rank quality is averaged within folds, never pooled across them.** The share
of stocks that rose over the following six months ranges from 0.00 to 1.00
across the quarters in this panel. Pool the folds together and a model emitting
one constant per quarter — the same number for every stock, no ranking at all —
scores a pooled ROC-AUC of 0.57, because its constants happen to sort the good
quarters above the bad ones. That is the market's direction leaking into a
metric meant to measure stock selection. Scored inside each cross-section, the
same model gets exactly 0.500.

**Capacity is matched to the sample.** Measured on purged folds, the depth-3,
250-tree configuration this project shipped in 0.2.0 reached an in-sample
ROC-AUC of 0.79–0.86 against an out-of-sample 0.43. It was not learning the
market; it was learning fifteen tickers. The candidates are now depth-1 and
depth-2 trees and regularised logistic regression, all of which roughly halve
that gap.

**The model is selected, not assumed.** Every candidate is validated on the
same folds and ranked on out-of-sample log loss — a proper scoring rule, so a
confidently wrong model cannot win — with rank quality breaking ties inside
0.005. `DummyClassifier(strategy="prior")` is always measured alongside, and
`train.py` states plainly whether anything beat it.

**Probabilities are shrunk toward the base rate.** The served number is
`w × model + (1 − w) × prior`, with `w` fitted leave-one-fold-out on
out-of-sample log loss. Blending is monotone, so the ranking never changes;
only how far a probability may travel from the historical frequency does. On
this panel `w` comes back at its floor, which is the honest answer rather than
a disappointing one.

**Reliability** = 40% normalised ROC-AUC + 25% normalised PR-AUC + 20%
calibration + 15% stability, and each of those three qualifiers is doing work:

- PR-AUC is normalised against the base rate, since a PR-AUC of 0.65 is
  worthless when 65% of observations are positive.
- Calibration is a Brier **skill score against the baseline's own out-of-sample
  Brier**, not against `p(1−p)`. The latter is the score of a forecaster who
  already knows the validation base rate — an oracle — and measured against it
  even a perfectly honest model scores zero. That is why every calibration
  component in the 0.2.0 artifacts read exactly 0.
- Stability is withheld from a model that does not discriminate. A classifier
  returning the same number for every stock has a fold-to-fold AUC standard
  deviation of zero and would otherwise collect a perfect 100 for it.

**A model with no measurable edge says so.** Below an out-of-sample ROC-AUC of
0.55 the reliability label becomes *No measurable edge*, the probability band
stops naming an edge it cannot demonstrate, and the Best 10 carries a warning
that its ordering is not evidence. A model cannot accumulate its way to a
reassuring label on calibration and consistency alone; those describe a
well-behaved forecast of the base rate, which is a different claim from a
signal.

**Risk is measured separately** from any probability, because a stock with a
high probability of a positive return can still be violently volatile.

---

## What the current snapshot actually measures

On the shipped 15-ticker snapshot, **neither horizon has a measurable edge**:

| | 6M | 12M |
|---|---:|---:|
| Purged folds | 9 | 5 |
| Out-of-sample rows | 135 | 75 |
| ROC-AUC (mean within fold) | 0.470 | 0.478 |
| Baseline ROC-AUC | 0.500 | 0.500 |
| Beats the prior-only baseline on log loss | no | no |
| Reliability | No measurable edge | No measurable edge |

That is reported, not hidden. The dashboard labels both horizons, shrinks the
probabilities to within about two points of the base rate, and warns on the
ranking view.

**Why, and what would change it.** The target is the sign of an absolute
return, and over 6–12 months that sign is mostly the market's, not the
company's — the per-quarter base rate in this panel runs from 0.00 to 1.00, and
cross-sectional fundamentals carry no information about the market's own
direction. On top of that, fifteen tickers means each quarterly cross-section
is fifteen points wide. The binding constraint is the **width of the universe**,
not the algorithm: every model tried, linear and tree, on fundamentals, on
zero-credit price features, and on both, landed between 0.40 and 0.53
out of sample. Widening the universe adds far more effective sample than adding
history to the same fifteen names.

## Limitations

- **Survivorship bias.** The universe reflects securities that exist today, so
  companies delisted during the period are absent and performance is biased
  upward.
- **The reporting lag is assumed.** A flat 90 days stands in for real
  publication dates, which the API does not reliably expose.
- **Overlapping targets.** Consecutive observations share most of their forward
  window, so the effective sample is smaller than the row count.
- **Small dataset on a small budget.** ~200 rows across 15 companies cannot
  support a strong claim, and on this snapshot it does not support one at all.
- **The target carries the market.** See above — this caps how well any model
  built only on cross-sectional fundamentals can score.
- **Not a backtest.** No transaction costs, slippage or position sizing.
- **One model across all sectors.** Banks and miners do not have comparable
  balance sheets. The rank transform softens this; it does not fix it.

---

## Disclaimer

NusaQuant provides quantitative analysis for research and decision support.
Model probabilities are estimates, not guarantees or financial advice. Market
data © [Sectors](https://sectors.app/), used under their API terms.

## Developer
Patty Kyoudai
-Yunus Patty
-Lukas Patty