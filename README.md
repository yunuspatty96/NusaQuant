# NusaQuant - IDX Machine Learning Market Intelligence

> **Problem statement.** NusaQuant is for retail investors and researchers on
> the Indonesia Stock Exchange who need to know how much a holding is likely
> to move — and how much confidence that estimate has actually earned.

A probability with no tested record behind it cannot be weighed: it looks
identical whether it was validated or invented. NusaQuant attaches the
out-of-sample score to every figure it publishes, including the forecast that
failed its test and is labelled as such rather than quietly dropped.

An IDX research dashboard built entirely on the
[Sectors Financial API v2](https://docs.sectors.app/). It forecasts how much a
share is likely to move, ranks the market on that, and reports — on the page,
not in a footnote — that it could not forecast which way.

| | |
|---|---|
| **Track** | Market Intelligence |
| **Team** | Patty Kyoudai — Yunus Patty, Lukas Patty |
| **Live** | **[nusaquant.streamlit.app](https://nusaquant.streamlit.app)** |
| **Data** | Sectors Financial API v2, five endpoints, no other source |
| **Universe** | 34 IDX companies · 427 quarterly observations · 23 rebalances |

---

## The two questions

Both were put to the same data, through the same purged walk-forward
protocol, against the same 0.55 threshold.

| Question | Out-of-sample ROC-AUC | Verdict |
|---|---|---|
| Will this company swing more than the median one? | **0.676** at 6M · **0.690** at 12M | measurable edge |
| Will its price be higher in 6 or 12 months? | 0.520 and 0.443 | no measurable edge |

The first is what the dashboard ranks on, because it is the only ordering here
resting on something tested. The second stays on the page, labelled for what
it is. A test that found nothing is still a result, and removing it would
leave a reader no way to judge the one that found something.

Every figure above is printed by `python train.py --offline`, which needs no
API key and spends nothing.

---

## Why this is not a stock tip generator

The rules ask that products not provide financial advice, and that is also the
honest engineering position here. Three things are built in rather than
promised:

- **A model that cannot rank is not allowed to sound confident.** Below an
  out-of-sample ROC-AUC of 0.55 the reliability label becomes
  *No measurable edge*, and the probability is shrunk toward the historical
  base rate by a weight fitted leave-one-fold-out.
- **Probabilities are labelled as probabilities.** 53% means a 53% chance the
  price is higher, not a 53% gain — stated under every figure, because that is
  the most expensive misreading available.
- **Nothing places an order.** The product analyses, screens and scores. It
  never executes.

---

## Run it

```bash
pip install -r requirements.txt
streamlit run app.py          # 0 credits, no API key needed
```

**Cloning this repository costs nothing.** `data/cache/` holds 34
companies and `models/` holds four trained artifacts, both committed, so the
whole dashboard runs offline. An API key is needed only to extend the universe
or to analyse a company outside the snapshot.

```bash
export SECTORS_API_KEY=your-key-here
python train.py --dry-run     # see the plan and cost, spend nothing
python train.py --screen      # universe, classification, dividends — 1 credit
python train.py               # extend the universe
python train.py --offline     # re-train from cache — no key, no network
python test_nusaquant.py      # 147 checks, no network, no credits
```

The API key is read from the environment. It appears in no source file, is
never written to disk, never logged, and is sent only in a request header.

---

## The data behind it

Everything comes from the Sectors Financial API v2. Remove it and there is no
product: every metric, every model input, every chart and the entire cached
snapshot derive from these five endpoints.

| Endpoint | Used for | What it gives |
|---|---|---|
| `companies/` | Universe screen | Companies above the market-cap floor, with the cap itself, IDX sector / sub-sector / industry, and trailing dividends. One credit covers 200 companies. |
| `company/report/{ticker}/` | Company overview | Name, market cap and last close, for Live mode where no snapshot exists. |
| `financials/quarterly/{ticker}/` | Quarterly financials | The income statement, balance sheet and cash flow items every ratio is computed from. |
| `daily/{ticker}/` | Daily price history | OHLC, volume and market cap per trading day, fetched in 90-day chunks. |
| `subsectors/` | Sub-sector list | IDX sub-sector names, for grouping companies against peers. |

Three details worth naming, because each was a bug first:

- **The screener ignores `order_by`.** It is asked for `-market_cap` and
  returns the list alphabetically. The market cap is read back from each
  screen through `include_query_values` — it was already arriving and being
  discarded — and the ordering is applied locally. Without this the next
  companies bought are the ones beginning with A.
- **`daily/` silently clamps** any window wider than 90 days to the most
  recent 90, with no error. Long histories are fetched in consecutive chunks.
- **Failed calls are not billed.** The credit meter refunds 4xx and 5xx rather
  than counting them, so the budget reflects what Sectors actually charges.

---

## The four views

**Single Stock Analysis** — one company end to end: price history with MA20
and MA50, support and resistance, a calibrated projected range, RSI and MACD,
a 52-week range strip, the income statement as bars and lines, all
27 fundamentals with a reference level for each, then Risk
Analysis and Return Forecast.

**Machine Learning Screening** — the universe ranked by the volatility
forecast, calmest first, with both estimates side by side and every column
sortable. Risk Class places each company among all companies on file rather
than on an absolute scale.

**Portfolio Analysis** — holdings entered in lots. Value, volatility, worst
drawdown, the diversification benefit, a projected range per position and for
the whole, both forecasts per holding, sector proportion and the correlation
matrix. This is the one view no per-stock page can replace: diversification is
not a property any holding has alone.

**About Us** — the method in full, with every figure read from the trained
artifacts as the page renders rather than typed in.

---

## The models

| Question | Algorithm | Inputs | ROC-AUC | Folds | Rows | Verdict |
|---|---|---:|---:|---:|---:|---|
| Volatility, 6M | Gradient boosting, depth 1 | 4 | **0.676** | 8 | 255 | edge |
| Volatility, 12M | Logistic regression, L2 | 4 | **0.690** | 4 | 126 | edge |
| Return, 6M | Gradient boosting, depth 2 | 5 | 0.520 | 9 | 286 | none |
| Return, 12M | Gradient boosting, depth 1 | 5 | 0.443 | 5 | 157 | none |

Each is chosen per horizon from four candidates — two gradient-boosted trees
and two L2 logistic regressions — on out-of-sample log loss, with rank quality
breaking ties inside a tolerance of 0.005. Ties are declared generously
because heavily shrunk candidates separate on noise in the fourth decimal
while their ROC-AUCs still differ by real margins.

**Volatility models read four inputs**: trailing 3-month volatility, debt to
equity, distance from the 52-week high, and last month's move. Nearly all the
skill sits in the first — drop it and the score falls to near chance — and
saying so is more useful than implying something cleverer.

**Return models read scale-free ratios only.** Of 11
eligible, those missing for more than 30% of the
panel are dropped, and survivors must clear an information coefficient of
0.06. That bar comes from the panel's own noise floor: a column
of random numbers scores about 0.05 here, measured by permutation, so anything
below it cannot be told from noise.

Rupiah amounts are never inputs — a model splitting on the level splits on
company size rather than value. Dividend figures are shown but never modelled,
because they are current readings rather than point-in-time history.

---

## The metrics

27 metrics in 7 categories, all reconstructed point-in-time from the cached filings at zero credits except the three dividend figures, which are a screener snapshot and are therefore shown but never modelled.

| Category | Metrics | Model eligibility |
|---|---|---|
| Valuation | P/E, P/S, PBV, P/CF, EV/EBITDA | eligible |
| Per Share | EPS, RPS, CPS, BVPS, CFPS | shown only — rupiah amounts cannot be compared across companies |
| Solvency | DER | eligible |
| Profitability | ROA, ROE, GPM, OPM, NPM | eligible |
| Dividend | Dividend, DPR, Dividend Yield | shown only — screener snapshot, not point-in-time |
| Income Statement | Revenue, Gross Profit, EBITDA, Net Income | shown only — rupiah amounts cannot be compared across companies |
| Balance Sheet | Cash, Total Assets, Total Liabilities, Total Equity | shown only — rupiah amounts cannot be compared across companies |

Only scale-free point-in-time ratios are eligible as inputs; the
missingness gate and the information-coefficient screen then decide
which of those actually reach a model. `train.py` prints both
decisions on every run.

---

## How it is validated

**Purged walk-forward, never a random split.** One fold per quarterly
rebalance. A model is fitted only on rows whose forward window closed before
the quarter it is scored on, so a 12-month target observed in June 2022 is
withheld from any fold validating June 2023.

**Scored within each fold, never pooled.** The share of stocks that rose in a
quarter ranges from 0 to 1 across this panel. Pooled, a model emitting one
constant per quarter — no ranking whatsoever — scores 0.57, because the
constants sort good quarters above bad ones. Scored inside each quarter that
same model gets exactly 0.500, which is the truth about it.

**Point-in-time throughout.** A filing is unknown for
90 days after its reporting date. A nine-point leakage
audit runs on every training run and blocks the export if any check fails.

**Feature screening happens inside every fold.** Screening once on the whole
panel and hardcoding the winners scores better and is worth nothing: the
ratios would have been chosen using the returns the model is then graded
against.

**58 commits since 2026-09-05**, and a test suite of 147 checks that
runs with no network and no credits — including one asserting the entire demo
makes zero API calls.

---

## What was tried and did not work

Reported because a negative result is evidence, and because a repository that
only shows what worked gives a judge no way to weigh it.

| Idea | Result |
|---|---|
| Momentum, reversal, 52-week-high distance | Information coefficient 0.000 for 12-1 momentum. Nothing above the noise floor. |
| OHLCV-derived features for direction | 6M best combination 0.514 against 0.500 for fundamentals alone — inside one standard error. 12M actively worse. |
| Stratified K-Fold | +0.169 apparent gain, entirely look-ahead leakage. Rejected. |
| SMOTE / class weighting | Classes are 53/47, not imbalanced. Moved the score in opposite directions at the two horizons. |
| Fair value: ML multiple regression | PBV R² of −0.036 — worse than the mean. Cheap-versus-dear predicted returns with the wrong sign. |
| Fair value: Graham, Gordon, DCF, peer multiples | Five methods disagreed by 36× on one company. Cheap half underperformed by up to 11.9%. |
| Anomaly detection (isolation forest, LOF, elliptic) | Found real oddities, predicted nothing: +0.015 on returns against a 0.05 noise floor. |
| Probability calibration (isotonic, Platt) | Both worse than the shrinkage already applied. |
| More companies | Panel grew 15 → 19 → 22 → 25 → 31 → 34; 6M scores 0.470, 0.483, 0.516, 0.521, 0.499, 0.520. No trend. |

The binding constraint is **measurement precision, not the algorithm**. With
9 folds the standard error on the 6M ROC-AUC is about 0.043, so
the interval straddles the 0.55 gate. More quarters per
company buys folds; more companies does not.

---

## Calculations that are not machine learning

**Projected range** — a volatility cone: trailing daily volatility scaled by
the square-root-of-time rule, widened by multipliers measured on this panel
rather than taken from a textbook, then checked by projecting every past
observation and counting how often the price landed inside.

**Support and resistance** — swing highs and lows clustered by proximity, each
compared against its cluster anchor rather than its last member. Chaining
comparisons produces one meaningless level spanning the whole price range.

**Portfolio risk** — volatility from the covariance of daily returns across
holdings, aligned on dates every holding traded. The diversification benefit
is the weighted average of the parts less the volatility of the whole.

**Reliability score** — rank quality, calibration against a prior-only
baseline, and fold-to-fold stability. A model that cannot rank is refused a
stability credit, so it cannot accumulate a reassuring label on consistency
alone.

---

## Cached snapshot and Live mode

**Cached snapshot** is the default and costs nothing. It runs the entire
dashboard on real Sectors data collected during training, labelled with the
date it was taken — currently **2026-06-30**. Real market data,
but not today's market, and the page says so.

**Live Sectors API** fetches current figures for any listed company, including
those outside the stored universe. Every screen that spends credits shows the
estimate before the button is pressed.

The same code computes a feature in both modes. That is why the shared logic
lives in one file: a ratio computed one way in training and another at
inference would serve the model inputs meaning something different from what
it learned.

---

## Files

```
app.py               Streamlit dashboard — four views
nusaquant.py         API client, cache, features, targets, risk, explanations
train.py             CLI: collect -> validate -> train -> export
test_nusaquant.py    147 checks, synthetic API, zero network
models/              four models plus metadata.json
data/cache/          one parquet pair per company, plus the universe screen
DEPLOY.md            deployment steps and credit costs
```

Shared logic lives in `nusaquant.py` because `train.py` and `app.py` must
compute a feature *identically*, or the model is served inputs that mean
something different from what it learned.

---

## Limitations

- **Survivorship bias.** The universe reflects securities listed today, so
  companies delisted during the period are absent and performance is biased
  upward. This is why size and illiquidity are excluded from the risk model
  despite scoring higher: on a universe selected by today's market cap, "small
  predicts rising" is the selection rule read backwards.
- **The reporting lag is assumed.** A flat 90 days stands
  in for real publication dates.
- **Overlapping targets.** Consecutive observations share most of their
  forward window, so the effective sample is smaller than the row count.
- **34 companies, 23 quarters.** Every conclusion here is
  conditional on a panel this size.

---

## Disclaimer

NusaQuant provides quantitative analysis to support research and
decision-making. Model probabilities, forecasts, and other analyses are
estimates and may be inaccurate; they are not guarantees of future outcomes
and do not constitute financial advice. You are solely responsible for your
own decisions and assume all associated risks.

---

NusaQuant © 2026 Patty Kyoudai · Yunus Patty · Lukas Patty
Built for the Sectors Hackathon 2026.
