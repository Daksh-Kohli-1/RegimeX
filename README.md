# RegimeX - Volatility Regime Forecasting with Deep Learning

A scoped ML project: engineer financial features, label volatility regimes,
forecast the next regime with classical and deep models, and evaluate the
predictions honestly, including with a simple risk-adjusted strategy.

This is **not** "predict stock price with LSTM." It is "predict market *state*
(calm / normal / turbulent), compare model families fairly, and be explicit
about what the numbers do and do not prove."

> **How to read this README.** Sections 1-7 describe what is built and how to
> run it. Section 8 reports results from the ORIGINAL pipeline (before the
> upgrades) and is kept for comparison; regenerate your own numbers with
> `python run_all.py`. Section 9 is the upgrade roadmap: each upgrade explains
> *why it matters*, *what changes if added*, *what happens if it is missing*,
> and *what the industry does*, and carries a **Status** line saying where it
> lives in the code. **All of Section 9 is now implemented, except intraday data
> in 9.10.** Section 10 is a Q&A for presentations.

---

## 1. The idea in plain English

Think of a weather forecast for market turbulence. Some periods are calm,
with small daily price moves. Others are stormy, with large swings. Storms
cluster: a turbulent day is usually followed by another. That makes
turbulence far more forecastable than price direction, and it matters for
risk: if you expect a storm, you can hold less.

The model reads today's data and answers one question:
**will tomorrow's market state be low-, medium-, or high-volatility?**

- **Target asset:** SPY only (an ETF tracking the S&P 500, the standard
  stand-in for "the market").
- **Inputs:** features computed from 9 ETFs (SPY, QQQ, DIA, XLF, XLK, XLE,
  XLV, TLT, GLD). The other 8 are inputs, not targets: tech, banks, energy,
  bonds and gold often move before or alongside the broad market, and
  correlated moves across them are a classic stress signal.
- **Data:** daily **adjusted closing prices** only (`auto_adjust=True`, which
  corrects for dividends and splits). Open/high/low/volume are not used.
  Features for day *t* use only data up to the close of day *t*.

---

## 2. Pipeline

```
raw prices → features → target vol → regime labels → models → walk-forward → strategy → evaluation
 (yfinance)   (returns,   (FUTURE 20d   (GMM/HMM/pct,   (Persistence, GARCH,  (purged,      (vol targeting  (accuracy/F1,
               vol, mom,   realized vol)  fit on TRAIN    LogReg, GBT, LSTM,    expanding)    + costs)        Sharpe, report)
               RSI, MACD)                 only)           Transformer)
```

1. **Feature engineering**: log returns, rolling volatility (5/10/20d), momentum (5/10/20d), RSI(14), MACD (divided by price so it is stationary), plus a cross-asset 20-day average pairwise correlation. About 100 features.
2. **Target and regime labeling**: the target for day *t* is the realized volatility of days *t+1 ... t+20* (forward label, no overlap with the inputs). A labeler (GMM by default; percentile or HMM available) turns it into low / medium / high. **The labeler and the scaler are fit on training rows only.**
3. **Models**: Persistence and GARCH benchmarks, Logistic Regression and Gradient Boosted Trees (classical baselines), LSTM and Transformer (sequence models), all scored on identical test dates.
4. **Validation**: one chronological holdout and an expanding-window walk-forward, both with a 20-day purge gap between train and test.
5. **Strategy layer**: the original toy rule and volatility targeting, with transaction costs and cash earning the risk-free rate (illustrative only).
6. **Evaluation**: accuracy, macro-F1, confusion matrices, margin over persistence, Sharpe vs. buy-and-hold, feature importance, with limitations stated explicitly.
7. **Deployment**: intentionally skipped.

---

## 3. Project layout

```
volatility_regime_forecasting/
├── README.md
├── requirements.txt           # + requirements-optional.txt (shap)
├── config.yaml                ← tickers, dates, label mode, hyperparams, costs
├── run_all.py                 ← runs the full pipeline (--no-deep, --no-walk-forward, --no-explain)
├── src/
│   ├── data_pipeline.py       # download/cache prices (or synthetic data); cache is validated against config
│   ├── features.py            # feature engineering
│   ├── labeling.py            # forward target + RegimeLabeler (GMM / percentile / HMM), fit on train only
│   ├── dataset.py             # Fold object: purged splits, train-only scaling/labeling, walk-forward folds
│   ├── metrics.py             # shared scores + common prediction format
│   ├── models_baseline.py     # LogReg, GBT
│   ├── models_benchmarks.py   # Persistence, GARCH / GJR-GARCH
│   ├── models_deep.py         # LSTM, Transformer (PyTorch)
│   ├── walkforward.py         # walk-forward validation over all models
│   ├── strategy.py            # toy rule, vol targeting, costs, backtest
│   ├── explain.py             # GBT / permutation importance, optional SHAP
│   ├── evaluate.py            # tables, plots, report.md
│   └── utils.py
├── data/                      # cached data (created on run)
├── outputs/
│   ├── figures/               # regimes, confusion matrices, equity curves, importance
│   ├── models/                # saved weights
│   ├── *.csv, *.json          # predictions, walk-forward scores, importance
│   └── report.md              # auto-generated results
├── sample_outputs_synthetic_demo/   # example outputs from a reduced synthetic run
└── tests/
    └── test_pipeline.py
```

---

## 4. Setup

Requires **Python 3.10+**.

```bash
cd volatility_regime_forecasting
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -r requirements-optional.txt   # optional: SHAP plots
```

If you have a GPU, install torch first per https://pytorch.org/get-started/locally/.

## 5. Run it

```bash
python run_all.py
```

Faster variants:

```bash
python run_all.py --no-deep            # skip LSTM/Transformer (much faster)
python run_all.py --no-walk-forward    # holdout only
python run_all.py --no-explain         # skip feature importance
```

Individual stages (each writes files that `evaluate` reads):

```bash
python -m src.data_pipeline
python -m src.features
python -m src.labeling            # must run before evaluate (writes the regime timeline + metadata)
python -m src.models_baseline
python -m src.models_benchmarks   # persistence + GARCH
python -m src.models_deep
python -m src.walkforward
python -m src.explain
python -m src.evaluate
```

**Runtime.** `GradientBoostingClassifier` is single-threaded and is trained once for the holdout and once per walk-forward fold, so a full run takes on the order of 10-30 minutes on CPU, depending on the machine. Use `--no-deep` or lower `gbt.n_estimators` for quick iterations.

**Offline?** Set `use_synthetic: true` in `config.yaml` to run the same pipeline on a synthetic regime-switching price series. The report labels it as synthetic so it is never confused with real results.

---

## 6. Configuration

The full file is `config.yaml`. The keys that matter most:

```yaml
tickers: [SPY, QQQ, DIA, XLF, XLK, XLE, XLV, TLT, GLD]
start_date: "2012-01-01"
end_date: "2025-01-01"
use_synthetic: false
use_extended_universe: false     # 9.10: add 13 more ETFs and start in 2007

vol_window: 20            # horizon/window of the realized-vol target
n_regimes: 3
labeling_method: gmm      # "gmm" | "percentile" | "hmm"
label_mode: forward       # "forward" (honest) | "backward" (legacy, overlapping)
embargo_days: auto        # purge gap: vol_window for forward labels

sequence_length: 30
test_size: 0.2
random_seed: 42

benchmarks: {persistence: true, garch: true, garch_variant: gjr}
walk_forward: {enabled: true, min_train_days: 1000, test_window_days: 252, step_days: 252, include_deep: true}

transaction_cost_bps: 5
cash_earns_rf: true
vol_target: {target_annual_vol: 0.12, max_leverage: 1.0, min_vol_floor: 0.03}
explain: {enabled: true, permutation_repeats: 5, use_shap: true}

lstm:        {hidden_size: 64, num_layers: 2, dropout: 0.2, epochs: 40, batch_size: 64, lr: 0.001, patience: 8}
transformer: {d_model: 64, nhead: 4, num_layers: 2, dim_feedforward: 128, dropout: 0.2, epochs: 40, batch_size: 64, lr: 0.001, patience: 8}
```

Setting `label_mode: backward` and `macd_normalize_by_price: false` reproduces the original pipeline's behavior (apart from the shared test dates and purge).

---

## 7. Tests

```bash
python -m pytest tests/ -v
```

40 checks, no network needed: features and forward targets use no future data; the labeler and scaler ignore the test period; the purge gap and chronological order hold in the holdout and every walk-forward fold; walk-forward test windows never overlap; sequence windows end at the day they describe; persistence uses only trailing data; GARCH parameters depend only on training returns; backtest timing (decide at *t*, earn *t+1*), costs and cash interest; buy-and-hold is identical across models; macro-F1 ignores absent regimes; Sharpe matches a hand-computed reference.

---

## 8. Results from the ORIGINAL pipeline (real data, 2012-2025)

> **Read this first.** These numbers come from the pipeline *before* the Section 9
> upgrades: backward (overlapping) labels, a GMM fit on the full sample, no purge,
> no persistence baseline, and raw-price MACD. They are kept as the "before" picture
> and are inflated by the label overlap described in 9.1. They have **not** been
> regenerated with the upgraded code, because that needs live market data: run
> `python run_all.py` and read `outputs/report.md` for current numbers. To
> approximate the old setup, set `label_mode: backward` and
> `macd_normalize_by_price: false`.

**Test period:** about June 2022 to December 2024 (the newest 20% of days,
roughly 650 trading days).

| Model | Accuracy | Macro-F1 |
|---|---|---|
| Gradient Boosted Trees | 95.2% | 0.947 |
| Logistic Regression | 92.0% | 0.909 |
| LSTM | 85.7% | 0.844 |
| Transformer | 85.0% | 0.830 |

Reference floors: random = 33.3%; majority-class (always "low") = 73.4% over
the full label history.

### How to interpret these numbers

- **All four models beat the majority-class floor**, so they learned something
  real. The classical models beat the deep ones. That is a normal result for a
  small daily-frequency dataset (about 3,000 rows): deep networks have little
  data to learn from and overfit, while boosted trees do well on tabular data.
- **Confusion matrices** (rows = true regime, columns = predicted). For GBT:
  414 low days and 205 medium days correct, 17 low days called medium, 14
  medium days called low. The LSTM tends to raise false alarms (78 low days
  called medium); the Transformer errs in both directions.
- **The high-volatility class is absent from the test period.** The whole
  "high_vol" row and column are zero. Only about 1.1% of all days are
  high-vol (essentially the March 2020 COVID crash), and 2020 falls in the
  training period. **The models were therefore never tested on the dangerous
  regime**, and macro-F1 effectively averages two classes. We cannot claim
  the models detect crises.
- **The majority baseline on the test set itself is lower than 73.4%.**
  About 431 of 650 test days (about 66%) are low-vol, so the models' margin
  over the majority baseline is smaller than the headline floor suggests.
- **Sequence models see slightly fewer test days** (621 vs. 650) because the
  30-day lookback consumes the first days of the test period.

### Regime label sanity check

`outputs/figures/regime_labels_timeline.png` shows SPY realized volatility
with regime shading. The shaded periods match real events: late 2015 to early
2016, February 2018, late 2018, the March 2020 COVID crash (volatility spike
near 0.94), and the long 2022 bear market. This supports the labels being
meaningful, although they remain an unsupervised proxy.

### Strategy results (illustrative only)

| Model | Strategy Sharpe | Buy&hold Sharpe | Strategy return | Buy&hold return |
|---|---|---|---|---|
| LogisticRegression | 0.859 | 0.906 | 35.4% | 48.4% |
| GradientBoostedTrees | 0.978 | 0.906 | 39.3% | 48.4% |
| LSTM | 1.028 | 1.198 | 37.0% | 61.1% |
| Transformer | 1.051 | 1.198 | 39.5% | 61.1% |

- The strategy **trades some return for a smoother ride**. In the 2022
  drawdown it fell less than buy-and-hold (roughly $0.93-0.95 vs. $0.87 low),
  then lagged during the steady 2023-2024 rally.
- Only GBT beats its buy-and-hold on Sharpe, and the margin is small. With
  one test period, no trading costs and a toy rule, this does not show the
  strategy "works."
- **Open item (now resolved):** buy-and-hold differed between rows (0.906 / 48.4%
  vs. 1.198 / 61.1%). The cause was confirmed in the code: the sequence models
  lost their first 29 test days to the 30-day lookback, so they were scored on a
  shorter, later window. The upgraded code lets sequence windows look back into
  pre-test rows (past information) so every model is scored on identical dates,
  and a test asserts that buy-and-hold is identical across models.

---

## 9. Upgrade roadmap: why each matters, what changes, what the industry does

Each entry answers: **What is it? Why does it matter? What happens without
it? What would change if added? What does the industry use?**

### 9.1 Forward-looking labels (fix the overlap)

**Status: implemented.** `label_mode: forward` (default) in `src/labeling.py::compute_vol_frame`; `backward` kept for comparison.

- **What it is.** Currently the label for "tomorrow" is the 20-day realized
  volatility *ending* tomorrow. Nineteen of those 20 days are already inside
  today's input features. A forward label is computed from the 20 days
  *starting* tomorrow, so inputs describe the past and the label describes a
  separate future, with zero shared days.
- **Why it matters.** The project claims to forecast. With overlapping
  windows the model is partly copying information it already has, so accuracy
  overstates real forecasting skill.
- **Without it.** Headline accuracy (mid-90s) is inflated. Part of it is
  genuine volatility persistence, but part is a measurement artifact.
- **If added.** Accuracy will likely fall and become more honest. What matters
  is whether the models still beat the persistence baseline (9.2).
- **Why we did not do it originally.** Rolling windows are the standard way to
  measure volatility because a single day's return is mostly noise. The
  overlap is a side effect of a sliding window, not a deliberate choice. It is
  fine if the question is "can the model recognise the current state"; it is a
  problem when the question is "can the model predict the future."
- **Industry practice.** Forecast targets are defined over a *future* horizon
  (for example, realized volatility over the next 5-22 trading days), with
  features strictly from the past.

### 9.2 Persistence baseline

**Status: implemented.** `src/models_benchmarks.py::persistence_forecast`; reported in every table, with an "accuracy minus persistence" column.

- **What it is.** The simplest possible predictor: "tomorrow's regime equals
  today's regime." No learning involved.
- **Why it matters.** Volatility is very sticky, so this trivial rule may score
  very high. It tells you how much of each model's accuracy comes from "no
  change" and how much the model genuinely adds.
- **Without it.** A 95% score looks impressive but is uninterpretable. If
  persistence scores 94%, the model added almost nothing.
- **If added.** One extra row in the results table, and the central comparison
  becomes "model vs. persistence" rather than "model vs. 33% or 73%."
- **Industry practice.** Naive/random-walk or "last value" benchmarks are
  mandatory in forecasting. A model that cannot beat the naive forecast is not
  considered useful.

### 9.3 Fit the labeling model on training data only

**Status: implemented.** `src/labeling.py::RegimeLabeler` is fit on training rows only, per fold. Side effect: regimes are relative to each training window.

- **What it is.** The GMM that defines the regimes is currently fit on the
  full sample, including the test period. It should be fit on training data
  and then applied forward.
- **Why it matters.** Otherwise the label boundaries use some future
  information.
- **Without it.** A mild leakage. Because the clustering is unsupervised, the
  effect is small, but it is technically a leak.
- **If added.** Cleaner experimental design with little change to results.
- **Industry practice.** Every fitted component (scalers, clusterers,
  encoders) is fit on training data only, then applied to validation and test.

### 9.4 Walk-forward validation

**Status: implemented.** `src/walkforward.py` (expanding window, ~1-year test windows, all models re-fit per fold; pooled out-of-sample strategy results).

- **What it is.** Repeated train/test cycles that move forward in time, for
  example train 2012-2016 and test 2017; train 2012-2017 and test 2018; and so
  on, then average the scores.
- **Why it matters.** A single split can be lucky or unlucky. Our single test
  window (2022-2024) contains no high-volatility days at all.
- **Without it.** Results reflect one historical period and may not
  generalise. The crisis regime is never tested.
- **If added.** Several test windows, including 2020, and a distribution of
  scores instead of one number.
- **Industry practice.** Walk-forward (rolling or expanding window) testing is
  the standard for time-series models in quantitative finance.

### 9.5 Purging and embargo (a gap between train and test)

**Status: implemented.** `embargo_days: auto` drops the last 20 training rows before each test window (`src/dataset.py::make_fold`), including between train and validation for the deep models.

- **What it is.** When labels look forward 20 days, the last training labels
  overlap the start of the test period. Leave a gap of about 20 days between
  train and test, and drop the affected samples.
- **Why it matters.** Without it, training labels contain information about
  the test period.
- **Without it.** A subtle leak that inflates test performance.
- **If added.** Slightly smaller datasets and more trustworthy scores.
- **Industry practice.** Popularised by Marcos López de Prado as purged
  k-fold cross-validation with an embargo, used wherever labels span time
  windows.

### 9.6 Volatility targeting and transaction costs in the strategy

**Status: implemented.** `src/strategy.py`: volatility targeting, `transaction_cost_bps`, cash earning the risk-free rate. The Persistence row under vol targeting is the no-ML "scale by recent vol" benchmark. Kelly/risk parity are not implemented.

- **What it is.** (a) Replace the toy rule with *volatility targeting*: scale
  exposure so portfolio risk stays at a constant level (hold less when
  volatility is high, more when low). (b) Subtract trading costs and slippage
  on every change in position.
- **Why it matters.** The toy rule only de-risks and assumes trading is free,
  so it will usually trail buy-and-hold and ignores costs real traders pay.
- **Without it.** The strategy result is illustrative only and cannot support
  any claim of usefulness.
- **If added.** A fairer test of whether regime predictions add value
  compared with simply scaling exposure by recent volatility.
- **Industry practice.** Volatility targeting (also risk parity, Kelly
  sizing) is common in managed-futures and multi-asset funds; every
  professional backtest includes a cost model.

### 9.7 Hidden Markov Model (HMM) labeling

**Status: implemented.** `labeling_method: hmm` (needs `hmmlearn`; best of 5 EM restarts, ordered by severity). The report prints the transition matrix and expected regime durations. Forecasting directly from transition probabilities is not implemented.

- **What it is.** The GMM labels each day independently. An HMM assumes the
  market *switches* between hidden states with transition probabilities (for
  example a 98% chance calm stays calm).
- **Why it matters.** It models regime persistence explicitly and gives
  transition probabilities, which is closer to how regimes actually behave.
- **Without it.** Labels can flicker near cluster boundaries, and persistence
  is only implicit.
- **If added.** Smoother regimes and the ability to forecast with transition
  probabilities.
- **Industry practice.** Markov regime-switching models (Hamilton) are a
  well-established tool in econometrics and macro/asset-allocation research.

### 9.8 GARCH benchmark

**Status: implemented.** `src/models_benchmarks.py::garch_forecast`: GJR-GARCH(1,1) (or GARCH) parameters estimated on training returns with `arch`, variance recursion run forward with fixed parameters, 20-day average forecast mapped to a regime. Verified against `arch`'s own forecasts.

- **What it is.** GARCH is the classic finance model for volatility
  forecasting: tomorrow's variance is a combination of a long-run average,
  yesterday's surprise (squared return), and yesterday's variance.
- **Why it matters.** It is the standard reference model in the field. Beating
  it, or honestly losing to it, makes the project's claims credible to
  practitioners.
- **Without it.** The comparison is machine learning vs. machine learning
  only, with no link to established finance practice.
- **If added.** A domain-appropriate benchmark that a finance reviewer will
  recognise.
- **Industry practice.** GARCH and its variants (EGARCH, GJR-GARCH) are widely
  used for risk models, and implied volatility (the VIX) is the market's own
  forecast.

### 9.9 Explainability (SHAP / feature importance)

**Status: implemented.** `src/explain.py`: GBT impurity importance, permutation importance on the test window, importance grouped by feature family and ticker, optional SHAP.

- **What it is.** Tools that show which input features drove predictions.
- **Why it matters.** It turns a black-box score into a story, for example
  "short-window volatility features dominate, and cross-asset correlation
  matters during stress."
- **Without it.** You can report *that* a model works but not *why*.
- **If added.** Insight into what the model relies on, and a check that it
  uses sensible signals rather than artifacts (such as the overlap in 9.1).
- **Industry practice.** Model risk teams require interpretability; SHAP and
  permutation importance are common.

### 9.10 More data

**Status: partly implemented.** `use_extended_universe: true` adds 13 ETFs and starts in 2007 (cache is rebuilt automatically). Intraday realized volatility is **not** implemented: free intraday history is too short to be useful.

- **What it is.** More years, more assets, or higher-frequency (for example
  5-minute) data, from which daily volatility can be measured much more
  precisely.
- **Why it matters.** About 3,000 daily rows is small for LSTMs and
  Transformers. Intraday data is how professionals build realized-volatility
  measures.
- **Without it.** Deep models tend to overfit and lose to boosted trees, as in
  these results.
- **If added.** Deep models become more competitive, though they are not
  guaranteed to win.
- **Industry practice.** Realized volatility from intraday data, and models
  trained across many assets, are the norm.

### Summary table

| Topic | Original | Weakness | Upgrade (status) | Industry standard |
|---|---|---|---|---|
| Label | Backward 20-day realized vol | 19 days overlap with inputs | Forward-looking label (done) | Future-horizon realized vol target |
| Baseline | Majority class / random | Not the hard benchmark | Persistence baseline (done) | Naive/last-value benchmark |
| Label fitting | GMM on full sample | Mild future leakage | Fit on train only (done) | Fit everything on train only |
| Validation | One chronological split | One lucky/unlucky window | Walk-forward + purge/embargo (done) | Walk-forward, purged CV |
| Strategy | Toy exposure rule, free trading | Cannot support claims | Vol targeting + costs (done) | Vol targeting/risk parity, cost models |
| Regime model | GMM | No persistence | HMM (done) | Markov regime switching |
| Finance benchmark | None | No link to the field | GARCH (done) | GARCH family, implied vol (VIX) |
| Interpretability | None | Black box | SHAP / importance (done) | Required in model-risk review |
| Data | Daily closes, 9 ETFs | Small for deep nets | More assets (done) / intraday (not done) | Intraday realized vol |

---

## 10. Presentation Q&A: why we did what we did

**Why classify regimes instead of predicting prices?**
Price direction is close to unpredictable. Volatility clusters and persists,
so it is a more defensible target, and it directly supports risk decisions.

**Why only SPY as the target if you use 9 ETFs?**
SPY represents "the market." The other 8 ETFs are inputs: sector, bond and
gold moves carry information about market stress that SPY alone may not.

**Why a chronological split instead of a random one?**
Shuffling would let the model train on days after the ones it is tested on,
which is leakage. Financial data is autocorrelated, so random splits inflate
scores.

**Why scale features on the training set only?**
Fitting the scaler on all data leaks test-set statistics into training.

**What are train, validation and test sets?**
Train is where the model learns. Validation is used during training for early
stopping (to prevent overfitting). Test is touched once at the end for an
honest score.

**Why four models?**
Logistic Regression is the simplest floor, Gradient Boosted Trees is a strong
classical model for tabular data, and the LSTM and Transformer test whether
sequence models add value. Deep learning only counts if it beats simpler
models.

**Why did the simple models win?**
About 3,000 daily samples is small for neural networks, and trees handle
tabular features well. This is a common, legitimate finding.

**Why accuracy and macro-F1?**
Accuracy is the share correct but is misleading when one class dominates
(73% of days are low-vol). Macro-F1 averages per-class performance so rare
classes are not ignored. The confusion matrix shows *which* mistakes happen.

**Why is accuracy so high? Is that suspicious?**
In the ORIGINAL pipeline, partly yes: the 20-day label window shared 19 days
with the input features (9.1). The upgraded pipeline uses forward labels and
reports a persistence baseline, so the question becomes "by how much does each
model beat persistence?". Expect much lower, more honest accuracy than the
original mid-90s numbers.

**What is the Sharpe ratio and why use it?**
Return divided by risk. A strategy earning 10% with wild swings is worse than
one earning 8% smoothly. It is the standard way to compare strategies, so it
is used instead of raw profit.

**Does your strategy work?**
In the original run, only weakly and only for one model on Sharpe: it lowered
risk (smaller 2022 drawdown) at the cost of return. The upgraded report adds
volatility targeting, costs and walk-forward results, and compares every model
with the no-ML "scale by recent vol" benchmark. Even so it is illustrative, not
evidence of alpha.

**What is the biggest weakness?**
In the original run, the test period contained no high-volatility days and the
label overlap inflated accuracy. Forward labels and walk-forward validation
address both; the remaining weakness is that rare regimes still have few
observations and everything rests on one market history.

**What would you do next?**
Those steps are done (Section 9). Next: intraday realized volatility,
hyper-parameter tuning inside each walk-forward fold, Kelly/risk-parity sizing,
and forecasting directly from HMM transition probabilities.

**"Aren't LSTMs old?"**
Model fashion is not the point. The value is in framing the problem honestly,
evaluating it correctly, and reporting what actually happened. Here the
simpler models beat the newer ones, which is a real finding. Professionals in
volatility forecasting still rely on GARCH-type models and gradient boosting
because they perform well.

---

## 11. Explicit limitations

- **Regime labels are a proxy**, not ground-truth economic regimes, and they are defined relative to each training window (a calm training period calls smaller spikes "high").
- **One market history.** Walk-forward gives several test windows, but rare regimes still have few observations.
- **Persistence and GARCH are strong benchmarks.** A model that does not clearly beat them adds little.
- **Simple cost model** (proportional bps): no size-dependent slippage, market impact or taxes.
- **Simple sizing**: vol targeting only; no Kelly, risk parity or drawdown control.
- **Fixed hyper-parameters**, not tuned inside each fold.
- **Small universe** (9 ETFs by default) and **daily closes only**; no intraday data.
- **Section 8 numbers are from the original pipeline** and are not comparable with the upgraded report.

---

## 12. Course/portfolio framing

Lead with the model comparison table, the confusion matrices, and the
limitations. The honest scoping is a strength: it shows the difference between
"this works on this dataset with these caveats" and "this is a trading
system." Then point to Section 9 to show you know what a stronger version
would look like and what practitioners use.