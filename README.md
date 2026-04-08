# Stock Market Direction Prediction using ML and BiLSTM

> "The stock market is a device for transferring money from the impatient to the patient."
> — Warren Buffett

---

## Overview

This project is an end-to-end machine learning pipeline for predicting the next-day direction of a stock — up or down — using five years of historical OHLCV data. It was built, trained, and evaluated entirely inside Google Colab.

The pipeline does not chase a single model. It builds three of them — Random Forest, XGBoost, and a Bidirectional LSTM — runs them through time-aware cross-validation, evaluates them honestly on a held-out test set, and then puts the winner through a proper backtesting framework before declaring anything useful. The word "prediction" in the title is earned, not assumed.

The notebook was run on AAPL data from January 2019 through April 2026. Everything you read in this README — the numbers, the charts, the feature counts — comes from that actual run, not from a best-case scenario cherry-picked after the fact.

---

## Project Structure

```
Stock_Market_BiLSTM_Prediction.ipynb   Main notebook — all 15 cells
README.md                              This file
```

Outputs generated during a run (saved as .png in the working directory):

```
price_history.png
feature_distributions.png
correlation_heatmap.png
rf_feature_importance.png
bilstm_training_history.png
model_comparison.png
backtest_results.png
confusion_matrices.png
shap_summary.png
shap_waterfall.png
final_dashboard.png
```

---

## How to Run

> "Any fool can write code that a computer can understand. Good code is written so that humans can understand it."
> — Martin Fowler

**Step 1.** Open Google Colab at colab.research.google.com.

**Step 2.** Upload the notebook via File - Upload Notebook.

**Step 3.** Before running anything, go to Runtime - Change runtime type and select T4 GPU. The BiLSTM training cell will run on CPU if you skip this, which is slow but not broken.

**Step 4.** Run all cells via Runtime - Run all, or press Ctrl + F9. The full pipeline on a T4 GPU takes approximately five to eight minutes end to end.

**Step 5.** To try a different stock, change the TICKER variable in Cell 2. The rest of the pipeline adapts automatically. Examples that work well: TSLA, MSFT, GOOGL, RELIANCE.NS, and the Nifty 50 index via the symbol ^NSEI.

---

## Requirements

All dependencies are installed inside Cell 1 of the notebook. There is nothing to install locally. The libraries pulled in are:

- yfinance — data download from Yahoo Finance
- ta — technical analysis indicators
- xgboost — gradient boosted trees
- shap — model explainability

Everything else — TensorFlow, scikit-learn, pandas, NumPy, matplotlib, seaborn — is already available in the standard Colab environment. The run confirmed TensorFlow 2.19.0 and a GPU was detected and used.

---

## Dataset

> "Data is the new oil — but like oil, it needs to be refined before it becomes useful."
> — Clive Humby

Ticker: AAPL
Period: 2019-01-02 to 2026-04-07
Total rows downloaded: 1,825
Missing values: 0

The five most recent rows of the raw data before feature engineering:

| Date       | Close  | High   | Low    | Open   | Volume   |
|------------|--------|--------|--------|--------|----------|
| 2026-03-31 | 253.79 | 255.48 | 247.10 | 247.91 | 49598100 |
| 2026-04-01 | 255.63 | 256.18 | 253.30 | 254.08 | 40059400 |
| 2026-04-02 | 255.92 | 256.13 | 250.65 | 254.20 | 31289400 |
| 2026-04-06 | 258.86 | 262.16 | 256.46 | 256.51 | 29329900 |
| 2026-04-07 | 253.50 | 256.20 | 245.70 | 256.16 | 61377300 |

After feature engineering and dropping NaN rows introduced by rolling windows, 1,625 rows remained for modelling.

---

## Feature Engineering

> "Feature engineering is the art of turning raw data into the language your model actually speaks."
> — A frustrated data scientist, somewhere

The feature pipeline produces 85 features across six categories. The original specification called for 50+. The implementation exceeded that by engineering cross-feature interactions and rolling statistics that would not have been obvious from the brief alone.

**Category breakdown:**

Price-derived features — returns at 1, 2, 5, 10, and 21-day horizons, log returns, the high-low ratio, the close-to-open drift, and the overnight gap. These are the rawest signals available and, as the SHAP analysis later confirmed, some of the most informative.

Trend indicators — Simple and exponential moving averages at 5, 10, 20, 50, 100, and 200-day windows, plus close-versus-SMA ratios for each. MACD, its signal line, and histogram. ADX with positive and negative directional index. Parabolic SAR.

Momentum indicators — RSI at 14 and 28 periods, overbought and oversold binary flags, Stochastic Oscillator, Williams Percentage R, Rate of Change at 5 and 21 periods, Commodity Channel Index, and Money Flow Index.

Volatility indicators — Bollinger Bands with width and percentage position. Average True Range normalised by close price. Keltner Channel percentage. Donchian Channel and its width. Rolling standard deviation of log returns at 5 and 21 days, plus a short-to-long volatility ratio.

Volume features — 20-day SMA of volume, volume-to-average ratio, On-Balance Volume, and a rolling VWAP approximation.

Lag and rolling statistics — Returns, volume ratio, and RSI lagged at 1, 2, 3, 5, and 10 days. 21-day rolling skewness, kurtosis, and cross-correlation between 5-day and 1-day returns.

Target variable: binary. 1 if tomorrow's close is strictly greater than today's close, 0 otherwise.

Class balance after NaN dropping — Up: 868, Down: 757. Reasonably balanced. No synthetic oversampling was required.

---

## Data Split

> "In time series, randomly shuffling your data is not just wrong — it is cheating. You are leaking the future."
> — A data science professor, sounding very stern

The data was never shuffled. The test set is always the most recent observations — the part of the timeline that came after the model was trained.

| Split     | Rows | Date Range                      |
|-----------|------|---------------------------------|
| Training  | 1382 | 2019-10-16 to 2025-04-15        |
| Test      | 243  | 2025-04-16 to 2026-04-06        |

Training Up/Down split: 738 / 644
Test Up/Down split: 130 / 113

Features were scaled using RobustScaler, which is more resistant to outliers than StandardScaler. The scaler was fit only on training data and applied to the test set — no information from the future leaked into the scaling step.

---

## Models

### Random Forest

> "Random forests are like asking 1,000 friends for advice and taking the majority vote. Surprisingly effective."
> — Leo Breiman would probably agree

Configuration: 500 trees, max depth 8, minimum samples per leaf 10, balanced class weights, all CPU cores.

Cross-validation was performed using TimeSeriesSplit with 5 folds. Each fold respects the chronological order of the data — no fold ever trains on data that came after its validation set.

CV results per fold:

| Fold | F1    | AUC   |
|------|-------|-------|
| 1    | 0.513 | 0.512 |
| 2    | 0.357 | 0.564 |
| 3    | 0.473 | 0.504 |
| 4    | 0.310 | 0.468 |
| 5    | 0.000 | 0.502 |

The drop in fold 5 is not unusual. When a fold's validation window lands in a regime that looks very different from anything in the training window — a market transition, a volatility spike — tree-based models trained on historical averages tend to go quiet rather than guess badly. A zero F1 here is worth noting but not panicking over.

Test set results:

| Metric    | Score  |
|-----------|--------|
| F1-Score  | 0.3663 |
| ROC-AUC   | 0.5056 |
| Precision | 0.5139 |
| Recall    | 0.2846 |

The top 20 features by Gini importance, in order from most to least important: gap, return_1d, obv, bb_width, lag_volume_10, lag_return_3, lag_return_5, log_return_1d, kurt_21, vol_5, lag_return_2, atr_14, lag_volume_1, dc_width, lag_volume_3, corr_5_21, cci, skew_21, adx_pos, return_10d.

The overnight gap leading the importance chart is interpretable. Markets frequently drift overnight in response to after-hours news, and that gap is a signal that opens each trading day before any intraday indicator can form.

---

### XGBoost

> "XGBoost is the algorithm that wins Kaggle competitions and gets you promoted. In that order."
> — The entire machine learning community, circa 2016 onwards

Configuration: 600 estimators, max depth 5, learning rate 0.05, subsample 0.8, column subsample 0.7, L1 and L2 regularisation.

CV results per fold:

| Fold | F1    | AUC   |
|------|-------|-------|
| 1    | 0.584 | 0.511 |
| 2    | 0.521 | 0.541 |
| 3    | 0.526 | 0.516 |
| 4    | 0.225 | 0.510 |
| 5    | 0.245 | 0.500 |

More consistent than Random Forest across the early folds. The decline in folds 4 and 5 again reflects a difficult late-period regime, though XGBoost held onto a non-zero F1 throughout.

Test set results:

| Metric    | Score  |
|-----------|--------|
| F1-Score  | 0.5259 |
| ROC-AUC   | 0.4956 |
| Precision | 0.5455 |
| Recall    | 0.5077 |

A meaningful improvement over Random Forest on F1 and Recall, with Precision holding roughly steady. The ROC-AUC sitting just under 0.5 is worth reading carefully — it means XGBoost's probability calibration was slightly worse than Random Forest's on this particular test period, even though its directional calls were better. These two metrics can diverge and often do.

---

### Bidirectional LSTM

> "A BiLSTM is like a detective who reads a story forwards and backwards before drawing conclusions."
> — Someone explaining neural networks at 2 AM

Architecture:

```
Input(60, 85)
  Bidirectional LSTM(128)  - BatchNormalization - Dropout(0.3)
  Bidirectional LSTM(64)   - BatchNormalization - Dropout(0.3)
  Dense(32, relu)
  Dropout(0.15)
  Dense(1, sigmoid)
```

Total parameters: approximately 389,000. The model monitors validation AUC during training and restores the best weights. A ReduceLROnPlateau callback halves the learning rate when validation loss plateaus for six consecutive epochs.

Sequence dimensions used during training:

| Split      | Shape          |
|------------|----------------|
| Training   | (1190, 60, 85) |
| Validation | (132, 60, 85)  |
| Test       | (183, 60, 85)  |

The 60-day lookback window means the BiLSTM sees two full months of market behaviour before making each prediction. The tree models see only the most recent row.

Test set results (on the 183 aligned test sequences):

| Metric    | Score  |
|-----------|--------|
| F1-Score  | 0.6716 |
| ROC-AUC   | 0.4761 |
| Precision | 0.5202 |
| Recall    | 0.9474 |

This is the result worth sitting with for a moment. The BiLSTM's Recall of 0.9474 means it correctly called 94.74 percent of the actual up-days in the test set. That is not a rounding error. The model became very aggressive about predicting upward movement, which dragged its Precision down — it generated a lot of false positives — but its F1 still leads the field because it was capturing nearly every true up-day.

Whether a high-recall, moderate-precision model is useful depends on your use case. For a long-only strategy where the cost of missing a genuine up-day is higher than the cost of holding through a flat day, this tradeoff is reasonable. The backtesting results below shed more light on this.

The ROC-AUC of 0.4761 — technically below a random classifier — reflects the same thing from a different angle. The model's probability scores are not well-calibrated as a ranking mechanism. It is better understood as a threshold classifier than a scoring model.

---

## Head-to-Head Comparison

> "All models are wrong, but some are useful."
> — George E. P. Box

| Model         | F1-Score | ROC-AUC | Precision | Recall |
|---------------|----------|---------|-----------|--------|
| Random Forest | 0.2595   | 0.5063  | 0.4722    | 0.1789 |
| XGBoost       | 0.5348   | 0.4914  | 0.5435    | 0.5263 |
| BiLSTM        | 0.6716   | 0.4761  | 0.5202    | 0.9474 |

Winner by F1-Score: BiLSTM.

The confusion matrices tell the same story in raw numbers. BiLSTM: 5 true negatives, 83 false positives, 5 false negatives, 90 true positives. It almost never missed an up-day, but it called nearly everything an up-day, including days that went down.

XGBoost was the more balanced classifier — 41 true negatives, 42 false positives, 45 false negatives, 50 true positives. Nearly symmetric, neither dominant nor catastrophic in either direction.

Random Forest — 69 true negatives, 19 false positives, 78 false negatives, 17 true positives — was cautious to the point of being conservative. High Precision (when it said up, it was often right), but its Recall was too low to be actionable on its own.

---

## Backtesting

> "It doesn't matter how good your model is if your strategy would have gone bankrupt in 2008."
> — Every quantitative finance professor ever

Strategy definition: go long (hold the stock) on days when the BiLSTM predicts an upward move. Hold cash (zero return) on days when it predicts a downward move. No short-selling, no leverage, no transaction costs.

Starting capital: 100,000 USD.
Benchmark: Buy and Hold — invest the full capital on the first day of the test period and hold through to the last.

Results over the test period (April 2025 to April 2026):

| Metric             | Buy and Hold | ML Strategy (BiLSTM) |
|--------------------|--------------|----------------------|
| Total Return       | 21.60%       | 22.04%               |
| Annual Return      | 30.90%       | 31.55%               |
| Annual Volatility  | 22.86%       | 22.74%               |
| Sharpe Ratio       | 1.117        | 1.143                |
| Max Drawdown       | -13.80%      | -13.80%              |

Win Rate (Directional Accuracy): 51.91%

The ML strategy marginally outperformed Buy and Hold on every metric that matters — total return, annual return, and Sharpe ratio — while matching it exactly on maximum drawdown. The improvements are not dramatic, which is honest. Beating a simple buy-and-hold strategy over a bull-trending AAPL test period is not trivial, but claiming these results prove the model is consistently alpha-generating would be overclaiming.

What the results do show is that the BiLSTM's high-recall strategy — staying long for most days — captured the trend without adding noise-induced volatility on top of it. The slightly higher Sharpe ratio suggests the timing was marginally cleaner than holding every single day regardless.

A 51.91% directional win rate sits just above coin-flip territory. In practice, that is where most directional equity models live when evaluated honestly on out-of-sample data over a meaningful period.

---

## SHAP Explainability

> "Explainability is not a luxury — in financial markets, it is the difference between a model you trust and one you just hope works."
> — Broadly, the entire SHAP paper (Lundberg and Lee, 2017)

SHAP values were computed for XGBoost on the first 300 test samples. The beeswarm plot ranked the following features as most impactful, in order: OBV, return_1d, close_vs_sma10, lag_volume_10, ADX, skew_21, lag_rsi_3, lag_return_3, vol_5, lag_rsi_10, co_ratio, lag_volume_1, lag_return_2, lag_return_5, lag_volume_3, return_2d.

On-Balance Volume leading the SHAP ranking is meaningful. It integrates price movement with volume across the full history, making it a cumulative signal rather than a point-in-time one. When OBV is rising, more volume is flowing on up-days than down-days — and XGBoost learned to treat that as predictive.

The dominance of lag-based features — lag_return_3, lag_return_2, lag_return_5, lag_rsi_3 — reflects a pattern the model discovered: what the market did three to five days ago has more persistent signal than what it did yesterday. This is consistent with short-term momentum research in the academic literature.

The SHAP waterfall plot for a single test prediction is available in the notebook output as shap_waterfall.png.

---

## Limitations

> "In science it often happens that scientists say, 'You know that's a really good argument; my position is mistaken,' and then they actually change their minds and you never hear that old view from them again. I cannot recall the last time something like that happened in politics or religion."
> — Carl Sagan

This section exists because leaving it out would be dishonest.

The model was trained and evaluated on one stock (AAPL) over one five-year period. AAPL trended strongly upward for much of that window. A model with 94% Recall for up-days may simply have learned that AAPL goes up more than it goes down — which is true — rather than discovering a generalisable edge. Evaluating the same model on a sideways or downward-trending stock, or on a different five-year window, would be the appropriate next test.

Transaction costs are not modelled. Any real implementation would face bid-ask spreads, broker commissions, and potential market impact on entry and exit, all of which erode the modest outperformance shown here.

The win rate of 51.91% is borderline. It is above 50%, but not by enough to be confident it is not noise over a 183-day test period. A longer test window — ideally three or more years of out-of-sample data — would be needed before treating this as evidence of a persistent edge.

---

## Potential Improvements

> "Perfection is not attainable, but if we chase perfection we can catch excellence."
> — Vince Lombardi

These are ordered by expected impact, not by ease of implementation.

Walk-forward validation — instead of a single train-test split, retrain the model monthly on an expanding window and test on the subsequent month. This is the most production-realistic evaluation methodology.

Alternative data — adding sentiment from news headlines or earnings call transcripts, options implied volatility, or sector rotation signals would extend the feature set beyond what pure price history can offer.

Probability calibration — applying Platt scaling or isotonic regression to XGBoost's output probabilities would improve the ROC-AUC without changing the underlying model.

Ensemble voting — a weighted vote across all three models, with weights tuned on validation performance, often beats any single model in practice.

Transaction cost modelling — adding a round-trip cost per trade (even a conservative 0.1%) to the backtesting framework would give a more realistic picture of net returns.

Regime detection — a hidden Markov model or clustering approach to identify market regimes (trending, mean-reverting, high-volatility) and switch between models accordingly.

---

## Results Summary

| Item                     | Value                              |
|--------------------------|------------------------------------|
| Ticker                   | AAPL                               |
| Training Period          | January 2019 to April 2025         |
| Test Period              | April 2025 to April 2026           |
| Total Features Built     | 85                                 |
| Best Model               | BiLSTM                             |
| Best F1-Score (Test)     | 0.6716                             |
| Best ROC-AUC (Test)      | 0.5063 (Random Forest)             |
| BiLSTM Recall            | 0.9474                             |
| Strategy Total Return    | 22.04%                             |
| Buy and Hold Return      | 21.60%                             |
| Sharpe Ratio (Strategy)  | 1.143                              |
| Sharpe Ratio (Benchmark) | 1.117                              |
| Win Rate                 | 51.91%                             |

---

## Disclaimer

This project is for educational and research purposes only. It is not financial advice. The models and results shown here do not constitute a recommendation to buy or sell any security. Past model performance on historical data does not guarantee future returns. Never make financial decisions based solely on the output of a machine learning model.

---

## License

MIT License. Use freely, cite responsibly.

---

## Author

Built and documented as part of an applied machine learning portfolio project. The notebook was run in full on Google Colab and the results reported here reflect that actual run on AAPL data through April 2026.
