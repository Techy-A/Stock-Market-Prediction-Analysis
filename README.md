# Stock Market Direction Prediction with ML and BiLSTM

![Python](https://img.shields.io/badge/Python-3-3776AB?style=flat-square&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.19-FF6F00?style=flat-square&logo=tensorflow&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-2c5364?style=flat-square)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat-square&logo=scikitlearn&logoColor=white)
![Colab](https://img.shields.io/badge/Google%20Colab-F9AB00?style=flat-square&logo=googlecolab&logoColor=white)

An end-to-end pipeline that predicts whether a stock will close **up or down tomorrow**. It builds 85 technical features from five years of daily prices, compares Random Forest, XGBoost and a Bidirectional LSTM with time-aware validation, and backtests the winner against buy-and-hold.

Every number below comes from one full run on **AAPL, January 2019 to April 2026**.

## Results at a glance

| | |
|:--|:--|
| Features engineered | **85** (the brief asked for 50+) |
| Best model (test F1) | **BiLSTM, F1 0.672**, recall 0.947 |
| Strategy return vs buy-and-hold | **22.04% vs 21.60%** |
| Sharpe ratio, strategy vs benchmark | **1.143 vs 1.117** |
| Max drawdown, strategy vs benchmark | -13.80% vs -13.80% |
| Directional win rate | 51.91% |

The edge over buy-and-hold is small, and the limitations section explains why it should not be overclaimed.

## How to run

1. Open the notebook in [Google Colab](https://colab.research.google.com) (File, then Upload notebook).
2. Set the runtime to a **T4 GPU** (Runtime, then Change runtime type). The BiLSTM also trains on CPU, only slower.
3. Run all cells (Ctrl + F9). The full pipeline takes about 5 to 8 minutes on a T4.
4. To try another stock, change `TICKER` in cell 2, for example `TSLA`, `MSFT`, `RELIANCE.NS` or the Nifty 50 index `^NSEI`.

Cell 1 installs everything extra (`yfinance`, `ta`, `xgboost`, `shap`); the rest ships with Colab.

## Data

| | |
|:--|:--|
| Ticker | AAPL, from Yahoo Finance |
| Period | 2019-01-02 to 2026-04-07 |
| Rows | 1,825 downloaded, 1,625 after the rolling-window warm-up |
| Target | 1 if tomorrow's close is above today's, else 0 |
| Class balance | 868 up, 757 down (no resampling needed) |

## Feature engineering (85 features)

| Group | Features |
|:--|:--|
| Price | Returns over 1, 2, 5, 10 and 21 days, log returns, high-low ratio, close-to-open drift, overnight gap |
| Trend | SMA and EMA at 5 to 200 days, close-vs-SMA ratios, MACD with signal and histogram, ADX with +DI and -DI, Parabolic SAR |
| Momentum | RSI 14 and 28 with overbought and oversold flags, Stochastic, Williams %R, ROC 5 and 21, CCI, MFI |
| Volatility | Bollinger width and %B, ATR / close, Keltner %, Donchian channel and width, rolling volatility at 5 and 21 days and their ratio |
| Volume | 20-day volume SMA, volume ratio, On-Balance Volume, rolling VWAP |
| Lags and statistics | Lags of return, volume ratio and RSI at 1, 2, 3, 5 and 10 days, 21-day skew and kurtosis, return cross-correlation |

## Validation design

The data is never shuffled: the test set is always the most recent year.

| Split | Rows | Dates |
|:--|:--|:--|
| Train | 1,382 | 2019-10-16 to 2025-04-15 |
| Test | 243 | 2025-04-16 to 2026-04-06 |

- `RobustScaler` is fitted on the training data only, so nothing leaks from the future.
- Tree models use 5-fold `TimeSeriesSplit`, where every validation fold comes after its training data.
- The BiLSTM uses 60-day sequences and keeps the weights with the best validation AUC.

## Models

**Random Forest:** 500 trees, max depth 8, at least 10 samples per leaf, balanced class weights.

**XGBoost:** 600 trees, max depth 5, learning rate 0.05, row and column subsampling, L1 and L2 regularisation.

**Bidirectional LSTM:** about 389,000 parameters.

```
Input (60 days x 85 features)
BiLSTM(128) -> BatchNorm -> Dropout(0.3)
BiLSTM(64)  -> BatchNorm -> Dropout(0.3)
Dense(32, relu) -> Dropout(0.15) -> Dense(1, sigmoid)
```

## Head-to-head on the test period

Scored on the 183 test days that all three models can predict (the BiLSTM needs 60 days of history first).

| Model | F1 | ROC-AUC | Precision | Recall |
|:--|:--|:--|:--|:--|
| Random Forest | 0.260 | 0.506 | 0.472 | 0.179 |
| XGBoost | 0.535 | 0.491 | 0.544 | 0.526 |
| **BiLSTM** | **0.672** | 0.476 | 0.520 | **0.947** |

- **BiLSTM** catches almost every up-day but also calls most down-days "up" (90 TP, 83 FP, 5 FN, 5 TN). It works as a threshold classifier, not as a well-calibrated ranker.
- **XGBoost** is the balanced one (50 TP, 42 FP, 45 FN, 41 TN).
- **Random Forest** is cautious: precise when it says "up", but it misses most up-days (17 TP, 19 FP, 78 FN, 69 TN).

On the full 243-day test set, the tree models score F1 0.366 (Random Forest) and 0.526 (XGBoost).

## Backtest

The strategy is long on days the BiLSTM predicts "up" and in cash otherwise. It uses no shorting, no leverage and no transaction costs, and starts with $100,000.

| Metric | Buy and hold | BiLSTM strategy |
|:--|:--|:--|
| Total return | 21.60% | **22.04%** |
| Annual return | 30.90% | **31.55%** |
| Annual volatility | 22.86% | **22.74%** |
| Sharpe ratio | 1.117 | **1.143** |
| Max drawdown | -13.80% | -13.80% |

## Explainability (SHAP)

SHAP values on XGBoost (first 300 test days) rank **On-Balance Volume** first, followed by the 1-day return, close vs 10-day SMA, 10-day volume lag and ADX. Lagged returns at 2 to 5 days carry more signal than yesterday's return, which is consistent with research on short-term momentum.

## Limitations

- **One stock, one period.** AAPL trended up for most of the window, so a model with 95% recall on up-days may just have learned that AAPL usually rises. It needs testing on sideways and falling markets.
- **No transaction costs.** Spreads and commissions would eat most of the small outperformance.
- **A 51.9% win rate over 183 days** is not enough to rule out noise. A multi-year out-of-sample test is needed.

## Next steps

1. Walk-forward validation with monthly retraining
2. Probability calibration (Platt or isotonic) for XGBoost
3. A weighted ensemble of all three models
4. Transaction costs in the backtest
5. Regime detection to switch models between trending and choppy markets
6. News sentiment and implied volatility as extra features

## Files

| File | Purpose |
|:--|:--|
| `Stock_Market_BiLSTM_Prediction.ipynb` | The full pipeline in 15 cells |

When it runs, the notebook saves its charts as PNG files: price history, feature distributions, correlation heatmap, feature importance, BiLSTM training curves, model comparison, confusion matrices, backtest, SHAP plots and a final dashboard.

## Disclaimer

This project is for education and research only and is not financial advice.

## License

MIT

---

<sub>Built by [Aryan Awasthi](https://github.com/Techy-A). ENC @ Thapar Institute of Engineering and Technology.</sub>
