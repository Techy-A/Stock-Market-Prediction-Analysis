# Stock Market Direction Prediction with ML and BiLSTM

![Python](https://img.shields.io/badge/Python-3-3776AB?style=flat-square&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=flat-square&logo=tensorflow&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-2c5364?style=flat-square)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat-square&logo=scikitlearn&logoColor=white)
![Colab](https://img.shields.io/badge/Google%20Colab-F9AB00?style=flat-square&logo=googlecolab&logoColor=white)

Predicts whether a stock will close **up or down tomorrow**. The notebook builds 71 technical features from daily prices, trains Random Forest, XGBoost and a Bidirectional LSTM, and then asks the question most stock-prediction notebooks skip: **does any model beat a trivial rule, after trading costs, by more than luck would?**

## Version 2: what changed and why

Version 1 reported the BiLSTM as the winner with an F1 of 0.672 and a backtest that beat buy-and-hold. A closer look showed that this result came from the evaluation, not the model.

**The finding.** On the 183 test days, the BiLSTM predicted "up" on 94.5% of them. Its accuracy (51.9%) was exactly the share of up-days, and its balanced accuracy was 0.502, a coin flip. A rule that says "up" every day scores a **higher** F1 (0.683) than the BiLSTM did. Because the strategy was invested almost every day, its backtest was buy-and-hold under another name.

**The proof.** Running the version 1 notebook on a pure random walk, where tomorrow cannot be predicted at all, gives the same picture: the BiLSTM "wins" with F1 0.68 by predicting "up" every day, and its backtest is identical to buy-and-hold.

Version 2 fixes the evaluation so this cannot happen silently:

| Problem in version 1 | Fix in version 2 |
|:--|:--|
| F1 on the "up" class picks the model that says "up" most often | Balanced accuracy, MCC and ROC-AUC reported next to F1, plus two baselines: **always up** and **same as today** |
| The winner was chosen by its **test** score, then backtested on the same test data | The winner is chosen on a separate **validation** block; the test block is used once |
| Price-level features (moving averages, band edges, ATR in dollars, cumulative OBV) drift outside the training range | All 71 features are scale-free: returns, ratios to moving averages, oscillators, normalised widths |
| The next-day BiLSTM signal used the window ending at the **end of training**, a year old | It uses the most recent 60 days |
| The latest trading day was dropped, so "tomorrow" was a day already known | The latest day is kept for the live signal and only excluded from training |
| BiLSTM windows ended the day **before** the prediction day | Windows include the prediction day's own features |
| The BiLSTM skipped the first 60 test days, so models were compared on different rows | Windows borrow earlier features, so all models predict exactly the same test days |
| Early stopping and model choice used the same validation data | Early stopping uses its own slice of the training block |
| Fixed 0.5 cutoff, which made models with shifted probabilities predict one class only | Each model's cutoff is the median of its own validation probabilities (no labels used) |
| No trading costs; the "AUC" in the confusion-matrix titles was computed from hard labels | 5 bps per trade, exposure and trade count reported; titles show balanced accuracy |
| No way to tell skill from luck | Binomial test and a 2,000-shuffle permutation test of the strategy's Sharpe ratio |
| One train/test split | Walk-forward evaluation that retrains the tree models every month |
| No seed for TensorFlow, so runs were not reproducible | One seed for Python, NumPy and TensorFlow |

## How the pipeline was verified

The new notebook has a switch, `SANITY_CHECK = True`, that replaces real prices with simulated ones. It was run end to end three ways:

| Simulated market | What should happen | What happened |
|:--|:--|:--|
| **Random walk** (nothing is predictable), 3 different seeds | No model beats the baselines; tests say "luck" | Best model's balanced accuracy 0.51 to 0.53; binomial p = 0.20 to 0.36, permutation p = 0.06 to 0.28: no false alarm |
| **Weak momentum** (each day's return carries 20% into the next) | A small edge, too small to prove on one year | Every model slightly above a coin flip (MCC +0.02 to +0.09); tests not significant |
| **Strong momentum** (40% carry-over) | A clear edge that the tests detect | Balanced accuracy up to 0.59; binomial p = 0.002, permutation p = 0.001 |

So the pipeline does not invent an edge where there is none, and it does find one where there is.

## Results on real data

Open the notebook in Colab and run it to produce the real-data results for any ticker. The cells to read are:

1. **Cell 10**: models next to the baselines. The question is whether any model beats *always up* on balanced accuracy, MCC and ROC-AUC.
2. **Cell 11**: the backtest after costs, with exposure. A strategy invested 95% of the time is buy-and-hold.
3. **Cell 12**: the significance tests. A p-value above 0.05 means the timing is not distinguishable from luck.

Daily direction for a large-cap stock is close to unpredictable from price history alone, so the expected honest outcome is a balanced accuracy near 0.5. Showing that clearly, with the tooling to prove it, is the point of the project.

## How to run

1. Open the notebook in [Google Colab](https://colab.research.google.com) (File, then Upload notebook).
2. Set the runtime to a **T4 GPU** (Runtime, then Change runtime type). The BiLSTM also trains on a CPU, only more slowly.
3. Run all cells. The full pipeline, including the permutation and walk-forward tests, takes about 5 minutes on a T4.
4. Change `TICKER` in cell 2 to try another stock, for example `TSLA`, `MSFT`, `RELIANCE.NS` or `^NSEI`.
5. Set `SANITY_CHECK = True` to rerun everything on a random walk and confirm that no edge appears.

## Pipeline

| Cell | Step |
|:--|:--|
| 1 to 3 | Imports, configuration, data download (or a simulated random walk) |
| 4 | 71 scale-free features in six groups: returns, trend, momentum, volatility, volume, lags |
| 5 | Feature distributions on up vs down days, correlations |
| 6 | Chronological train, validation and test split; metrics and baselines |
| 7 to 9 | Random Forest, XGBoost and BiLSTM, each with time-series cross-validation |
| 10 | Model choice on validation; test comparison against the baselines; ROC curves |
| 11 | Backtest with costs, exposure and trade counts |
| 12 | Binomial and permutation tests |
| 13 | Walk-forward retraining of the tree models |
| 14 to 15 | Confusion matrices and SHAP explanations |
| 16 | Next-day signal from the latest data |
| 17 | Summary dashboard |

## Features (71)

| Group | Features |
|:--|:--|
| Returns and shape | Returns over 1, 2, 5, 10 and 21 days, log return, high-low range, open-to-close drift, overnight gap |
| Trend | Close vs SMA and EMA at 5 to 200 days, SMA crossovers, MACD scaled by price, ADX with +DI and -DI, close vs Parabolic SAR |
| Momentum | RSI 14 and 28 with overbought and oversold flags, Stochastic, Williams %R, ROC 5 and 21, CCI, MFI |
| Volatility | Bollinger width and %B, ATR as % of price, Keltner %, Donchian position and width, rolling volatility at 5 and 21 days and their ratio |
| Volume | Volume vs its 20-day average, 21-day On-Balance Volume flow, close vs 20-day VWAP |
| Lags and statistics | Lags of return, volume ratio and RSI at 1 to 10 days, 21-day skew and kurtosis, return cross-correlation |

## Limitations

- **Price history alone carries very little information** about tomorrow's direction for a liquid stock. Better inputs (news, earnings, options data) matter more than a bigger model.
- **One year of test data is short.** The tests can only detect a fairly large edge; a small real one would need several years of out-of-sample data.
- **Closing-auction execution is assumed.** The signal uses day *t*'s close and trades at that same close, which is optimistic for anything but closing-auction orders.

## Disclaimer

This project is for education and research only and is not financial advice.

## License

MIT

---

<sub>Built by [Aryan Awasthi](https://github.com/Techy-A). ENC @ Thapar Institute of Engineering and Technology.</sub>
