# Hybrid GARCH + Temporal Fusion Transformer for S&P 500 Volatility Forecasting

> A reproducible research repository benchmarking classical econometric volatility
> models (GARCH, EGARCH, GJR-GARCH, HAR-RV) against a **hybrid Temporal Fusion
> Transformer (TFT)** that consumes GARCH-derived features, and evaluating every
> model with both statistical loss functions **and** a real volatility-targeting
> trading strategy on SPY.

---

## 1. Motivation

Volatility is the single most important input to option pricing, risk management,
and systematic trading. Two research traditions coexist:

1. **Econometric models** (ARCH/GARCH family, HAR-RV) — interpretable, well-grounded
   in the stylized facts of financial returns (volatility clustering, fat tails,
   leverage effect), but limited by rigid parametric forms.
2. **Deep sequence models** — Transformers, LSTMs, and in particular the
   **Temporal Fusion Transformer** (Lim et al., 2019) — flexible, able to mix
   static, known-future, and observed covariates, but data-hungry and prone to
   overfitting on financial time series.

This project argues that the two should be **composed, not competed**. We feed
the GARCH likelihood-optimized conditional variance and its standardized residuals
into the TFT as covariates; the transformer is therefore free to model only what
GARCH misses (regime breaks, cross-feature interactions, non-linear dependencies
on VIX and realized measures).

## 2. Data

* **Symbol:** `^GSPC` (S&P 500 index) and `SPY` (ETF, for trading simulation).
* **Exogenous:** `^VIX` (implied volatility, sentiment proxy).
* **Range:** 2000-01-01 through most recent close.
* **Volatility proxies:**
  * Close-to-close log-return squared (realized variance, `rv_cc`).
  * Garman–Klass OHLC estimator (`rv_gk`), ~7.4× more efficient than close-to-close.
* **Splits (walk-forward):**
  * Train: 2000 – 2017
  * Validation: 2018 – 2019
  * Out-of-sample test: 2020 – present *(includes COVID crash for robustness)*.

## 3. Models

| Family | Model | File |
| --- | --- | --- |
| Benchmark | Naïve (last observed RV) | `src/evaluation.py` |
| Classical | GARCH(1,1) | `src/models/garch_models.py` |
| Classical | EGARCH(1,1) | `src/models/garch_models.py` |
| Classical | GJR-GARCH(1,1,1) | `src/models/garch_models.py` |
| Classical | HAR-RV (daily / weekly / monthly lags) | `src/models/har_model.py` |
| Deep | Temporal Fusion Transformer | `src/models/tft_model.py` |
| **Hybrid** | **TFT + GARCH features** | `src/models/tft_model.py` |

## 4. Evaluation

### Statistical
* **RMSE / MAE** on `log(σ²)` and on `σ` (variance and volatility scale).
* **QLIKE** — the standard proper loss for volatility:
  `QLIKE = σ̂²/σ² − log(σ̂²/σ²) − 1`.
* **Diebold–Mariano** test for equal predictive accuracy.

### Risk
* **VaR** at 95% / 99% assuming conditional normality and Student-t innovations.
* **Expected Shortfall** at the same levels.
* **Kupiec unconditional coverage** and **Christoffersen conditional coverage**
  backtests on VaR breaches.

### Trading
* **Volatility targeting**: position size `w_t = σ_target / σ̂_t`, capped at
  `w_max = 2.0`.
* Compared against **buy-and-hold SPY** and an **equal-weight risk-parity**
  baseline, using `vectorbt` for fast portfolio simulation.
* Reported: CAGR, annualized Sharpe, Sortino, Calmar, max drawdown, turnover.

## 5. Key Results (illustrative — run the pipeline to reproduce)

| Model | RMSE (σ) | QLIKE | DM vs. GARCH | Sharpe | Max DD |
| --- | --- | --- | --- | --- | --- |
| Buy & Hold SPY | — | — | — | 0.58 | −33.7% |
| GARCH(1,1) | 0.0082 | 0.241 | — | 0.71 | −24.1% |
| EGARCH(1,1) | 0.0079 | 0.227 | −1.8* | 0.76 | −22.8% |
| GJR-GARCH | 0.0078 | 0.223 | −2.1* | 0.78 | −22.0% |
| HAR-RV | 0.0074 | 0.210 | −3.5** | 0.83 | −19.9% |
| TFT (vanilla) | 0.0076 | 0.215 | −2.9** | 0.81 | −21.1% |
| **Hybrid GARCH-TFT** | **0.0070** | **0.198** | −4.7*** | **0.94** | **−17.3%** |

\* significant at 10%, ** at 5%, \*** at 1%.

## 6. Reproducing

```bash
# 1. Environment (Python 3.10 recommended)
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Run the full pipeline
python scripts/01_download_data.py
python scripts/02_diagnostics.py
python scripts/03_fit_classical.py
python scripts/04_train_tft.py          # requires CUDA-capable GPU for speed
python scripts/05_evaluate_and_backtest.py

# 3. Or open the end-to-end notebook
jupyter lab notebooks/00_full_pipeline.ipynb
```

All RNG seeds are fixed in `config.yaml` via `src/utils.set_global_seed`.

## 7. Limitations & Future Work

* **Regime shifts.** The walk-forward scheme partially addresses this, but a
  *Markov-switching* GARCH or a regime-aware prior on the TFT would be a
  cleaner solution.
* **Intraday data.** Realized-kernel estimators on 5-minute bars dominate
  close-to-close proxies; a natural extension is HAR-RV-CJ with jump
  decomposition from TAQ data.
* **Cross-sectional signal.** Only SPY is modeled here. A multi-asset TFT with
  a shared encoder could exploit volatility spillovers across sectors or
  currencies.
* **Foundation models.** `TimesFM`, `Moirai`, and `Chronos` are obvious
  candidates to drop into `models/tft_model.py` via the same feature pipeline.
* **Transaction costs.** The backtest charges 1 bp per trade; a realistic
  execution model (market impact, borrow cost for short vol) is left for future
  work.

## 8. References

1. Bollerslev, T. (1986). *Generalized autoregressive conditional
   heteroskedasticity*. Journal of Econometrics.
2. Nelson, D. B. (1991). *Conditional heteroskedasticity in asset returns: a
   new approach*. Econometrica.
3. Glosten, Jagannathan, Runkle (1993). *On the relation between the expected
   value and the volatility of the nominal excess return on stocks*. JoF.
4. Corsi, F. (2009). *A simple approximate long-memory model of realized
   volatility*. JFEC.
5. Lim, B., Arık, S. Ö., Loeff, N., & Pfister, T. (2019). *Temporal Fusion
   Transformers for Interpretable Multi-horizon Time Series Forecasting*.
6. Kim, H. Y., & Won, C. H. (2018). *Forecasting the volatility of stock price
   index: A hybrid model integrating LSTM with multiple GARCH-type models*.
   Expert Systems with Applications.
7. Liu, Y. (2019). *Novel volatility forecasting using deep learning – Long
   Short Term Memory Recurrent Neural Networks*. ESwA.
8. Patton, A. J. (2011). *Volatility forecast comparison using imperfect
   volatility proxies*. Journal of Econometrics.
9. Christoffersen, P. F. (1998). *Evaluating interval forecasts*.
   International Economic Review.

## License

MIT — see `LICENSE`.
