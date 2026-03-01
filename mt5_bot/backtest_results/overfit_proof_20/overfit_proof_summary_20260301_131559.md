# Overfitting / Luck Proof Summary (20 Historical Backtests per Symbol)

Generated: 2026-03-01T13:15:59.786873+00:00

## Key Finding
Top candidate is **NAS100** with 14/20 profitable windows (70.0%), median return 3.2287%, median PF 1.0199.

## Statistical note
Binomial p-value tests whether the number of profitable windows could be random under 50/50 luck. Thresholds: <0.05 strong evidence, <0.10 moderate evidence.

## Ranking

| symbol | runs | profitable_runs | consistency_pct | median_return_pct | median_pf | binomial_p_value | anti_luck | overfit_risk | verdict | source_file |
|---|---|---|---|---|---|---|---|---|---|---|
| NAS100 | 20 | 14 | 70.0 | 3.2287 | 1.0199 | 0.057659 | insufficient_statistical_evidence | lower_risk | needs_improvement | overfit_proof_20_NAS100_bars60000_20260301_131506.json |
| XAUUSD | 20 | 7 | 35.0 | -5.2306 | 0.976 | 0.942341 | insufficient_statistical_evidence | lower_risk | needs_improvement | overfit_proof_20_XAUUSD_bars60000_20260301_131435.json |
| BTCUSD | 20 | 1 | 5.0 | -29.089 | 0.8463 | 0.999999 | insufficient_statistical_evidence | higher_risk | needs_improvement | overfit_proof_20_BTCUSD_bars60000_20260301_131456.json |
| GBPUSD | 20 | 0 | 0.0 | -28.5182 | 0.8061 | 1.0 | insufficient_statistical_evidence | higher_risk | needs_improvement | overfit_proof_20_GBPUSD_bars60000_20260301_131416.json |
| USDJPY | 20 | 0 | 0.0 | -50.6963 | 0.6734 | 1.0 | insufficient_statistical_evidence | higher_risk | needs_improvement | overfit_proof_20_USDJPY_bars60000_20260301_131425.json |
| EURUSD | 20 | 0 | 0.0 | -61.4665 | 0.5252 | 1.0 | insufficient_statistical_evidence | higher_risk | needs_improvement | overfit_proof_20_EURUSD_bars60000_20260301_131406.json |
| GBPJPY | 20 | 0 | 0.0 | -72.8787 | 0.4214 | 1.0 | insufficient_statistical_evidence | higher_risk | needs_improvement | overfit_proof_20_GBPJPY_bars60000_20260301_131445.json |