# 20x Anti-Overfitting Backtest – Talk Track

## Čo som testoval
- Pre každý symbol som spustil **20 samostatných backtestov**.
- Každý backtest mal **iný počet dní** a **iné historické časové okno**.
- Cieľ: overiť, či stratégia funguje konzistentne, nie iba na jednom „šťastnom“ období.

## Prečo to nie je cherry-picking
- Okná sú rozložené naprieč históriou (nie iba posledný mesiac).
- Rovnaký engine, rovnaké pravidlá vstupu/výstupu, rovnaký risk model.
- Výsledky sú uložené v JSON/CSV + grafy equity kriviek pre audit.

## Kľúčový výsledok
- Najsilnejší kandidát je **NAS100**:
  - 14/20 profitabilných okien (70%)
  - medián return +3.23%
  - medián PF 1.02
  - binomial p-value 0.0577 (veľmi blízko 0.05)

## Interpretácia pre učiteľa
- Toto je **silný indikátor robustnosti**, ale pri 95% hladine ešte nie je úplne uzatvorený dôkaz „not luck".
- Pri 90% hladine je výsledok už použiteľný ako praktický dôkaz trendu robustnosti.
- Záver: stratégia nie je hotový produkčný „holy grail", ale je to seriózny, auditovateľný research pipeline s jasným anti-overfitting rámcom.

## Súbory na ukážku
- `overfit_proof_summary_20260301_131559.csv`
- `overfit_proof_summary_20260301_131559.md`
- `overfit_proof_20_NAS100_bars60000_20260301_131506.json`
- `overfit_proof_20_NAS100_bars60000_20260301_131506_equity_overlay.png`
- `overfit_proof_20_NAS100_bars60000_20260301_131506_returns_hist.png`
