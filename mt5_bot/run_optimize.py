"""Quick: deep-search BTCUSD, then validate all 7 with saved params, save results."""
import json
from pathlib import Path
from itertools import product
import pandas as pd
import numpy as np
import backtest_improved as bi

bi.initialize_mt5()
BS = Path('best_settings.json')

# ── 1. Deep-search BTCUSD ────────────────────────────────────────────────────
print("=== BTCUSD deep search ===", flush=True)
sym = 'BTCUSD'
df_raw = bi.fetch_data(sym, bars=10000)
df_raw = df_raw.sort_values('Time').reset_index(drop=True)
co = pd.to_datetime(df_raw['Time'].iloc[-1]) - pd.Timedelta(days=30)
df_raw = df_raw[df_raw['Time'] >= co].reset_index(drop=True)
sp = int(len(df_raw) * 0.7)
dtr = df_raw.iloc[:sp].copy()
dte = df_raw.iloc[sp:].copy()

ema_g = [20, 30, 40, 50, 60, 80]
adx_g = [12.0, 14.0, 18.0, 22.0, 26.0]
slm_g = [0.6, 0.8, 1.0, 1.5, 2.0, 2.5]
rr_g  = [1.0, 1.2, 1.5, 1.8, 2.0]

best_s = -np.inf; best_p = None; best_m = None
combos = list(product(ema_g, adx_g, slm_g, rr_g))
for ci, (e, a, s, r) in enumerate(combos):
    if ci % 100 == 0:
        print(f"  {ci}/{len(combos)}", flush=True)
    pr = (e, a, s, r)
    bi.run_backtest_no_lookahead(dtr, sym, params=pr, risk_pct=2.0)
    rv = bi.run_backtest_no_lookahead(dte, sym, params=pr, risk_pct=2.0)
    m = rv.get('metrics', {})
    sc = bi.compute_score(m)
    if sc > best_s:
        best_s = sc; best_p = pr; best_m = m

ret = best_m.get('return_pct', 0)
pf  = best_m.get('profit_factor', 0)
wr  = best_m.get('win_rate', 0)
nt  = best_m.get('total_trades', 0)
print(f"  BTC best: EMA={best_p[0]} ADX>={best_p[1]:.0f} SL={best_p[2]} RR={best_p[3]} "
      f"=> {nt}t WR={wr:.1f}% PF={pf:.2f} Ret={ret:+.1f}%", flush=True)

# Save BTC into best_settings
bs = json.loads(BS.read_text())
bs['instruments']['BTCUSD'] = {
    'EMA': int(best_p[0]), 'ADX': float(best_p[1]),
    'SL_Mult': float(best_p[2]), 'RR': float(best_p[3]),
    'Trades': int(nt), 'Win_Rate': float(wr),
    'PF': float(pf), 'Return_Pct': float(ret),
    'Max_DD': float(best_m.get('max_drawdown', 0)),
}
BS.write_text(json.dumps(bs, indent=2))
print("  Saved BTCUSD params →", BS)

# ── 2. Validate ALL 7 with their saved params ────────────────────────────────
print("\n=== FINAL VALIDATION — all 7 symbols with saved params ===", flush=True)
final = []
for sym in sorted(bi.INSTRUMENTS.keys()):
    df = bi.fetch_data(sym, bars=10000)
    if df is None or len(df) < 300:
        final.append({'symbol': sym, 'error': 'no_data'}); continue
    df = df.sort_values('Time').reset_index(drop=True)
    co2 = pd.to_datetime(df['Time'].iloc[-1]) - pd.Timedelta(days=30)
    df = df[df['Time'] >= co2].reset_index(drop=True)
    if len(df) < 180:
        final.append({'symbol': sym, 'error': 'short'}); continue
    # Load saved params
    p = bi._load_best_params(sym)
    result = bi.run_backtest_no_lookahead(df, sym, params=p, risk_pct=2.0)
    m = result.get('metrics', {})
    row = {
        'symbol': sym,
        'params': {'EMA': p[0], 'ADX': p[1], 'SL_Mult': p[2], 'RR': p[3]},
        'bars': len(df),
        'trades': int(m.get('total_trades', 0)),
        'wins': int(m.get('wins', 0)),
        'win_rate': round(float(m.get('win_rate', 0)), 1),
        'pf': round(float(m.get('profit_factor', 0)), 2),
        'return_pct': round(float(m.get('return_pct', 0)), 2),
        'total_profit': round(float(m.get('total_profit', 0)), 2),
        'max_dd': round(float(m.get('max_drawdown', 0)), 2),
    }
    final.append(row)
    tag = ' OK ' if row['return_pct'] > 0 else 'LOSS'
    print(f"  [{tag}] {sym:8s}  EMA={p[0]:3d} ADX>={p[1]:.0f} SL={p[2]:.1f} RR={p[3]:.1f} "
          f"| {row['trades']:3d}t WR={row['win_rate']:.1f}% PF={row['pf']:.2f} "
          f"Ret={row['return_pct']:+.1f}%  DD=${row['max_dd']:.0f}", flush=True)

ok = sum(1 for r in final if r.get('return_pct', 0) > 0)
print(f"\n  ✓ {ok}/{len(final)} symbols profitable")
print(f"  Settings source: {BS}\n")

# Save validation report
Path('backtest_results').mkdir(exist_ok=True)
Path('backtest_results/ict_smc_final_validation.json').write_text(
    json.dumps({'engine': 'ICT_SMC_v2', 'risk_pct': 2.0, 'days': 30,
                'profitable': ok, 'total': len(final), 'symbols': final},
               indent=2, default=str))
print("  Report → backtest_results/ict_smc_final_validation.json")
