"""Optimize on 90 days, validate on most-recent 30 days."""
import json
from pathlib import Path
from itertools import product
import pandas as pd
import numpy as np
import backtest_improved as bi

bi.initialize_mt5()
BS = Path('best_settings.json')

ema_g = [15, 20, 30, 40, 50, 60, 80]
adx_g = [14.0, 18.0, 22.0, 26.0]
slm_g = [0.6, 1.0, 1.5, 2.0, 2.5]
rr_g  = [1.0, 1.2, 1.5, 2.0]

final = {}
for sym in sorted(bi.INSTRUMENTS.keys()):
    print(f"\n=== {sym} ===", flush=True)
    df = bi.fetch_data(sym, bars=15000)
    if df is None or len(df) < 600:
        print("  SKIP - no data"); continue
    df = df.sort_values('Time').reset_index(drop=True)
    last_ts = pd.to_datetime(df['Time'].iloc[-1])

    # 90-day optimization window (ends 30 days ago — no peaking into validation)
    opt_end   = last_ts - pd.Timedelta(days=30)
    opt_start = last_ts - pd.Timedelta(days=120)  # 90-day opt window
    df_opt = df[(df['Time'] >= opt_start) & (df['Time'] < opt_end)].reset_index(drop=True)

    # 30-day validation window (most recent)
    val_start = last_ts - pd.Timedelta(days=30)
    df_val = df[df['Time'] >= val_start].reset_index(drop=True)

    if len(df_opt) < 400 or len(df_val) < 180:
        print(f"  SKIP - opt={len(df_opt)} val={len(df_val)} bars"); continue
    print(f"  Opt bars: {len(df_opt)}  Val bars: {len(df_val)}", flush=True)

    # Optimize on opt window (70/30 split within it)
    best_s = -np.inf; best_p = None
    sp = int(len(df_opt) * 0.7)
    dtr = df_opt.iloc[:sp].copy(); dte = df_opt.iloc[sp:].copy()
    combos = list(product(ema_g, adx_g, slm_g, rr_g))
    for ci, (e, a, s, r) in enumerate(combos):
        if ci % 100 == 0: print(f"  {ci}/{len(combos)}", flush=True)
        pr = (e, a, s, r)
        bi.run_backtest_no_lookahead(dtr, sym, params=pr, risk_pct=2.0)
        rv = bi.run_backtest_no_lookahead(dte, sym, params=pr, risk_pct=2.0)
        sc = bi.compute_score(rv.get('metrics', {}))
        if sc > best_s:
            best_s = sc; best_p = pr

    if best_p is None:
        print("  FAILED"); continue

    # Validate on the held-out 30-day window
    result = bi.run_backtest_no_lookahead(df_val, sym, params=best_p, risk_pct=2.0)
    m = result.get('metrics', {})
    ret = m.get('return_pct', 0); pf = m.get('profit_factor', 0)
    wr  = m.get('win_rate', 0);   nt = m.get('total_trades', 0)
    dd  = m.get('max_drawdown', 0)
    tag = ' OK ' if ret > 0 else 'LOSS'
    print(f"  [{tag}] EMA={best_p[0]} ADX>={best_p[1]:.0f} SL={best_p[2]} RR={best_p[3]}"
          f" | {nt}t WR={wr:.1f}% PF={pf:.2f} Ret={ret:+.1f}% DD=${dd:.0f}", flush=True)
    final[sym] = {'p': best_p, 'm': m}

# Build merged best_settings
existing = {}
try:
    existing = json.loads(BS.read_text()).get('instruments', {})
except Exception:
    pass

merged = existing.copy()
for sym, data in final.items():
    p = data['p']; m = data['m']
    # Only overwrite if new params are better (higher PF)
    prev_pf = merged.get(sym, {}).get('PF', 0)
    pf = m.get('profit_factor', 0)
    if pf > prev_pf:
        merged[sym] = {
            'EMA': int(p[0]), 'ADX': float(p[1]),
            'SL_Mult': float(p[2]), 'RR': float(p[3]),
            'Trades': int(m.get('total_trades', 0)),
            'Win_Rate': float(m.get('win_rate', 0)),
            'PF': float(pf),
            'Return_Pct': float(m.get('return_pct', 0)),
            'Max_DD': float(m.get('max_drawdown', 0)),
        }

import datetime
BS.write_text(json.dumps({
    'generated_at': datetime.datetime.utcnow().isoformat() + 'Z',
    'source': 'oot_optimizer_v2 (90d opt, 30d val)',
    'instruments': merged,
}, indent=2))

ok = sum(1 for s, d in final.items() if d['m'].get('return_pct', 0) > 0)
print(f"\n{'='*70}")
print(f"  FINAL: {ok}/{len(final)} profitable on out-of-sample 30-day validation")
print(f"{'='*70}")
for sym in sorted(final.keys()):
    p = final[sym]['p']; m = final[sym]['m']
    print(f"  {sym:8s}  EMA={p[0]:3d} ADX>={p[1]:.0f} SL={p[2]:.1f} RR={p[3]:.1f}"
          f"  =>  {m.get('total_trades',0):3d}t WR={m.get('win_rate',0):.1f}%"
          f"  PF={m.get('profit_factor',0):.2f}  Ret={m.get('return_pct',0):+.1f}%")

print(f"\n  Saved → {BS}")

# Also save validation report
Path('backtest_results').mkdir(exist_ok=True)
report = {
    'engine': 'ICT_SMC_v2', 'method': 'oot_optimization',
    'opt_window_days': 90, 'val_window_days': 30,
    'risk_pct': 2.0, 'profitable': ok, 'total': len(final),
    'symbols': [
        {'symbol': s, 'params': {'EMA': d['p'][0], 'ADX': d['p'][1],
                                  'SL_Mult': d['p'][2], 'RR': d['p'][3]},
         **{k: v for k, v in d['m'].items()
            if k in ('total_trades','wins','win_rate','profit_factor',
                     'return_pct','total_profit','max_drawdown','final_balance')}}
        for s, d in sorted(final.items())
    ]
}
Path('backtest_results/ict_smc_oot_validation.json').write_text(
    json.dumps(report, indent=2, default=str))
print(f"  Report → backtest_results/ict_smc_oot_validation.json")
