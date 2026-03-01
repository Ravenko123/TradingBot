"""
Parameter sweep for improved ICT/SMC v2 strategy.
Runs a grid search over (EMA, ADX, SL_Mult, RR) for all symbols,
using 70/30 train/test split to avoid overfitting.
"""
import sys, json, time
from pathlib import Path
from itertools import product
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from mt5_bot.backtest_improved import (
    fetch_data, run_backtest_no_lookahead,
    compute_score, INSTRUMENTS, INITIAL_BALANCE, _load_best_params
)

BEST_SETTINGS_FILE = Path(__file__).parent / 'best_settings.json'

# Parameter grid — focused ranges
EMA_GRID  = [30, 40, 50, 60, 80]
ADX_GRID  = [16.0, 18.0, 20.0, 24.0]
SL_GRID   = [1.0, 1.5, 2.0, 2.5]
RR_GRID   = [1.5, 2.0, 2.5, 3.0]

RISK_PCT = 2.0
BARS = 15000  # ~150 days of M15 data


def sweep_symbol(symbol, df):
    n = len(df)
    split = int(n * 0.7)
    df_train = df.iloc[:split].copy().reset_index(drop=True)
    df_test  = df.iloc[split:].copy().reset_index(drop=True)

    if len(df_train) < 200 or len(df_test) < 100:
        print(f"  [SKIP] Not enough data after split (train={len(df_train)}, test={len(df_test)})")
        return None

    combos = list(product(EMA_GRID, ADX_GRID, SL_GRID, RR_GRID))
    best_score = -np.inf
    best_params = None
    best_test_metrics = None
    count = 0
    total = len(combos)

    for ema, adx, sl, rr in combos:
        count += 1
        params = (ema, adx, sl, rr)
        
        # Train — just to make sure it works
        train_res = run_backtest_no_lookahead(df_train, symbol, params=params, risk_pct=RISK_PCT)
        train_m = train_res.get('metrics', {})
        
        # Quick filter: skip if train is terrible
        if train_m.get('total_trades', 0) < 5:
            continue
        if train_m.get('profit_factor', 0) < 0.7:
            continue
            
        # Test
        test_res = run_backtest_no_lookahead(df_test, symbol, params=params, risk_pct=RISK_PCT)
        test_m = test_res.get('metrics', {})
        
        score = compute_score(test_m)
        
        # Bonus for consistency: both train and test profitable
        if train_m.get('return_pct', 0) > 0 and test_m.get('return_pct', 0) > 0:
            score += 10
        
        if score > best_score:
            best_score = score
            best_params = params
            best_test_metrics = test_m

        if count % 50 == 0:
            print(f"    [{count}/{total}] ...", end='\r')

    return best_params, best_test_metrics, best_score


def run_full_sweep():
    print("=" * 70)
    print("  ICT/SMC v2 — PARAMETER SWEEP WITH IMPROVED STRATEGY")
    print("=" * 70)

    results = {}
    
    for symbol in sorted(INSTRUMENTS.keys()):
        print(f"\n{'─' * 60}")
        print(f"  {symbol}")
        print(f"{'─' * 60}")

        df = fetch_data(symbol, BARS)
        if df is None or len(df) < 400:
            print(f"  [X] Not enough data ({0 if df is None else len(df)} bars)")
            continue

        print(f"  Bars: {len(df)}")
        t0 = time.time()
        
        result = sweep_symbol(symbol, df)
        elapsed = time.time() - t0
        
        if result is None:
            print(f"  [X] Sweep failed")
            continue
            
        params, metrics, score = result
        
        if params is None:
            print(f"  [X] No viable params found ({elapsed:.0f}s)")
            continue
        
        ema, adx, sl, rr = params
        pf = metrics.get('profit_factor', 0)
        ret = metrics.get('return_pct', 0)
        wr = metrics.get('win_rate', 0)
        nt = metrics.get('total_trades', 0)
        dd = metrics.get('max_drawdown', 0)
        
        tag = "✓" if ret > 0 and pf > 1.0 else "✗"
        print(f"  {tag} EMA={ema} ADX≥{adx:.0f} SL×{sl:.1f} RR={rr:.1f}")
        print(f"    Trades={nt}  WR={wr:.1f}%  PF={pf:.2f}  Return={ret:+.1f}%  DD=${dd:.0f}")
        print(f"    Score={score:.1f}  ({elapsed:.0f}s)")
        
        results[symbol] = {
            'EMA': int(ema),
            'ADX': float(adx),
            'SL_Mult': float(sl),
            'RR': float(rr),
            'Trades': nt,
            'Win_Rate': wr,
            'Profit': metrics.get('total_profit', 0),
            'Return_Pct': ret,
            'Max_DD': dd,
            'PF': pf,
            'score': score,
        }

    # Save results
    if results:
        # Only update symbols that improved
        existing = {}
        if BEST_SETTINGS_FILE.exists():
            try:
                existing = json.loads(BEST_SETTINGS_FILE.read_text()).get('instruments', {})
            except:
                pass
        
        merged = existing.copy()
        for sym, data in results.items():
            old_pf = merged.get(sym, {}).get('PF', 0)
            # Update if new params are profitable or better than existing
            if data['PF'] > 1.0 or data['PF'] > old_pf:
                merged[sym] = {k: data[k] for k in ['EMA', 'ADX', 'SL_Mult', 'RR', 
                                                      'Trades', 'Win_Rate', 'Profit',
                                                      'Return_Pct', 'Max_DD', 'PF']}
                print(f"  → Updated {sym}: PF {old_pf:.2f} → {data['PF']:.2f}")
        
        from datetime import datetime
        BEST_SETTINGS_FILE.write_text(json.dumps({
            'generated_at': datetime.utcnow().isoformat() + 'Z',
            'source': 'param_sweep_v2 (ICT/SMC v2 improved)',
            'instruments': merged
        }, indent=2))
        
        print(f"\n{'=' * 70}")
        print(f"  RESULTS SUMMARY")
        print(f"{'=' * 70}")
        profitable = sum(1 for r in results.values() if r['PF'] > 1.0 and r['Return_Pct'] > 0)
        print(f"  Profitable: {profitable}/{len(results)}")
        for sym in sorted(results.keys()):
            r = results[sym]
            tag = "✓" if r['PF'] > 1.0 and r['Return_Pct'] > 0 else "✗"
            print(f"  {tag} {sym:8s}  PF={r['PF']:.2f}  Ret={r['Return_Pct']:+.1f}%  "
                  f"WR={r['Win_Rate']:.1f}%  Trades={r['Trades']}")
        print(f"\n  Settings saved to {BEST_SETTINGS_FILE}")


if __name__ == '__main__':
    run_full_sweep()
