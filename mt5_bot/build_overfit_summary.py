import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


def load_latest_reports(base_dir: Path, symbols):
    latest = {}
    for symbol in symbols:
        files = sorted(base_dir.glob(f"overfit_proof_20_{symbol}_bars60000_*.json"))
        if not files:
            continue
        latest[symbol] = files[-1]
    return latest


def main():
    symbols = ["EURUSD", "GBPUSD", "USDJPY", "XAUUSD", "GBPJPY", "BTCUSD", "NAS100"]
    base_dir = Path(__file__).parent / "backtest_results" / "overfit_proof_20"
    out_dir = base_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    latest = load_latest_reports(base_dir, symbols)
    rows = []
    details = {}

    for symbol, fp in latest.items():
        payload = json.loads(fp.read_text())
        consistency = payload.get("consistency", {})
        dist = payload.get("distribution", {})
        anti = payload.get("anti_luck", {})
        overfit = payload.get("overfitting_risk", {})

        row = {
            "symbol": symbol,
            "runs": consistency.get("runs_total", 0),
            "profitable_runs": consistency.get("profitable_runs", 0),
            "consistency_pct": consistency.get("consistency_pct", 0),
            "median_return_pct": dist.get("median_return_pct", 0),
            "median_pf": dist.get("median_pf", 0),
            "binomial_p_value": anti.get("binomial_p_value", 1.0),
            "anti_luck": anti.get("interpretation", "n/a"),
            "overfit_risk": overfit.get("interpretation", "n/a"),
            "verdict": payload.get("verdict", "n/a"),
            "source_file": fp.name,
        }
        rows.append(row)
        details[symbol] = payload

    if not rows:
        raise RuntimeError("No overfit proof files found.")

    df = pd.DataFrame(rows).sort_values(
        ["consistency_pct", "median_return_pct", "binomial_p_value"],
        ascending=[False, False, True],
    )

    now_utc = datetime.now(timezone.utc)
    stamp = now_utc.strftime("%Y%m%d_%H%M%S")
    csv_path = out_dir / f"overfit_proof_summary_{stamp}.csv"
    json_path = out_dir / f"overfit_proof_summary_{stamp}.json"
    md_path = out_dir / f"overfit_proof_summary_{stamp}.md"

    df.to_csv(csv_path, index=False)
    json_path.write_text(json.dumps({"generated_at": now_utc.isoformat(), "rows": rows, "details": details}, indent=2, default=str))

    top = df.iloc[0]
    md = []
    md.append("# Overfitting / Luck Proof Summary (20 Historical Backtests per Symbol)")
    md.append("")
    md.append(f"Generated: {now_utc.isoformat()}")
    md.append("")
    md.append("## Key Finding")
    md.append(
        f"Top candidate is **{top['symbol']}** with {int(top['profitable_runs'])}/20 profitable windows "
        f"({top['consistency_pct']}%), median return {top['median_return_pct']}%, median PF {top['median_pf']}."
    )
    md.append("")
    md.append("## Statistical note")
    md.append(
        "Binomial p-value tests whether the number of profitable windows could be random under 50/50 luck. "
        "Thresholds: <0.05 strong evidence, <0.10 moderate evidence."
    )
    md.append("")
    md.append("## Ranking")
    md.append("")
    cols = list(df.columns)
    md.append("| " + " | ".join(cols) + " |")
    md.append("|" + "|".join(["---"] * len(cols)) + "|")
    for _, row in df.iterrows():
        vals = []
        for c in cols:
            vals.append(str(row[c]))
        md.append("| " + " | ".join(vals) + " |")

    md_path.write_text("\n".join(md), encoding="utf-8")

    print(f"Saved: {csv_path}")
    print(f"Saved: {json_path}")
    print(f"Saved: {md_path}")


if __name__ == "__main__":
    main()
