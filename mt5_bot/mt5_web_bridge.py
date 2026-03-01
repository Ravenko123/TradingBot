import argparse
import json
import time
from datetime import datetime, timezone

import requests

try:
    import MetaTrader5 as mt5
except Exception:
    mt5 = None


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def fetch_mt5_snapshot() -> dict:
    if mt5 is None:
        return {
            "connected": False,
            "account_id": "unknown",
            "strategy": "mt5_ea",
            "balance": 0.0,
            "equity": 0.0,
            "profit": 0.0,
            "open_positions": 0,
            "positions": [],
            "timestamp": utc_now_iso(),
            "error": "MetaTrader5 module not available",
        }

    inited = False
    try:
        inited = bool(mt5.initialize())
        if not inited:
            return {
                "connected": False,
                "account_id": "unknown",
                "strategy": "mt5_ea",
                "balance": 0.0,
                "equity": 0.0,
                "profit": 0.0,
                "open_positions": 0,
                "positions": [],
                "timestamp": utc_now_iso(),
                "error": f"mt5 initialize failed: {mt5.last_error()}",
            }

        acc = mt5.account_info()
        if acc is None:
            return {
                "connected": False,
                "account_id": "unknown",
                "strategy": "mt5_ea",
                "balance": 0.0,
                "equity": 0.0,
                "profit": 0.0,
                "open_positions": 0,
                "positions": [],
                "timestamp": utc_now_iso(),
                "error": "account_info is None",
            }

        rows = mt5.positions_get() or []
        positions = []
        for pos in rows:
            direction = "BUY" if int(pos.type) == mt5.ORDER_TYPE_BUY else "SELL"
            positions.append({
                "ticket": int(pos.ticket),
                "symbol": str(pos.symbol),
                "direction": direction,
                "volume": float(pos.volume),
                "open_price": float(pos.price_open),
                "current_price": float(pos.price_current),
                "sl": float(pos.sl) if pos.sl is not None else 0.0,
                "tp": float(pos.tp) if pos.tp is not None else 0.0,
                "profit": float(pos.profit),
                "swap": float(pos.swap),
            })

        return {
            "connected": True,
            "account_id": str(getattr(acc, "login", "unknown")),
            "strategy": "mt5_ea",
            "balance": float(getattr(acc, "balance", 0.0) or 0.0),
            "equity": float(getattr(acc, "equity", 0.0) or 0.0),
            "profit": float(getattr(acc, "profit", 0.0) or 0.0),
            "open_positions": int(len(positions)),
            "positions": positions,
            "timestamp": utc_now_iso(),
        }
    except Exception as e:
        return {
            "connected": False,
            "account_id": "unknown",
            "strategy": "mt5_ea",
            "balance": 0.0,
            "equity": 0.0,
            "profit": 0.0,
            "open_positions": 0,
            "positions": [],
            "timestamp": utc_now_iso(),
            "error": str(e),
        }
    finally:
        if inited:
            try:
                mt5.shutdown()
            except Exception:
                pass


def push_snapshot(api_url: str, token: str, payload: dict) -> tuple[bool, str]:
    url = f"{api_url.rstrip('/')}/api/mt5/runtime/push"
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }
    try:
        resp = requests.post(url, headers=headers, data=json.dumps(payload), timeout=12)
        if resp.status_code >= 200 and resp.status_code < 300:
            return True, "ok"
        return False, f"http {resp.status_code}: {resp.text[:220]}"
    except Exception as e:
        return False, str(e)


def main():
    ap = argparse.ArgumentParser(description="Push MT5 runtime data to Zenith dashboard account")
    ap.add_argument("--api-url", default="http://127.0.0.1:5000", help="Dashboard API base URL")
    ap.add_argument("--token", required=True, help="Bearer token from dashboard login")
    ap.add_argument("--interval", type=int, default=5, help="Push interval seconds")
    args = ap.parse_args()

    interval = max(2, int(args.interval))
    print(f"Starting MT5→Web bridge: {args.api_url} every {interval}s")

    while True:
        snap = fetch_mt5_snapshot()
        ok, msg = push_snapshot(args.api_url, args.token, snap)
        tag = "PUSH_OK" if ok else "PUSH_FAIL"
        print(f"[{utc_now_iso()}] {tag} account={snap.get('account_id')} open={snap.get('open_positions')} msg={msg}")
        time.sleep(interval)


if __name__ == "__main__":
    main()
