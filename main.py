import os
import threading
import webbrowser

from web.alpha_api import app, init_db, _start_bot_engine


def main() -> None:
    init_db()
    host = os.getenv("HOST", "127.0.0.1")
    port = int(os.getenv("PORT", "5000"))
    debug = os.getenv("FLASK_DEBUG", "0") == "1"
    open_browser = os.getenv("OPEN_BROWSER", "1") == "1"
    dashboard_url = f"http://{host}:{port}/dashboard"

    # Auto-start the trading bot alongside the web server
    auto_start = os.getenv("AUTO_START_BOT", "1") == "1"
    if auto_start:
        def _auto_start():
            import time; time.sleep(2)
            result, code = _start_bot_engine("auto")
            if code == 200:
                print(f"✅ Bot auto-started (PID {result.get('pid', '?')})")
            else:
                print(f"⚠️  Bot auto-start failed: {result.get('message', 'unknown error')}")
        threading.Thread(target=_auto_start, daemon=True).start()

    if open_browser:
        threading.Timer(1.0, lambda: webbrowser.open(dashboard_url)).start()

    print(f"Starting web app at {dashboard_url}")
    app.run(host=host, port=port, debug=debug, use_reloader=False)


if __name__ == "__main__":
    main()
