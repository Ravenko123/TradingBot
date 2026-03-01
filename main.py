import os
import threading
import webbrowser

from web.alpha_api import app, init_db


def main() -> None:
    init_db()
    host = os.getenv("HOST", "127.0.0.1")
    port = int(os.getenv("PORT", "5000"))
    debug = os.getenv("FLASK_DEBUG", "0") == "1"
    open_browser = os.getenv("OPEN_BROWSER", "1") == "1"
    dashboard_url = f"http://{host}:{port}/dashboard"

    if open_browser:
        threading.Timer(1.0, lambda: webbrowser.open(dashboard_url)).start()

    print(f"Starting web app at {dashboard_url}")
    app.run(host=host, port=port, debug=debug, use_reloader=False)


if __name__ == "__main__":
    main()
