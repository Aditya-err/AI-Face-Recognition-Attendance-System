"""Application entry point.

Usage:
    python run.py                 # start on http://127.0.0.1:8000
    python run.py --port 9000     # custom port
    python run.py --host 0.0.0.0  # allow other machines on the LAN

Optional dependencies:
    waitress   -> production-grade WSGI server (recommended on Windows).
    Otherwise the Flask development server is used with threads enabled.
"""

import argparse
import logging

from face_attendance.config import Config
from face_attendance.web import create_app

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
)


def main() -> None:
    parser = argparse.ArgumentParser(description="AI Face Attendance")
    parser.add_argument("--host", default=None, help="Bind address")
    parser.add_argument("--port", type=int, default=None,
                        help="TCP port (default from FA_PORT or 8000)")
    parser.add_argument("--no-camera", action="store_true",
                        help="Do not auto-start the camera thread")
    args = parser.parse_args()

    cfg = Config()
    if args.host:
        cfg.host = args.host
    if args.port:
        cfg.port = args.port
    if args.no_camera:
        cfg.autostart_camera = False

    app = create_app(cfg)
    url = f"http://{cfg.host}:{cfg.port}"
    print("\n  AI Face Attendance running at", url)
    if not cfg.models_available():
        print("  [WARNING] Face model files are missing.")
        print("            Run:  python download_models.py")
    print()

    try:
        from waitress import serve
        serve(app, host=cfg.host, port=cfg.port, threads=16)
    except ImportError:
        app.run(host=cfg.host, port=cfg.port, debug=cfg.debug,
                threaded=True, use_reloader=False)


if __name__ == "__main__":
    main()
