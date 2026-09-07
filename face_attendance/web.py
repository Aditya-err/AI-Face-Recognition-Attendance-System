"""Flask application: pages + JSON API.

The heavy lifting lives in the other modules; this file only wires HTTP to
them.  All JSON responses use the helper below so numpy values are never
leaked into responses.
"""

import logging

from flask import (Flask, Response, jsonify, render_template, request,
                   stream_with_context)

from . import attendance as attendance_mod
from . import config as config_module
from . import db
from . import people as people_mod
from .camera import CameraService

log = logging.getLogger("web")

DEFAULT_PAGE_SIZE = 50
MIN_REGISTRATION_SAMPLES = 3


def _clean(value):
    """Recursively convert non-JSON values (numpy) into plain Python."""
    import numpy as np
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {k: _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    return value


def _session_summary(session):
    """Lightweight, JSON-safe view of a registration session (the raw session
    holds numpy embeddings, which must never leave the server)."""
    if not session:
        return None
    return {
        "mode": session["mode"],
        "name": session["name"],
        "code": session["code"],
        "person_db_id": session.get("person_db_id"),
        "sample_count": len(session.get("samples", [])),
        "target": int(db.get_setting("registration_samples")),
    }


def create_app(cfg: config_module.Config) -> Flask:
    db.init(cfg.db_path)

    app = Flask(
        __name__,
        template_folder=str(cfg.project_root / "face_attendance" / "templates"),
        static_folder=str(cfg.project_root / "face_attendance" / "static"),
    )
    app.config["SECRET_KEY"] = cfg.secret_key
    app.config["CFG"] = cfg

    camera = CameraService(cfg)
    app.extensions["camera"] = camera

    # ---------------------------------------------------------- pages
    @app.route("/")
    def dashboard_page():
        return render_template("dashboard.html", active="dashboard")

    @app.route("/recognize")
    def recognize_page():
        return render_template("recognize.html", active="recognize")

    @app.route("/register")
    def register_page():
        return render_template("register.html", active="register")

    @app.route("/people")
    def people_page():
        return render_template("people.html", active="people")

    @app.route("/records")
    def records_page():
        return render_template("records.html", active="records")

    @app.route("/settings")
    def settings_page():
        return render_template("settings.html", active="settings")

    # ---------------------------------------------------------- video
    @app.route("/video_feed")
    def video_feed():
        if camera.state not in ("running", "error", "starting"):
            camera.start()  # lazy start (e.g. autostart disabled)
        return Response(
            stream_with_context(camera.mjpeg_frames()),
            mimetype="multipart/x-mixed-replace; boundary=frame")

    # ---------------------------------------------------------- status
    @app.route("/api/camera/status")
    def camera_status():
        return jsonify(_clean(camera.status()))

    @app.route("/api/camera/restart", methods=["POST"])
    def camera_restart():
        camera.restart()
        return jsonify({"ok": True, "status": camera.status()})

    # ---------------------------------------------------------- people
    @app.route("/api/people")
    def api_people():
        return jsonify({"people": _clean(people_mod.list_people())})

    @app.route("/api/people/<int:pid>")
    def api_person(pid):
        person = people_mod.get_person(pid)
        if person is None:
            return jsonify({"error": "Person not found."}), 404
        return jsonify({"person": _clean(person)})

    @app.route("/api/people/<int:pid>", methods=["DELETE"])
    def api_person_delete(pid):
        if not people_mod.delete_person(pid):
            return jsonify({"error": "Person not found."}), 404
        return jsonify({"ok": True,
                        "message": "Person and their face data were deleted. "
                                   "Attendance history was kept."})

    # ---------------------------------------------------------- dashboard
    @app.route("/api/dashboard")
    def api_dashboard():
        return jsonify(_clean(attendance_mod.dashboard_stats()))

    @app.route("/api/activity")
    def api_activity():
        limit = min(int(request.args.get("limit", 25)), 200)
        return jsonify({"activity": _clean(attendance_mod.activity(limit))})

    # ---------------------------------------------------------- records
    def _record_args():
        return {
            "search": request.args.get("q") or None,
            "date": request.args.get("date") or None,
            "person_db_id": (int(request.args["person"])
                             if request.args.get("person") else None),
            "status": request.args.get("status") or None,
        }

    @app.route("/api/records")
    def api_records():
        kwargs = _record_args()
        offset = max(int(request.args.get("offset", 0)), 0)
        limit = min(int(request.args.get("limit", DEFAULT_PAGE_SIZE)), 500)
        rows = attendance_mod.records(limit=limit, offset=offset, **kwargs)
        total = attendance_mod.records_count(**kwargs)
        return jsonify({
            "records": _clean(rows),
            "total": total,
            "offset": offset,
            "limit": limit,
            "page": offset // limit if limit else 0,
        })

    @app.route("/api/records/meta")
    def api_records_meta():
        return jsonify({
            "dates": attendance_mod.distinct_dates(),
            "people": _clean(people_mod.list_people()),
        })

    @app.route("/api/records/export")
    def api_records_export():
        csv_text = attendance_mod.export_csv(**_record_args())
        return Response(
            "\ufeff" + csv_text,  # BOM so Excel opens UTF-8 correctly
            mimetype="text/csv",
            headers={"Content-Disposition":
                     "attachment; filename=attendance_export.csv"})

    # ---------------------------------------------------------- settings
    @app.route("/api/settings")
    def api_settings():
        return jsonify(_clean({
            "settings": db.all_settings(),
            "camera": camera.status(),
            "models_available": cfg.models_available(),
        }))

    @app.route("/api/settings", methods=["POST"])
    def api_settings_update():
        data = request.get_json(silent=True) or {}
        updates = data.get("updates")
        if not isinstance(updates, dict) or not updates:
            return jsonify({"error": "No settings provided."}), 400
        if ("similarity_threshold" in updates
                and "uncertain_threshold" in updates
                and float(updates["uncertain_threshold"]) >
                float(updates["similarity_threshold"])):
            return jsonify({"ok": False, "errors": {
                "uncertain_threshold": ("The uncertainty floor must be "
                                         "below the match threshold.")}}), 400

        errors = {}
        applied = {}
        for key, value in updates.items():
            try:
                db.set_setting(key, value)
                applied[key] = value
            except ValueError as exc:
                errors[key] = str(exc)
        if errors:
            return jsonify({"ok": False, "errors": errors,
                            "applied": applied}), 400
        return jsonify({"ok": True, "applied": applied,
                        "settings": db.all_settings()})

    # ---------------------------------------------------------- register
    @app.route("/api/register/start", methods=["POST"])
    def api_register_start():
        data = request.get_json(silent=True) or {}
        mode = data.get("mode", "new")
        name = (data.get("name") or "").strip()
        code = (data.get("code") or "").strip()

        if mode == "new":
            if not name:
                return jsonify({"error": "Name is required."}), 400
            if not code:
                return jsonify({"error": "Person ID is required."}), 400
            if len(name) > people_mod.NAME_MAX:
                return jsonify({"error": f"Name must be at most "
                                f"{people_mod.NAME_MAX} characters."}), 400
            if len(code) > people_mod.PERSON_CODE_MAX:
                return jsonify({"error": f"Person ID must be at most "
                                f"{people_mod.PERSON_CODE_MAX} characters."}), 400
            existing = people_mod.find_by_code(code)
            if existing:
                return jsonify({
                    "error": f"Person ID '{code}' is already registered to "
                             f"'{existing['name']}'.",
                    "code": "duplicate_code",
                    "person": existing,
                }), 409
        elif mode == "existing":
            person = people_mod.find_by_code(code)
            if person is None:
                return jsonify({"error": f"No person with ID '{code}' was "
                                         f"found."}), 404
            name = person["name"]
        else:
            return jsonify({"error": "Unknown registration mode."}), 400

        if not cfg.models_available():
            return jsonify({
                "error": "Face models are not installed. Run "
                         "`python download_models.py` first."}), 503
        if camera.state != "running":
            if camera.state != "error":
                camera.start()
            return jsonify({
                "error": "The camera is not running, so samples cannot be "
                         "captured. Check the camera and try again.",
                "code": "camera_not_running",
                "camera": camera.status(),
            }), 409

        camera.begin_registration(mode, name, code,
                                  person["id"] if mode == "existing" else None)
        return jsonify({"ok": True,
                        "session": _session_summary(camera.session())})

    @app.route("/api/register/capture", methods=["POST"])
    def api_register_capture():
        result = camera.capture_registration_sample()
        result["session"] = _session_summary(camera.session())
        status = 200 if result.get("ok") else 422
        return jsonify(_clean(result)), status

    @app.route("/api/register/finish", methods=["POST"])
    def api_register_finish():
        data = request.get_json(silent=True) or {}
        force = bool(data.get("force"))
        session = camera.session()
        if session is None:
            return jsonify({"error": "No registration session is active."}), 400

        samples = camera.pending_samples()
        target = int(db.get_setting("registration_samples"))
        if len(samples) < min(MIN_REGISTRATION_SAMPLES, target):
            return jsonify({
                "error": f"Need at least {MIN_REGISTRATION_SAMPLES} face "
                         f"samples, have {len(samples)}."}), 400

        if session["mode"] == "new":
            # Refuse accidental duplicate registration: same face under a
            # different name/ID is almost certainly a mistake.
            threshold = float(db.get_setting("similarity_threshold"))
            match = None
            for emb in samples:
                best = people_mod.highest_similarity(emb)
                if best and best["similarity"] >= threshold:
                    if match is None or best["similarity"] > match["similarity"]:
                        match = best
            if match and not force:
                return jsonify({
                    "error": (f"This face already matches "
                              f"'{match['name']}' (ID {match['code']}) with "
                              f"{int(round(match['similarity'] * 100))}% "
                              "similarity. Registering the same face twice is "
                              "not allowed - use re-enrolment for that person "
                              "if you need more samples."),
                    "code": "duplicate_face",
                    "match": _clean(match),
                }), 409
            try:
                person = people_mod.create_person(session["code"],
                                                  session["name"], samples)
            except ValueError as exc:
                return jsonify({"error": str(exc)}), 409
            message = (f"'{person['name']}' registered with "
                       f"{person['sample_count']} face samples.")
        else:
            person_row = people_mod.get_person(session["person_db_id"])
            if person_row is None:
                return jsonify({"error": "Person no longer exists."}), 404
            try:
                person = people_mod.add_embeddings(
                    person_row["id"], samples,
                    int(db.get_setting("max_embeddings_per_person")))
            except ValueError as exc:
                return jsonify({"error": str(exc)}), 409
            message = (f"Added samples to '{person['name']}' - now "
                       f"{person['sample_count']} total.")

        camera.cancel_registration()
        return jsonify({"ok": True, "person": _clean(person),
                        "message": message})

    @app.route("/api/register/cancel", methods=["POST"])
    def api_register_cancel():
        camera.cancel_registration()
        return jsonify({"ok": True})

    @app.route("/api/register/session")
    def api_register_session():
        return jsonify({"session": _session_summary(camera.session())})

    # ---------------------------------------------------------- errors
    @app.errorhandler(404)
    def not_found(_e):
        if request.path.startswith("/api/"):
            return jsonify({"error": "Not found."}), 404
        return render_template("error.html", code=404,
                               message="Page not found."), 404

    @app.errorhandler(500)
    def server_error(_e):
        log.exception("Unhandled error")
        if request.path.startswith("/api/"):
            return jsonify({"error": "Internal server error."}), 500
        return render_template("error.html", code=500,
                               message="Internal server error."), 500

    if cfg.autostart_camera:
        camera.start()

    return app
