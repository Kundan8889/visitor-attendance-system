import atexit
import datetime as dt
import json
import os
import re
import shutil
import subprocess
import threading
import sys

from flask import Flask, Response, jsonify, request, render_template, redirect, url_for

import live1 as live
import numpy as np


app = Flask(__name__)
EMBEDDINGS_PATHS = [
    os.path.join("embeddings", "faces_embeddings_done_4classes.npz"),
    "faces_embeddings_with_labels.npz",
]
VISITOR_DB_PATH = os.path.join("data", "visitor_passes.json")
VISITOR_REGISTRY_PATH = os.path.join("data", "visitor_registry.json")
VISITOR_OUTPUT_ROOT = os.path.join("data", "visitors")
VISITOR_LOG_PATH = os.path.join("logs", "visitor_embeddings.log")
MODEL_TRAIN_LOG_PATH = os.path.join("logs", "model_training.log")
visitor_embed_lock = threading.Lock()
visitor_embed_tasks = {}
VISITOR_PASS_EDIT_FIELDS = {
    "company",
    "contact",
    "id_proof",
    "laptop_no",
    "mobile_qty",
    "storage_details",
    "person_to_visit",
    "department",
    "purpose",
    "security_issued_by",
    "employee_visited",
}


@atexit.register
def _cleanup():
    try:
        live.stop_all_cameras()
    except Exception:
        pass


@app.route("/")
def index():
    return render_template("index.html")


def _sanitize_name(name):
    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "_", str(name).strip())
    return cleaned.strip("_") or "person"


def _start_face_collection(name, source, output_root=None):
    script_path = os.path.join(os.path.dirname(__file__), "face_data_collection.py")
    if not os.path.exists(script_path):
        raise FileNotFoundError("face_data_collection.py not found")
    args = [
        sys.executable,
        script_path,
        "--name",
        name,
        "--source",
        str(source or "0"),
    ]
    if output_root:
        args.extend(["--output-root", output_root])
    return subprocess.Popen(args)


def _ensure_parent_dir(path):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def _normalize_key(value):
    return str(value or "").strip().lower()


def _load_visitors():
    if not os.path.exists(VISITOR_DB_PATH):
        return []
    try:
        with open(VISITOR_DB_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return []
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and isinstance(data.get("visitors"), list):
        return data["visitors"]
    return []


def _load_visitor_registry():
    if not os.path.exists(VISITOR_REGISTRY_PATH):
        return []
    try:
        with open(VISITOR_REGISTRY_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return []
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and isinstance(data.get("visitors"), list):
        return data["visitors"]
    return []


def _save_visitor_registry(entries):
    _ensure_parent_dir(VISITOR_REGISTRY_PATH)
    with open(VISITOR_REGISTRY_PATH, "w", encoding="utf-8") as f:
        json.dump(entries, f, indent=2)


def _update_visitor_registry_fields(name, label, updates):
    if not updates or not name:
        return False
    entries = _load_visitor_registry()
    key = _normalize_key(name)
    label_key = _normalize_key(label)
    changed = False
    for item in entries:
        if not isinstance(item, dict):
            continue
        item_key = _normalize_key(item.get("name"))
        item_label = _normalize_key(item.get("label"))
        if item_key != key and item_label != key and item_key != label_key and item_label != label_key:
            continue
        for field, value in updates.items():
            if field not in VISITOR_PASS_EDIT_FIELDS:
                continue
            if item.get(field) != value:
                item[field] = value
                changed = True
        if changed:
            item["updated_at"] = _iso(_utc_now())
        break
    if changed:
        _save_visitor_registry(entries)
    return changed


def _register_visitor_face(payload):
    name = (payload.get("name") or "").strip()
    if not name:
        raise ValueError("Visitor name is required")
    label = _sanitize_name(name)
    now_iso = _iso(_utc_now())
    details = {
        "company": (payload.get("company") or "").strip(),
        "contact": (payload.get("contact") or "").strip(),
        "id_proof": (payload.get("id_proof") or "").strip(),
        "laptop_no": (payload.get("laptop_no") or "").strip(),
        "mobile_qty": (payload.get("mobile_qty") or "").strip(),
        "storage_details": (payload.get("storage_details") or "").strip(),
        "person_to_visit": (payload.get("person_to_visit") or "").strip(),
        "department": (payload.get("department") or "").strip(),
        "purpose": (payload.get("purpose") or "").strip(),
        "security_issued_by": (payload.get("security_issued_by") or "").strip(),
        "employee_visited": (payload.get("employee_visited") or "").strip(),
    }
    if not details["employee_visited"] and details["person_to_visit"]:
        details["employee_visited"] = details["person_to_visit"]
    entries = _load_visitor_registry()
    key = _normalize_key(name)
    for item in entries:
        if not isinstance(item, dict):
            continue
        if _normalize_key(item.get("name")) != key:
            continue
        changed = False
        if item.get("name") != name:
            item["name"] = name
            changed = True
        if not item.get("label"):
            item["label"] = label
            changed = True
        for field, value in details.items():
            if value and item.get(field) != value:
                item[field] = value
                changed = True
        if changed:
            item["updated_at"] = now_iso
            _save_visitor_registry(entries)
        return name, item.get("label") or label

    entry = {
        "name": name,
        "label": label,
        "created_at": now_iso,
        "updated_at": now_iso,
    }
    entry.update(details)
    entries.append(entry)
    _save_visitor_registry(entries)
    return name, label


def _resolve_visitor_label(name):
    label = _sanitize_name(name)
    entries = _load_visitor_registry()
    key = _normalize_key(name)
    for item in entries:
        if not isinstance(item, dict):
            continue
        if _normalize_key(item.get("name")) == key and item.get("label"):
            return str(item.get("label"))
    return label


def _log_visitor_update(message):
    _ensure_parent_dir(VISITOR_LOG_PATH)
    stamp = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        with open(VISITOR_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(f"[{stamp}] {message}\n")
    except Exception:
        pass


def _mark_visitor_embed_start(label):
    with visitor_embed_lock:
        visitor_embed_tasks[label] = {
            "label": label,
            "started_at": _iso(_utc_now()),
        }


def _mark_visitor_embed_end(label):
    with visitor_embed_lock:
        visitor_embed_tasks.pop(label, None)


def _visitor_embed_status_snapshot():
    with visitor_embed_lock:
        labels = list(visitor_embed_tasks.keys())
        started = {k: v.get("started_at") for k, v in visitor_embed_tasks.items()}
    return labels, started


def _tail_file(path, max_lines=5):
    if not path or max_lines <= 0:
        return []
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            lines = f.readlines()
    except Exception:
        return []
    if len(lines) <= max_lines:
        return [line.rstrip("\n") for line in lines if line.strip()]
    return [line.rstrip("\n") for line in lines[-max_lines:] if line.strip()]


def _log_model_training(message):
    _ensure_parent_dir(MODEL_TRAIN_LOG_PATH)
    stamp = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        with open(MODEL_TRAIN_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(f"[{stamp}] {message}\n")
    except Exception:
        pass


def _python_can_import(exe, module_name):
    try:
        proc = subprocess.run(
            [exe, "-c", f"import {module_name}"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except Exception:
        return False
    return proc.returncode == 0


def _resolve_python_executable():
    candidates = []
    env_exe = os.getenv("VISITOR_PYTHON") or os.getenv("VENV_PYTHON")
    if env_exe:
        candidates.append(env_exe)
    base_dir = os.path.dirname(__file__)
    venv_exe = os.path.join(base_dir, ".venv", "Scripts", "python.exe")
    candidates.append(venv_exe)
    candidates.append(sys.executable)
    for exe in candidates:
        if not exe or not os.path.exists(exe):
            continue
        if _python_can_import(exe, "cv2"):
            return exe
    return sys.executable


def _collect_visitor_samples(label):
    person_dir = os.path.join(VISITOR_OUTPUT_ROOT, label)
    if not os.path.isdir(person_dir):
        return 0
    images_dir = os.path.join(person_dir, "images")
    os.makedirs(images_dir, exist_ok=True)

    sessions = [
        d for d in os.listdir(person_dir)
        if os.path.isdir(os.path.join(person_dir, d)) and d != "images"
    ]
    if not sessions:
        return 0
    copied = 0
    sessions.sort()
    for session in sessions:
        samples_root = os.path.join(person_dir, session, "angle_samples")
        if not os.path.isdir(samples_root):
            continue
        for root, _, files in os.walk(samples_root):
            for fname in files:
                lower = fname.lower()
                if not (lower.endswith(".jpg") or lower.endswith(".jpeg") or lower.endswith(".png")):
                    continue
                src = os.path.join(root, fname)
                stamp = os.path.basename(session)
                rel = os.path.relpath(root, samples_root).replace(os.sep, "_")
                dst_name = f"{label}_{stamp}_{rel}_{fname}".replace("__", "_")
                dst = os.path.join(images_dir, dst_name)
                if os.path.exists(dst):
                    continue
                try:
                    shutil.copy2(src, dst)
                    copied += 1
                except Exception:
                    continue
    return copied


def _run_visitor_embeddings_update(label):
    _mark_visitor_embed_start(label)
    try:
        data_dir = VISITOR_OUTPUT_ROOT
        emb_path = os.path.join("embeddings", "faces_embeddings_done_4classes.npz")
        gen_script = os.path.join(os.path.dirname(__file__), "generate_embeddings.py")
        train_script = os.path.join(os.path.dirname(__file__), "model_training.py")
        if not os.path.exists(gen_script) or not os.path.exists(train_script):
            _log_visitor_update("Embedding update skipped: scripts not found.")
            return

        copied = _collect_visitor_samples(label)
        _log_visitor_update(f"Collected {copied} samples for {label}.")
        if copied == 0:
            _log_visitor_update(f"No samples found for {label}.")
        python_exe = _resolve_python_executable()
        _log_visitor_update(f"Using python: {python_exe}")

        max_images_raw = os.getenv("VISITOR_EMBED_MAX_IMAGES", "120").strip()
        augs_raw = os.getenv("VISITOR_EMBED_AUGS", "1").strip()
        max_images = None
        augmentations = None
        if max_images_raw.isdigit():
            max_images = int(max_images_raw)
        if augs_raw.isdigit():
            augmentations = int(augs_raw)

        gen_args = [
            python_exe,
            gen_script,
            "--data-dir",
            data_dir,
            "--embeddings-path",
            emb_path,
            "--replace-existing",
            "--people",
            label,
        ]
        if max_images and max_images > 0:
            gen_args.extend(["--max-images-per-person", str(max_images)])
        if augmentations is not None and augmentations >= 0:
            gen_args.extend(["--augmentations", str(augmentations)])
        _log_visitor_update(f"generate_embeddings args: {' '.join(gen_args)}")

        try:
            with open(VISITOR_LOG_PATH, "a", encoding="utf-8") as logf:
                proc = subprocess.run(gen_args, stdout=logf, stderr=logf, text=True)
            if proc.returncode != 0:
                _log_visitor_update(f"generate_embeddings failed with code {proc.returncode}")
                return
        except Exception:
            _log_visitor_update("generate_embeddings crashed.")
            return

        try:
            with open(VISITOR_LOG_PATH, "a", encoding="utf-8") as logf:
                proc = subprocess.run(
                    [python_exe, train_script, "--embeddings-path", emb_path],
                    stdout=logf,
                    stderr=logf,
                    text=True,
                )
            if proc.returncode != 0:
                _log_visitor_update(f"model_training failed with code {proc.returncode}")
                return
        except Exception:
            _log_visitor_update("model_training crashed.")
            return
    finally:
        _mark_visitor_embed_end(label)


def _schedule_visitor_update(proc, label):
    def _worker():
        try:
            proc.wait()
        except Exception:
            return
        _log_visitor_update(f"Capture finished for {label}. Starting embedding update.")
        _run_visitor_embeddings_update(label)

    thread = threading.Thread(target=_worker, daemon=True)
    thread.start()


def _run_model_training(emb_path):
    train_script = os.path.join(os.path.dirname(__file__), "model_training.py")
    if not os.path.exists(train_script):
        _log_model_training("model_training.py not found.")
        return False
    if not os.path.exists(emb_path):
        _log_model_training(f"Embeddings not found: {emb_path}")
        return False
    python_exe = _resolve_python_executable()
    _log_model_training(f"Using python: {python_exe}")
    args = [python_exe, train_script, "--embeddings-path", emb_path]
    _log_model_training(f"model_training args: {' '.join(args)}")
    try:
        with open(MODEL_TRAIN_LOG_PATH, "a", encoding="utf-8") as logf:
            proc = subprocess.run(args, stdout=logf, stderr=logf, text=True)
        if proc.returncode != 0:
            _log_model_training(f"model_training failed with code {proc.returncode}")
            return False
    except Exception:
        _log_model_training("model_training crashed.")
        return False
    return True


def _save_visitors(visitors):
    _ensure_parent_dir(VISITOR_DB_PATH)
    with open(VISITOR_DB_PATH, "w", encoding="utf-8") as f:
        json.dump(visitors, f, indent=2)


def _generate_visitor_id(visitors):
    existing = {_normalize_key(v.get("id")) for v in visitors}
    base = dt.datetime.now().strftime("VIS%Y%m%d%H%M%S")
    candidate = base
    counter = 1
    while _normalize_key(candidate) in existing:
        counter += 1
        candidate = f"{base}-{counter}"
    return candidate


def _coerce_float(value):
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _utc_now():
    return dt.datetime.now(dt.timezone.utc)


def _iso(dt_obj):
    return dt_obj.replace(microsecond=0).isoformat()


def _parse_datetime(value):
    if not value:
        return None
    text = str(value).strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = f"{text[:-1]}+00:00"
    try:
        parsed = dt.datetime.fromisoformat(text)
        if parsed.tzinfo is not None:
            return parsed.astimezone().replace(tzinfo=None)
        return parsed
    except ValueError:
        pass
    try:
        return dt.datetime.strptime(text, "%Y-%m-%d %H:%M:%S")
    except ValueError:
        return None


def _build_visitor_record(payload, visitors):
    name = (payload.get("name") or "").strip()
    if not name:
        raise ValueError("Visitor name is required")
    pass_no = (payload.get("pass_no") or "").strip()
    if not pass_no:
        raise ValueError("Pass number is required")
    existing = {str(v.get("id") or "").strip().lower() for v in visitors}
    if pass_no.lower() in existing:
        raise ValueError("Pass number already exists")
    now = _utc_now()
    local_now = dt.datetime.now()
    valid_hours = _coerce_float(payload.get("valid_hours"))
    if not valid_hours or valid_hours <= 0:
        valid_hours = 4.0
    valid_until = (payload.get("valid_until") or "").strip()
    if not valid_until:
        valid_until = _iso(now + dt.timedelta(hours=valid_hours))
    time_in = (payload.get("time_in") or "").strip()
    person_to_visit = (payload.get("person_to_visit") or payload.get("host") or "").strip()
    employee_visited = (payload.get("employee_visited") or person_to_visit).strip()
    return {
        "id": pass_no,
        "name": name,
        "company": (payload.get("company") or "").strip(),
        "contact": (payload.get("contact") or "").strip(),
        "id_proof": (payload.get("id_proof") or "").strip(),
        "laptop_no": (payload.get("laptop_no") or "").strip(),
        "mobile_qty": (payload.get("mobile_qty") or "").strip(),
        "storage_details": (payload.get("storage_details") or "").strip(),
        "purpose": (payload.get("purpose") or "").strip(),
        "person_to_visit": person_to_visit,
        "department": (payload.get("department") or "").strip(),
        "created_at": _iso(now),
        "valid_until": valid_until,
        "status": "active",
        "time_in": time_in,
        "time_out": (payload.get("time_out") or "").strip(),
        "security_issued_by": (payload.get("security_issued_by") or "").strip(),
        "employee_visited": employee_visited,
    }


def _existing_embedding_paths():
    return [path for path in EMBEDDINGS_PATHS if os.path.exists(path)]


def _load_embeddings(path):
    data = np.load(path, allow_pickle=True)
    if "arr_0" in data and "arr_1" in data:
        return data["arr_0"], data["arr_1"], "arr"
    if "X" in data and "Y" in data:
        return data["X"], data["Y"], "xy"
    if "embeddings" in data and "labels" in data:
        return data["embeddings"], data["labels"], "emb_labels"
    raise ValueError("Unsupported embeddings format")


def _save_embeddings(path, x, y, mode):
    if mode == "arr":
        np.savez(path, x, y)
    elif mode == "xy":
        np.savez(path, X=x, Y=y)
    elif mode == "emb_labels":
        np.savez(path, embeddings=x, labels=y)
    else:
        raise ValueError("Unsupported embeddings format")


def _normalize_label(value):
    if value is None:
        return ""
    if isinstance(value, (bytes, bytearray, np.bytes_)):
        try:
            text = value.decode("utf-8", errors="ignore")
        except Exception:
            text = str(value)
    else:
        text = str(value)
    return text.strip().lower()


def _remove_embeddings_labels(labels, emb_path):
    if not emb_path or not os.path.exists(emb_path):
        return False, "Embeddings file not found"

    try:
        x, y, mode = _load_embeddings(emb_path)
    except Exception as exc:
        return False, str(exc)

    y = np.asarray(y, dtype=object)
    label_set = {_normalize_label(l) for l in labels if _normalize_label(l)}
    if not label_set:
        return False, "No label provided"

    normalized_y = np.array([_normalize_label(v) for v in y], dtype=object)
    keep = np.array([val not in label_set for val in normalized_y], dtype=bool)
    if keep.all():
        return False, "Label not found in embeddings"

    x_new = x[keep]
    y_new = y[keep]
    if y_new.size == 0:
        return False, "No embeddings left after removal"

    _save_embeddings(emb_path, x_new, y_new, mode)
    return True, None


def _remove_visitor_from_embeddings(name_or_label):
    label = _sanitize_name(name_or_label)
    labels = {label}
    if name_or_label:
        labels.add(str(name_or_label))
        labels.add(str(name_or_label).replace("_", " "))
        labels.add(str(name_or_label).replace(" ", "_"))

    emb_path = os.path.join("embeddings", "faces_embeddings_done_4classes.npz")
    changed, err = _remove_embeddings_labels(labels, emb_path)
    if not changed:
        return False, err or "Label not found in embeddings"
    return True, None


def _retrain_model(emb_path):
    script_path = os.path.join(os.path.dirname(__file__), "model_training.py")
    if not os.path.exists(script_path):
        raise FileNotFoundError("model_training.py not found")
    args = [
        sys.executable,
        script_path,
        "--embeddings-path",
        emb_path,
    ]
    return subprocess.Popen(args)


@app.route("/video_feed/<camera_id>")
def video_feed(camera_id):
    try:
        stream = live.generate_mjpeg_stream(camera_id)
    except KeyError:
        return Response("Camera not found", status=404)
    return Response(stream, mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/api/cameras", methods=["GET", "POST"])
def api_cameras():
    if request.method == "GET":
        return jsonify(live.list_cameras())

    payload = request.get_json(silent=True) or {}
    try:
        cam = live.add_camera(
            camera_id=payload.get("camera_id"),
            name=payload.get("name", ""),
            source=payload.get("source", ""),
            host=payload.get("host", ""),
            username=payload.get("username", ""),
            password=payload.get("password", ""),
            port=payload.get("port", 554),
            path=payload.get("path", "Streaming/Channels/101"),
            role=payload.get("role", "general"),
            auto_start=bool(payload.get("auto_start", False)),
        )
        return jsonify({"ok": True, "camera": cam})
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400


@app.route("/api/cameras/<camera_id>/start", methods=["POST"])
def api_camera_start(camera_id):
    try:
        status = live.start_camera(camera_id)
        return jsonify({"ok": True, "status": status})
    except KeyError as exc:
        return jsonify({"ok": False, "error": str(exc)}), 404
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400


@app.route("/api/cameras/<camera_id>/stop", methods=["POST"])
def api_camera_stop(camera_id):
    try:
        status = live.stop_camera(camera_id)
        return jsonify({"ok": True, "status": status})
    except KeyError as exc:
        return jsonify({"ok": False, "error": str(exc)}), 404


@app.route("/api/cameras/<camera_id>", methods=["DELETE"])
def api_camera_remove(camera_id):
    try:
        live.remove_camera(camera_id)
        return jsonify({"ok": True})
    except KeyError as exc:
        return jsonify({"ok": False, "error": str(exc)}), 404


@app.route("/api/cameras/<camera_id>/status")
def api_camera_status(camera_id):
    try:
        return jsonify(live.build_status_payload(camera_id=camera_id))
    except KeyError as exc:
        return jsonify({"ok": False, "error": str(exc)}), 404


@app.route("/api/stop_all", methods=["POST"])
def api_stop_all():
    live.stop_all_cameras()
    return jsonify({"ok": True})


@app.route("/api/status")
def api_status():
    return jsonify(live.build_status_payload())


@app.route("/api/attendance")
def api_attendance():
    limit_raw = request.args.get("limit", default="100")
    try:
        limit = max(1, int(limit_raw))
    except ValueError:
        limit = 100
    return jsonify(live.get_attendance_rows(limit=limit))


@app.route("/api/visitors", methods=["GET"])
def api_visitors():
    return jsonify(_load_visitors())


@app.route("/api/visitors/auto", methods=["GET"])
def api_visitors_auto():
    since_raw = (request.args.get("since") or "").strip()
    since_dt = None
    if since_raw:
        try:
            since_val = float(since_raw)
            if since_val > 1e12:
                since_val = since_val / 1000.0
            since_dt = dt.datetime.fromtimestamp(since_val)
        except ValueError:
            since_dt = _parse_datetime(since_raw)

    visitors = _load_visitors()
    items = []
    for v in visitors:
        if not isinstance(v, dict):
            continue
        if str(v.get("name") or "").strip().lower() == "unknown":
            continue
        if v.get("auto_issued"):
            created = _parse_datetime(v.get("created_at"))
            if not (since_dt and created and created <= since_dt):
                entry_item = dict(v)
                entry_item["event"] = "entry"
                entry_item["event_at"] = v.get("created_at") or ""
                items.append(entry_item)
        exit_at = _parse_datetime(v.get("exit_detected_at"))
        if exit_at:
            if since_dt and exit_at <= since_dt:
                continue
            exit_item = dict(v)
            exit_item["event"] = "exit"
            exit_item["event_at"] = v.get("exit_detected_at") or ""
            items.append(exit_item)

    def _event_sort_key(item):
        return _parse_datetime(item.get("event_at")) or dt.datetime.min

    items.sort(key=_event_sort_key)
    for v in items:
        v["pass_url"] = f"/visitors/pass/{v.get('id')}"
    return jsonify(items)


@app.route("/api/visitors/register_face", methods=["POST"])
def api_visitors_register_face():
    payload = request.get_json(silent=True)
    if payload is None:
        payload = request.form.to_dict(flat=True)
    payload = payload or {}
    try:
        name, label = _register_visitor_face(payload)
    except ValueError as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400

    source = (payload.get("source") or "0").strip()
    try:
        proc = _start_face_collection(name, source, output_root=VISITOR_OUTPUT_ROOT)
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 500
    _schedule_visitor_update(proc, label)

    return jsonify(
        {
            "ok": True,
            "visitor": {"name": name},
            "pid": proc.pid,
            "message": "Visitor face capture started. Embeddings will update after capture finishes.",
        }
    )


@app.route("/api/visitors/rebuild_embeddings", methods=["POST"])
def api_visitors_rebuild_embeddings():
    payload = request.get_json(silent=True) or {}
    name = (payload.get("name") or "").strip()
    if not name:
        return jsonify({"ok": False, "error": "Visitor name is required"}), 400

    label = _resolve_visitor_label(name)

    def _worker():
        _log_visitor_update(f"Manual rebuild requested for {label}.")
        _run_visitor_embeddings_update(label)

    threading.Thread(target=_worker, daemon=True).start()
    return jsonify(
        {
            "ok": True,
            "label": label,
            "message": "Embedding rebuild started in background.",
        }
    )


@app.route("/api/visitors/embeddings/status", methods=["GET"])
def api_visitors_embeddings_status():
    labels, started = _visitor_embed_status_snapshot()
    tail = _tail_file(VISITOR_LOG_PATH, max_lines=6)
    return jsonify(
        {
            "running": bool(labels),
            "labels": labels,
            "started_at": started,
            "log_tail": tail,
            "updated_at": _iso(_utc_now()),
        }
    )


@app.route("/api/model/retrain", methods=["POST"])
def api_model_retrain():
    payload = request.get_json(silent=True) or {}
    emb_path = (payload.get("embeddings_path") or "").strip()
    if not emb_path:
        emb_path = os.path.join("embeddings", "faces_embeddings_done_4classes.npz")

    def _worker():
        _log_model_training("Manual retrain requested.")
        _run_model_training(emb_path)

    threading.Thread(target=_worker, daemon=True).start()
    return jsonify(
        {
            "ok": True,
            "embeddings_path": emb_path,
            "message": "Model retraining started in background.",
        }
    )


@app.route("/api/visitors/issue", methods=["POST"])
def api_visitors_issue():
    payload = request.get_json(silent=True)
    from_form = False
    if payload is None:
        payload = request.form.to_dict(flat=True)
        from_form = True
    payload = payload or {}
    visitors = _load_visitors()
    try:
        record = _build_visitor_record(payload, visitors)
    except ValueError as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400

    capture_pid = None
    capture_error = None
    capture_label = None
    if payload.get("capture"):
        capture_label = f"{record['id']}_{record['name']}"
        record["capture_label"] = capture_label
        record["capture_root"] = VISITOR_OUTPUT_ROOT

    visitors.append(record)
    _save_visitors(visitors)

    if capture_label:
        source = (payload.get("source") or "0").strip()
        try:
            proc = _start_face_collection(capture_label, source, output_root=VISITOR_OUTPUT_ROOT)
            capture_pid = proc.pid
            record["capture_pid"] = capture_pid
            _save_visitors(visitors)
        except Exception as exc:
            capture_error = str(exc)

    pass_url = f"/visitors/pass/{record['id']}"
    if from_form:
        return redirect(pass_url)

    return jsonify(
        {
            "ok": True,
            "visitor": record,
            "capture_pid": capture_pid,
            "capture_error": capture_error,
            "pass_url": pass_url,
        }
    )


@app.route("/api/visitors/revoke", methods=["POST"])
def api_visitors_revoke():
    payload = request.get_json(silent=True) or {}
    key = (payload.get("id") or payload.get("name") or "").strip()
    if not key:
        return jsonify({"ok": False, "error": "Visitor id or name is required"}), 400

    key_norm = _normalize_key(key)
    visitors = _load_visitors()
    if not visitors:
        return jsonify({"ok": False, "error": "No visitor passes found"}), 404

    to_remove = [v for v in visitors if _normalize_key(v.get("id")) == key_norm or _normalize_key(v.get("name")) == key_norm]
    if not to_remove:
        return jsonify({"ok": False, "error": "Visitor pass not found"}), 404

    remaining = [v for v in visitors if v not in to_remove]
    _save_visitors(remaining)

    cleaned_paths = []
    for visitor in to_remove:
        label = visitor.get("capture_label") or visitor.get("name") or visitor.get("id")
        if not label:
            continue
        folder = os.path.join(VISITOR_OUTPUT_ROOT, _sanitize_name(label))
        if os.path.isdir(folder):
            shutil.rmtree(folder, ignore_errors=True)
            cleaned_paths.append(folder)

    return jsonify(
        {
            "ok": True,
            "removed": [v.get("id") or v.get("name") for v in to_remove],
            "cleaned_paths": cleaned_paths,
        }
    )


@app.route("/api/visitors/pass/update", methods=["POST"])
def api_visitors_pass_update():
    payload = request.get_json(silent=True) or {}
    pass_id = (payload.get("id") or payload.get("pass_id") or "").strip()
    fields = payload.get("fields") or {}
    if not pass_id:
        return jsonify({"ok": False, "error": "Pass id is required"}), 400
    if not isinstance(fields, dict):
        return jsonify({"ok": False, "error": "Fields must be an object"}), 400

    updates = {}
    for key, value in fields.items():
        if key not in VISITOR_PASS_EDIT_FIELDS:
            continue
        updates[key] = str(value or "").strip()

    if not updates:
        return jsonify({"ok": False, "error": "No editable fields provided"}), 400

    visitors = _load_visitors()
    record = None
    for v in visitors:
        if not isinstance(v, dict):
            continue
        if _normalize_key(v.get("id")) == _normalize_key(pass_id):
            record = v
            break

    if record is None:
        return jsonify({"ok": False, "error": "Visitor pass not found"}), 404

    if not updates.get("employee_visited") and updates.get("person_to_visit"):
        if not str(record.get("employee_visited") or "").strip():
            updates["employee_visited"] = updates.get("person_to_visit", "")

    changed = False
    for key, value in updates.items():
        if record.get(key) != value:
            record[key] = value
            changed = True
    if changed:
        record["updated_at"] = _iso(_utc_now())
        _save_visitors(visitors)
        reg_label = _resolve_visitor_label(record.get("name") or "")
        _update_visitor_registry_fields(record.get("name") or "", reg_label, updates)

    return jsonify({"ok": True, "updated": updates, "changed": changed})


@app.route("/api/visitors/remove", methods=["POST"])
def api_visitors_remove():
    payload = request.get_json(silent=True) or {}
    name = (payload.get("name") or "").strip()
    if not name:
        return jsonify({"ok": False, "error": "Visitor name is required"}), 400

    label = _resolve_visitor_label(name)

    # Remove embeddings
    changed, err = _remove_visitor_from_embeddings(label)
    if not changed:
        return jsonify({"ok": False, "error": err or "Visitor not found in embeddings"}), 404

    # Remove visitor data folder
    cleaned_paths = []
    folder = os.path.join(VISITOR_OUTPUT_ROOT, _sanitize_name(label))
    if os.path.isdir(folder):
        shutil.rmtree(folder, ignore_errors=True)
        cleaned_paths.append(folder)

    # Remove from registry
    entries = _load_visitor_registry()
    key = _normalize_key(name)
    new_entries = []
    removed_registry = False
    for item in entries:
        if not isinstance(item, dict):
            new_entries.append(item)
            continue
        if _normalize_key(item.get("name")) == key or _normalize_key(item.get("label")) == _normalize_key(label):
            removed_registry = True
            continue
        new_entries.append(item)
    if removed_registry:
        _save_visitor_registry(new_entries)

    # Retrain model after removal
    retrain_pid = None
    retrain_error = None
    try:
        proc = _retrain_model(os.path.join("embeddings", "faces_embeddings_done_4classes.npz"))
        retrain_pid = proc.pid
    except Exception as exc:
        retrain_error = str(exc)

    return jsonify(
        {
            "ok": True,
            "removed": name,
            "label": label,
            "embeddings_updated": True,
            "cleaned_paths": cleaned_paths,
            "registry_removed": removed_registry,
            "retrain_pid": retrain_pid,
            "retrain_error": retrain_error,
        }
    )


@app.route("/visitors/pass/<pass_id>")
def visitor_pass_view(pass_id):
    visitors = _load_visitors()
    record = next((v for v in visitors if str(v.get("id") or "").lower() == str(pass_id).lower()), None)
    if record is None:
        return Response("Visitor pass not found", status=404)

    created = None
    raw_created = record.get("created_at")
    if raw_created:
        try:
            created = dt.datetime.fromisoformat(raw_created)
        except ValueError:
            created = None
    if created is None:
        created = dt.datetime.now()

    date_str = created.strftime("%d/%m/%Y")
    time_str = created.strftime("%H:%M:%S")
    time_in = record.get("time_in") or time_str

    return render_template(
        "visitor_pass.html",
        org_name=os.getenv("VISITOR_PASS_ORG", "MAGTORQ PRIVATE LIMITED"),
        visitor=record,
        pass_no=record.get("id") or "",
        pass_date=date_str,
        time_in=time_in,
    )


@app.route("/visitors/new")
def visitor_pass_form():
    return render_template(
        "visitor_form.html",
        org_name=os.getenv("VISITOR_PASS_ORG", "MAGTORQ PRIVATE LIMITED"),
    )


@app.route("/visitors/register")
def visitor_register_form():
    return render_template(
        "visitor_register.html",
        org_name=os.getenv("VISITOR_PASS_ORG", "MAGTORQ PRIVATE LIMITED"),
    )


@app.route("/api/employees/capture", methods=["POST"])
def api_employees_capture():
    payload = request.get_json(silent=True) or {}
    name = (payload.get("name") or "").strip()
    source = (payload.get("source") or "0").strip()
    if not name:
        return jsonify({"ok": False, "error": "Employee name is required"}), 400

    try:
        proc = _start_face_collection(name, source)
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 500

    return jsonify(
        {
            "ok": True,
            "employee": {"name": name},
            "pid": proc.pid,
            "message": "Face data collection started. A capture window should open on the server machine.",
        }
    )


@app.route("/api/employees/remove", methods=["POST"])
def api_employees_remove():
    payload = request.get_json(silent=True) or {}
    emp_id = (payload.get("emp_id") or "").strip().upper()
    name = (payload.get("name") or "").strip()
    if not emp_id and not name:
        return jsonify({"ok": False, "error": "Employee name or ID is required"}), 400

    key = emp_id or name
    removed_names = [key]
    labels = set()
    for n in removed_names:
        if not n:
            continue
        labels.add(n)
        labels.add(_sanitize_name(n))
        labels.add(n.replace("_", " "))
        labels.add(n.replace(" ", "_"))

    for label in labels:
        for path in [os.path.join("data", label), os.path.join("data", "face_collection", label)]:
            if os.path.isdir(path):
                shutil.rmtree(path, ignore_errors=True)

    emb_paths = _existing_embedding_paths()
    emb_changed = False
    emb_warning = None
    updated_paths = []
    found_any = False
    if labels and emb_paths:
        for emb_path in emb_paths:
            changed, err = _remove_embeddings_labels(labels, emb_path)
            if changed:
                emb_changed = True
                found_any = True
                updated_paths.append(emb_path)
            elif err and err not in ("Label not found in embeddings",):
                emb_warning = err
                found_any = True

    if emb_paths and not found_any:
        return jsonify({"ok": False, "error": "Employee not found in embeddings"}), 404

    retrain_pid = None
    retrain_error = None
    retrain_path = emb_paths[0] if emb_paths else None
    if emb_changed and retrain_path:
        try:
            proc = _retrain_model(retrain_path)
            retrain_pid = proc.pid
        except Exception as exc:
            retrain_error = str(exc)

    return jsonify(
        {
            "ok": True,
            "removed": removed_names,
            "embeddings_updated": emb_changed,
            "embeddings_paths": updated_paths,
            "retrain_pid": retrain_pid,
            "warning": emb_warning,
            "retrain_error": retrain_error,
        }
    )


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5001, debug=False, threaded=True)
