import argparse
import csv
import datetime
import json
import logging
import os
import pickle
import re
import sys
import tempfile
import time
from queue import Empty, Queue
from threading import Event, Lock, Thread
from types import SimpleNamespace
from urllib.parse import quote

import cv2 as cv
import numpy as np
from keras_facenet import FaceNet
from mtcnn.mtcnn import MTCNN

from config import RTSP_URL

try:
    from sort import Sort
except Exception as exc:
    Sort = None
    SORT_IMPORT_ERROR = exc
else:
    SORT_IMPORT_ERROR = None

try:
    from ultralytics.trackers.byte_tracker import BYTETracker
except Exception as exc:
    BYTETracker = None
    BYTE_TRACK_IMPORT_ERROR = exc
else:
    BYTE_TRACK_IMPORT_ERROR = None


FRAME_WIDTH = int(os.getenv("FRAME_WIDTH", "640"))
FRAME_HEIGHT = int(os.getenv("FRAME_HEIGHT", "360"))
FRAME_QUEUE_SIZE = 1
STREAM_JPEG_QUALITY = 80
STREAM_SLEEP_SECONDS = 0.03
MAX_DISPLAY_FPS = 120.0

CSV_HEADERS = ["date", "emp name", "time"]
ATTENDANCE_CSV_PATH = os.path.join("logs", "attendance.csv")
VISITOR_DB_PATH = os.path.join("data", "visitor_passes.json")
VISITOR_REGISTRY_PATH = os.path.join("data", "visitor_registry.json")
VISITOR_DATA_ROOT = os.path.join("data", "visitors")
VISITOR_AUTO_VALID_HOURS = float(os.getenv("VISITOR_AUTO_VALID_HOURS", "4"))
VISITOR_AUTO_PREFIXES = tuple(
    p.strip().lower()
    for p in os.getenv("VISITOR_AUTO_PREFIXES", "vis,visitor").split(",")
    if p.strip()
)

EPISODE_GAP_SECONDS = 150
DETECTION_CONF_THRESHOLD = 0.50
SVM_CONF_THRESHOLD = 0.91
MIN_DET_FACE_SIZE = max(8, int(os.getenv("MIN_DET_FACE_SIZE", "20")))
DETECT_EVERY_N = 1 #max(1, int(os.getenv("DETECT_EVERY_N", "1")))
_raw_detection_scale = float(os.getenv("DETECTION_SCALE", "1.0"))
DETECTION_SCALE = 1.0 #min(1.0, max(0.35, _raw_detection_scale))
MAX_DET_STALE_FRAMES = max(1, int(os.getenv("MAX_DET_STALE_FRAMES", str(DETECT_EVERY_N * 2))))
RECOGNIZE_RETRY_FRAMES = max(1, int(os.getenv("RECOGNIZE_RETRY_FRAMES", "4")))
TRACK_FORGET_AFTER_FRAMES = 45
CAMERA_REOPEN_AFTER_SECONDS = 5
RTSP_FFMPEG_OPTIONS = (
    "rtsp_transport;tcp|"
    "fflags;discardcorrupt|"
    "flags;low_delay|"
    "max_delay;500000|"
    "stimeout;5000000"
)
MAIN_HEALTHCHECK_INTERVAL_SECONDS = 1.0

# ByteTrack tuning
BYTE_TRACK_HIGH_THRESH = 0.25
BYTE_TRACK_LOW_THRESH = 0.10
BYTE_TRACK_NEW_TRACK_THRESH = 0.25
BYTE_TRACK_BUFFER = max(15, int(os.getenv("BYTE_TRACK_BUFFER", "45")))
BYTE_TRACK_MATCH_THRESH = 0.8
BYTE_TRACK_FUSE_SCORE = True

# SORT tuning
SORT_MAX_AGE = int(os.getenv("SORT_MAX_AGE", "10"))
SORT_MIN_HITS = 1  #int(os.getenv("SORT_MIN_HITS", "1"))
SORT_IOU_THRESHOLD = float(os.getenv("SORT_IOU_THRESHOLD", "0.1"))
TRACKER_BACKEND = os.getenv("TRACKER_BACKEND", "sort").strip().lower()

LOG_LEVEL = os.getenv("APP_LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger("attendance_live")


attendance_lock = Lock()
runtime_bootstrap_lock = Lock()
model_load_lock = Lock()
embedder_lock = Lock()
model_predict_lock = Lock()
workers_lock = Lock()
visitor_lock = Lock()

attendance_rows = []
active_sessions = {}
last_seen_by_name = {}
last_visitor_pass_by_name = {}
visitor_registry_cache = {"loaded_at": 0.0, "entries": set(), "records": {}}

embedder = None
model = None
encoder = None

workers = {}


class CameraWorker:
    def __init__(self, camera_id, source_cfg, display_name=None, show_window=False, role="general"):
        self.camera_id = str(camera_id)
        self.display_name = (display_name or self.camera_id).strip() or self.camera_id
        self.source_cfg = source_cfg
        self.show_window = bool(show_window)
        self.role = sanitize_camera_role(role)

        self.frame_queue = Queue(maxsize=FRAME_QUEUE_SIZE)
        self.stop_event = Event()

        self.latest_frame_lock = Lock()
        self.runtime_lock = Lock()
        self.track_state_lock = Lock()

        self.latest_raw_frame = None
        self.latest_annotated_frame = None

        self.track_id_to_name = {}
        self.track_id_to_conf = {}
        self.in_time_log = {}
        self.out_time_log = {}

        self.capture_thread = None
        self.recognition_thread = None

        self.detector = None
        self.tracker = None
        self.tracker_backend = TRACKER_BACKEND or "sort"

        self.runtime_stats = {
            "camera_id": self.camera_id,
            "display_name": self.display_name,
            "running": False,
            "source": redact_source(source_cfg),
            "fps": 0.0,
            "known_tracks": 0,
            "active_tracks": 0,
            "queue_size": 0,
            "last_frame_at": "",
            "error": "",
            "tracker": self.tracker_backend,
        }

    def _set_latest_frame(self, frame, annotated=False):
        with self.latest_frame_lock:
            if annotated:
                self.latest_annotated_frame = frame.copy()
            else:
                self.latest_raw_frame = frame.copy()

    def get_latest_frame_copy(self):
        with self.latest_frame_lock:
            if self.latest_annotated_frame is not None:
                return self.latest_annotated_frame.copy()
            if self.latest_raw_frame is not None:
                return self.latest_raw_frame.copy()
            return None

    def _initialize_detector_and_tracker(self):
        if self.detector is None:
            start = time.time()
            try:
                self.detector = MTCNN(min_face_size=20, steps_threshold=[0.45, 0.55, 0.65], scale_factor=0.6)
            except TypeError:
                self.detector = MTCNN()
                logger.warning("MTCNN kwargs unsupported; using default constructor.")
            logger.info("[%s] MTCNN loaded in %.3fs", self.camera_id, time.time() - start)

        if self.tracker is None:
            if self.tracker_backend == "sort":
                if Sort is None:
                    raise RuntimeError(
                        "SORT is not available. Install dependencies or fix import."
                    ) from SORT_IMPORT_ERROR
                start = time.time()
                self.tracker = Sort(
                    max_age=SORT_MAX_AGE,
                    min_hits=SORT_MIN_HITS,
                    iou_threshold=SORT_IOU_THRESHOLD,
                )
                self.tracker_backend = "sort"
                logger.info("[%s] SORT initialized in %.3fs", self.camera_id, time.time() - start)
            else:
                if BYTETracker is None:
                    raise RuntimeError(
                        "ByteTrack is not available. Install ultralytics in this environment."
                    ) from BYTE_TRACK_IMPORT_ERROR

                start = time.time()
                bt_args = SimpleNamespace(
                    track_high_thresh=BYTE_TRACK_HIGH_THRESH,
                    track_low_thresh=BYTE_TRACK_LOW_THRESH,
                    new_track_thresh=BYTE_TRACK_NEW_TRACK_THRESH,
                    track_buffer=BYTE_TRACK_BUFFER,
                    match_thresh=BYTE_TRACK_MATCH_THRESH,
                    fuse_score=BYTE_TRACK_FUSE_SCORE,
                )
                self.tracker = BYTETracker(args=bt_args, frame_rate=30)
                self.tracker_backend = "bytesort"
                logger.info("[%s] ByteTrack initialized in %.3fs", self.camera_id, time.time() - start)

    def _run_tracker(self, detections, frame):
        if self.tracker_backend == "sort":
            det_np = np.array(detections, dtype=np.float32) if detections else np.empty((0, 5), dtype=np.float32)
            tracked = self.tracker.update(det_np)
            if tracked is None:
                return np.empty((0, 5), dtype=np.float32)
            return tracked

        if detections:
            det_np = np.array(detections, dtype=np.float32)
            xyxy = det_np[:, :4]
            conf = det_np[:, 4]
            cls = np.zeros((det_np.shape[0],), dtype=np.float32)
            xywh = np.empty((det_np.shape[0], 4), dtype=np.float32)
            xywh[:, 0] = (xyxy[:, 0] + xyxy[:, 2]) * 0.5
            xywh[:, 1] = (xyxy[:, 1] + xyxy[:, 3]) * 0.5
            xywh[:, 2] = np.maximum(0.0, xyxy[:, 2] - xyxy[:, 0])
            xywh[:, 3] = np.maximum(0.0, xyxy[:, 3] - xyxy[:, 1])
        else:
            xywh = np.empty((0, 4), dtype=np.float32)
            conf = np.empty((0,), dtype=np.float32)
            cls = np.empty((0,), dtype=np.float32)

        bt_results = SimpleNamespace(xywh=xywh, conf=conf, cls=cls)
        tracked = self.tracker.update(bt_results, img=frame)
        if tracked is None:
            return np.empty((0, 5), dtype=np.float32)
        return tracked

    def _clear_runtime_state(self):
        with self.track_state_lock:
            self.track_id_to_name.clear()
            self.track_id_to_conf.clear()
            self.in_time_log.clear()
            self.out_time_log.clear()

        if self.tracker is not None:
            if self.tracker_backend == "sort":
                try:
                    self.tracker = Sort(
                        max_age=SORT_MAX_AGE,
                        min_hits=SORT_MIN_HITS,
                        iou_threshold=SORT_IOU_THRESHOLD,
                    )
                except Exception:
                    logger.exception("[%s] Failed to reset SORT tracker state", self.camera_id)
            elif hasattr(self.tracker, "reset"):
                try:
                    self.tracker.reset()
                except Exception:
                    logger.exception("[%s] Failed to reset tracker state", self.camera_id)

        with self.latest_frame_lock:
            self.latest_raw_frame = None
            self.latest_annotated_frame = None

        while not self.frame_queue.empty():
            try:
                self.frame_queue.get_nowait()
            except Empty:
                break

    def alive(self):
        return (
            self.capture_thread is not None
            and self.capture_thread.is_alive()
            and self.recognition_thread is not None
            and self.recognition_thread.is_alive()
            and not self.stop_event.is_set()
        )

    def start(self):
        with self.runtime_lock:
            if self.alive():
                return

        initialize_shared_models()
        self._initialize_detector_and_tracker()
        self._clear_runtime_state()
        self.stop_event.clear()

        with self.runtime_lock:
            self.runtime_stats["running"] = True
            self.runtime_stats["source"] = redact_source(self.source_cfg)
            self.runtime_stats["fps"] = 0.0
            self.runtime_stats["known_tracks"] = 0
            self.runtime_stats["active_tracks"] = 0
            self.runtime_stats["queue_size"] = 0
            self.runtime_stats["last_frame_at"] = ""
            self.runtime_stats["error"] = ""
            self.runtime_stats["tracker"] = self.tracker_backend

        self.capture_thread = Thread(
            target=self._capture_loop,
            daemon=True,
            name=f"capture-{self.camera_id}",
        )
        self.recognition_thread = Thread(
            target=self._recognition_loop,
            daemon=True,
            name=f"recognition-{self.camera_id}",
        )
        self.capture_thread.start()
        self.recognition_thread.start()

    def stop(self):
        self.stop_event.set()
        if self.capture_thread is not None:
            self.capture_thread.join(timeout=5.0)
        if self.recognition_thread is not None:
            self.recognition_thread.join(timeout=5.0)
        self.capture_thread = None
        self.recognition_thread = None

        if self.show_window:
            cv.destroyWindow(self._window_name())

        with self.runtime_lock:
            self.runtime_stats["running"] = False
            self.runtime_stats["queue_size"] = self.frame_queue.qsize()

    def _window_name(self):
        return f"Face Recognition + Tracking [{self.camera_id}]"

    def _capture_loop(self):
        logger.info("[%s] Capture thread starting on source %s", self.camera_id, redact_source(self.source_cfg))
        cap = open_video_source(self.source_cfg)
        if not cap.isOpened():
            logger.error("[%s] Unable to open camera source: %s", self.camera_id, redact_source(self.source_cfg))
            self.stop_event.set()
            with self.runtime_lock:
                self.runtime_stats["error"] = "Unable to open camera source"
            return

        cap.set(cv.CAP_PROP_BUFFERSIZE, 1)
        last_ok_ts = time.time()

        while not self.stop_event.is_set():
            res, frame = cap.read()
            if not res:
                with self.runtime_lock:
                    self.runtime_stats["error"] = "Camera read failed"
                if time.time() - last_ok_ts >= CAMERA_REOPEN_AFTER_SECONDS:
                    logger.warning("[%s] Camera read failed, reopening source...", self.camera_id)
                    cap.release()
                    cap = open_video_source(self.source_cfg)
                    if not cap.isOpened():
                        logger.error("[%s] Reopen failed: %s", self.camera_id, redact_source(self.source_cfg))
                        time.sleep(1.0)
                        continue
                    cap.set(cv.CAP_PROP_BUFFERSIZE, 1)
                    last_ok_ts = time.time()
                time.sleep(0.05)
                continue

            last_ok_ts = time.time()
            with self.runtime_lock:
                self.runtime_stats["error"] = ""
            frame = cv.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT), interpolation=cv.INTER_LINEAR)
            # Keep stream alive even if recognition is temporarily behind.
            self._set_latest_frame(frame, annotated=False)

            if self.frame_queue.full():
                try:
                    self.frame_queue.get_nowait()
                except Empty:
                    pass
            self.frame_queue.put(frame)

        cap.release()
        logger.info("[%s] Capture thread stopped", self.camera_id)

    def _recognition_loop(self):
        logger.info("[%s] Recognition thread started", self.camera_id)
        prev_frame_ts = time.perf_counter()
        fps_ema = 0.0
        frame_idx = 0
        last_seen_track_frame = {}
        last_recog_try_frame = {}
        last_detections = []
        last_det_frame = -10**9

        while not self.stop_event.is_set():
            try:
                frame = self.frame_queue.get(timeout=0.1)
            except Empty:
                continue

            frame_idx += 1
            now_ts = time.perf_counter()
            frame_dt = max(now_ts - prev_frame_ts, 1e-6)
            prev_frame_ts = now_ts
            inst_fps = min(1.0 / frame_dt, MAX_DISPLAY_FPS)
            fps_ema = inst_fps if fps_ema == 0.0 else (0.9 * fps_ema + 0.1 * inst_fps)

            rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            detections = []
            run_detection = (frame_idx % DETECT_EVERY_N == 0)
            if run_detection:
                try:
                    if DETECTION_SCALE and DETECTION_SCALE < 1.0:
                        small_w = max(1, int(frame.shape[1] * DETECTION_SCALE))
                        small_h = max(1, int(frame.shape[0] * DETECTION_SCALE))
                        rgb_small = cv.resize(rgb_frame, (small_w, small_h), interpolation=cv.INTER_AREA)
                        faces = self.detector.detect_faces(rgb_small)
                        scale_x = frame.shape[1] / float(small_w)
                        scale_y = frame.shape[0] / float(small_h)
                    else:
                        faces = self.detector.detect_faces(rgb_frame)
                        scale_x = 1.0
                        scale_y = 1.0
                except Exception:
                    faces = []
                    scale_x = 1.0
                    scale_y = 1.0

                for face in faces:
                    try:
                        x, y, w, h = face["box"]
                        det_conf = float(face.get("confidence", 0.99))
                        if det_conf < DETECTION_CONF_THRESHOLD:
                            continue
                        x1 = int(max(0, x * scale_x))
                        y1 = int(max(0, y * scale_y))
                        x2 = int(min(frame.shape[1], (x + w) * scale_x))
                        y2 = int(min(frame.shape[0], (y + h) * scale_y))
                        if x2 <= x1 or y2 <= y1:
                            continue
                        if (x2 - x1) < MIN_DET_FACE_SIZE or (y2 - y1) < MIN_DET_FACE_SIZE:
                            continue
                        detections.append([x1, y1, x2, y2, det_conf])
                    except Exception:
                        logger.exception("[%s] Failed to parse face detection", self.camera_id)

                last_detections = list(detections)
                last_det_frame = frame_idx
            else:
                if (frame_idx - last_det_frame) <= MAX_DET_STALE_FRAMES:
                    detections = list(last_detections)
                else:
                    detections = []

            tracked = self._run_tracker(detections, frame)

            for d in tracked:
                if len(d) < 5:
                    continue
                x1, y1, x2, y2 = int(d[0]), int(d[1]), int(d[2]), int(d[3])
                track_id = int(d[4])
                last_seen_track_frame[track_id] = frame_idx

                x1 = max(0, min(x1, frame.shape[1] - 1))
                y1 = max(0, min(y1, frame.shape[0] - 1))
                x2 = max(0, min(x2, frame.shape[1]))
                y2 = max(0, min(y2, frame.shape[0]))
                if x2 <= x1 or y2 <= y1:
                    continue

                with self.track_state_lock:
                    predicted_name = self.track_id_to_name.get(track_id, "Unknown")
                    confidence = self.track_id_to_conf.get(track_id, 0.0)

                should_retry_recognition = (
                    predicted_name == "Unknown"
                    and (frame_idx - last_recog_try_frame.get(track_id, -10**9)) >= RECOGNIZE_RETRY_FRAMES
                )

                if should_retry_recognition:
                    last_recog_try_frame[track_id] = frame_idx
                    try:
                        pad_x = int(0.15 * (x2 - x1))
                        pad_y = int(0.15 * (y2 - y1))
                        cx1 = max(0, x1 - pad_x)
                        cy1 = max(0, y1 - pad_y)
                        cx2 = min(frame.shape[1], x2 + pad_x)
                        cy2 = min(frame.shape[0], y2 + pad_y)
                        face_region = rgb_frame[cy1:cy2, cx1:cx2]
                        if face_region.size == 0:
                            raise ValueError("Empty face crop")
                        face_region = cv.resize(face_region, (160, 160), interpolation=cv.INTER_AREA)
                        emb = get_embeddings(face_region)
                        predicted_name, confidence = predict_identity(emb)
                    except Exception:
                        logger.exception("[%s] Recognition failed for track_id=%s", self.camera_id, track_id)
                        predicted_name = "Unknown"
                        confidence = 0.0

                    if predicted_name != "Unknown":
                        with self.track_state_lock:
                            self.track_id_to_name[track_id] = predicted_name
                            self.track_id_to_conf[track_id] = confidence

                if predicted_name != "Unknown":
                    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    with self.track_state_lock:
                        if track_id not in self.in_time_log:
                            self.in_time_log[track_id] = now
                        self.out_time_log[track_id] = now
                    update_attendance_csv(predicted_name, now)
                    if self.role != "exit":
                        _auto_issue_visitor_pass(predicted_name, now, camera_id=self.camera_id)
                    if self.role == "exit":
                        _close_visitor_pass_on_exit(predicted_name, now, camera_id=self.camera_id)

                cv.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"{predicted_name} | ID:{track_id} | {confidence:.2f}"
                cv.putText(
                    frame,
                    label,
                    (x1, max(20, y1 - 10)),
                    cv.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 0),
                    2,
                )

            for tid in list(last_seen_track_frame.keys()):
                if frame_idx - last_seen_track_frame[tid] <= TRACK_FORGET_AFTER_FRAMES:
                    continue
                last_seen_track_frame.pop(tid, None)
                last_recog_try_frame.pop(tid, None)
                with self.track_state_lock:
                    self.track_id_to_name.pop(tid, None)
                    self.track_id_to_conf.pop(tid, None)

            cv.putText(
                frame,
                f"FPS: {fps_ema:.1f}",
                (10, 30),
                cv.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2,
            )
            self._set_latest_frame(frame, annotated=True)

            if self.show_window:
                cv.imshow(self._window_name(), frame)
                if cv.waitKey(1) & 0xFF == ord("q"):
                    self.stop_event.set()
                    break

            with self.runtime_lock:
                self.runtime_stats["fps"] = fps_ema
                self.runtime_stats["active_tracks"] = int(len(tracked))
                with self.track_state_lock:
                    self.runtime_stats["known_tracks"] = sum(
                        1 for v in self.track_id_to_name.values() if v != "Unknown"
                    )
                self.runtime_stats["queue_size"] = self.frame_queue.qsize()
                self.runtime_stats["last_frame_at"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        logger.info("[%s] Recognition thread stopped", self.camera_id)

    def build_status_payload(self):
        with self.runtime_lock:
            payload = dict(self.runtime_stats)
        payload["running"] = self.alive()
        payload["queue_size"] = self.frame_queue.qsize()
        payload["role"] = self.role
        return payload

    def generate_mjpeg_stream(self):
        while True:
            frame = self.get_latest_frame_copy()
            if frame is None:
                if self.stop_event.is_set() and not self.alive():
                    break
                time.sleep(STREAM_SLEEP_SECONDS)
                continue

            ok, jpeg = cv.imencode(
                ".jpg",
                frame,
                [int(cv.IMWRITE_JPEG_QUALITY), STREAM_JPEG_QUALITY],
            )
            if not ok:
                continue

            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + jpeg.tobytes() + b"\r\n"
            )


def get_embeddings(face_img):
    global embedder
    face_img = face_img.astype("float32")
    face_img = np.expand_dims(face_img, axis=0)
    with embedder_lock:
        yhat = embedder.embeddings(face_img)
    return yhat[0]


def predict_identity(emb):
    global model, encoder
    with model_predict_lock:
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba([emb])[0]
            best_idx = int(np.argmax(proba))
            conf = float(proba[best_idx])
            name = encoder.inverse_transform([best_idx])[0]
            if conf < SVM_CONF_THRESHOLD:
                return "Unknown", 0.0
            return name, conf

        name = encoder.inverse_transform(model.predict([emb]))[0]
        return name, 1.0


def redact_source(source):
    if not isinstance(source, str):
        return repr(source)
    return re.sub(r"(://[^:/@]+:)[^@]+@", r"\1***@", source)


def normalize_video_source(source):
    if isinstance(source, int):
        return source
    if source is None:
        raise ValueError("Camera source is not set")
    if isinstance(source, str):
        source = source.strip()
        if not source:
            raise ValueError("Camera source is empty")
        if source.isdigit():
            return int(source)
        return source
    raise ValueError(f"Unsupported camera source type: {type(source)!r}")


def open_video_source(source):
    source = normalize_video_source(source)
    cam_index = source if isinstance(source, int) else None

    if cam_index is not None:
        # On Windows, one backend can open but fail to read.
        # Probe multiple backends and keep the first that returns frames.
        candidates = []
        if os.name == "nt":
            candidates = [cv.CAP_DSHOW, cv.CAP_MSMF, None]
        else:
            candidates = [None]

        for backend in candidates:
            if backend is None:
                cap = cv.VideoCapture(cam_index)
            else:
                cap = cv.VideoCapture(cam_index, backend)

            if not cap.isOpened():
                cap.release()
                continue

            ok = False
            for _ in range(10):
                grabbed, frame = cap.read()
                if grabbed and frame is not None and frame.size > 0:
                    ok = True
                    break
                time.sleep(0.02)

            if ok:
                return cap

            cap.release()

        # Last fallback.
        return cv.VideoCapture(cam_index)

    os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", RTSP_FFMPEG_OPTIONS)
    cap = cv.VideoCapture(source, cv.CAP_FFMPEG)
    if not cap.isOpened():
        cap = cv.VideoCapture(source)
    return cap


def build_rtsp_source(host, username="", password="", port=554, path="Streaming/Channels/101"):
    host = (host or "").strip()
    if not host:
        raise ValueError("Host/IP is required to build RTSP URL")

    try:
        port_int = int(port)
    except (TypeError, ValueError):
        port_int = 554

    path_clean = (path or "").strip().lstrip("/")
    if not path_clean:
        path_clean = "Streaming/Channels/101"

    user = quote((username or "").strip(), safe="")
    pwd = quote((password or "").strip(), safe="")

    if user and pwd:
        auth = f"{user}:{pwd}@"
    elif user:
        auth = f"{user}@"
    elif pwd:
        auth = f":{pwd}@"
    else:
        auth = ""

    return f"rtsp://{auth}{host}:{port_int}/{path_clean}"


def sanitize_camera_id(value):
    raw = str(value or "").strip().lower()
    safe = re.sub(r"[^a-z0-9_-]+", "-", raw).strip("-")
    return safe or "camera"


def sanitize_camera_role(value):
    raw = str(value or "").strip().lower()
    if raw in ("exit", "out", "checkout"):
        return "exit"
    if raw in ("entry", "in", "checkin"):
        return "entry"
    if raw in ("general", "both", "any", ""):
        return "general"
    return "general"


def ensure_unique_camera_id(base_id):
    base_id = sanitize_camera_id(base_id)
    if base_id not in workers:
        return base_id
    idx = 2
    while f"{base_id}-{idx}" in workers:
        idx += 1
    return f"{base_id}-{idx}"


def ensure_attendance_csv():
    os.makedirs("logs", exist_ok=True)
    if not os.path.exists(ATTENDANCE_CSV_PATH):
        with open(ATTENDANCE_CSV_PATH, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(CSV_HEADERS)


def flush_attendance_csv():
    os.makedirs(os.path.dirname(ATTENDANCE_CSV_PATH) or ".", exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(
        prefix="attendance_",
        suffix=".csv",
        dir=os.path.dirname(ATTENDANCE_CSV_PATH) or ".",
    )
    try:
        with os.fdopen(fd, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_HEADERS)
            writer.writeheader()
            for row in attendance_rows:
                writer.writerow(row)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, ATTENDANCE_CSV_PATH)
    except Exception:
        logger.exception("Failed to flush attendance CSV atomically")
        try:
            os.remove(tmp_path)
        except OSError:
            pass


def load_attendance_cache():
    ensure_attendance_csv()
    with attendance_lock:
        attendance_rows.clear()
        active_sessions.clear()
        last_seen_by_name.clear()

        today_str = datetime.datetime.now().strftime("%Y-%m-%d")
        today_events = {}
        needs_rewrite = False

        with open(ATTENDANCE_CSV_PATH, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames != CSV_HEADERS:
                needs_rewrite = True

            for row in reader:
                date_val = (row.get("date") or "").strip()
                name_val = (row.get("emp name") or "").strip()
                time_val = (row.get("time") or "").strip()
                checkin_val = (row.get("checkin") or "").strip()
                checkout_val = ((row.get("checkout") or "").strip() or (row.get("lastcheckout") or "").strip())
                if not date_val or not name_val:
                    continue

                event_times = []
                if time_val:
                    event_times.append(time_val)
                else:
                    if checkin_val:
                        event_times.append(checkin_val)
                    if checkout_val:
                        event_times.append(checkout_val)

                for event_time in event_times:
                    attendance_rows.append({"date": date_val, "emp name": name_val, "time": event_time})
                    if date_val != today_str:
                        continue
                    try:
                        event_dt = datetime.datetime.strptime(f"{date_val} {event_time}", "%Y-%m-%d %H:%M:%S")
                    except ValueError:
                        event_dt = datetime.datetime.now()
                    today_events.setdefault(name_val, []).append(event_dt)

        for name_val, events in today_events.items():
            if len(events) % 2 == 0:
                continue
            last_seen = events[-1]
            active_sessions[name_val] = {"last_seen": last_seen}
            last_seen_by_name[name_val] = last_seen

        if needs_rewrite:
            flush_attendance_csv()


def update_attendance_csv(emp_name, timestamp_str):
    try:
        dt_obj = datetime.datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S")
    except ValueError:
        logger.warning("Skipping attendance update due to invalid timestamp: %s", timestamp_str)
        return

    date_str = dt_obj.strftime("%Y-%m-%d")
    time_str = dt_obj.strftime("%H:%M:%S")

    with attendance_lock:
        last_seen = last_seen_by_name.get(emp_name)
        if last_seen is not None:
            gap_seconds = (dt_obj - last_seen).total_seconds()
            if gap_seconds < EPISODE_GAP_SECONDS:
                last_seen_by_name[emp_name] = dt_obj
                if emp_name in active_sessions:
                    active_sessions[emp_name]["last_seen"] = dt_obj
                return

        last_seen_by_name[emp_name] = dt_obj
        session = active_sessions.get(emp_name)
        if session is None:
            attendance_rows.append({"date": date_str, "emp name": emp_name, "time": time_str})
            active_sessions[emp_name] = {"last_seen": dt_obj}
            flush_attendance_csv()
            return

        attendance_rows.append({"date": date_str, "emp name": emp_name, "time": time_str})
        flush_attendance_csv()
        active_sessions.pop(emp_name, None)


def _sanitize_visitor_label(value):
    return re.sub(r"[^a-z0-9_-]+", "_", str(value or "").strip().lower()).strip("_")


def _is_visitor_label(emp_name):
    raw = str(emp_name or "").strip()
    if not raw:
        return False
    lowered = raw.lower()
    if any(lowered.startswith(prefix) for prefix in VISITOR_AUTO_PREFIXES):
        return True
    if _is_registered_visitor(raw):
        return True
    label = _sanitize_visitor_label(raw)
    if not label:
        return False
    for candidate in {raw, label}:
        candidate_label = _sanitize_visitor_label(candidate)
        if not candidate_label:
            continue
        if os.path.isdir(os.path.join(VISITOR_DATA_ROOT, candidate_label)):
            return True
    return False


def _load_visitor_registry_entries():
    if not os.path.exists(VISITOR_REGISTRY_PATH):
        return set(), {}
    try:
        with open(VISITOR_REGISTRY_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        logger.exception("Failed to load visitor registry")
        return set(), {}

    entries = set()
    records = {}

    def add_record(item):
        if not isinstance(item, dict):
            return
        raw_name = str(item.get("name") or "").strip()
        raw_label = str(item.get("label") or "").strip()
        keys = set()
        for raw in (raw_name, raw_label):
            if not raw:
                continue
            lowered = raw.lower()
            keys.add(lowered)
            keys.add(_sanitize_visitor_label(raw))
            keys.add(lowered.replace(" ", "_"))
            keys.add(lowered.replace("_", " "))
        for key in keys:
            if not key:
                continue
            entries.add(key)
            records[key] = item

    if isinstance(data, list):
        for item in data:
            if isinstance(item, str):
                key = item.strip().lower()
                if key:
                    entries.add(key)
            else:
                add_record(item)
    elif isinstance(data, dict) and isinstance(data.get("visitors"), list):
        for item in data["visitors"]:
            if isinstance(item, str):
                key = item.strip().lower()
                if key:
                    entries.add(key)
            else:
                add_record(item)
    return entries, records


def _get_visitor_registry_entries():
    now_ts = time.time()
    if now_ts - visitor_registry_cache["loaded_at"] > 5:
        entries, records = _load_visitor_registry_entries()
        visitor_registry_cache["entries"] = entries
        visitor_registry_cache["records"] = records
        visitor_registry_cache["loaded_at"] = now_ts
    return visitor_registry_cache["entries"]


def _get_visitor_registry_record(emp_name):
    key = str(emp_name or "").strip().lower()
    if not key:
        return None
    _get_visitor_registry_entries()
    records = visitor_registry_cache.get("records") or {}
    if key in records:
        return records.get(key)
    alt = _sanitize_visitor_label(key)
    if alt in records:
        return records.get(alt)
    alt = key.replace("_", " ")
    if alt in records:
        return records.get(alt)
    alt = key.replace(" ", "_")
    if alt in records:
        return records.get(alt)
    return None


def _is_registered_visitor(emp_name):
    key = str(emp_name or "").strip().lower()
    if not key:
        return False
    entries = _get_visitor_registry_entries()
    return key in entries


def _load_visitors():
    if not os.path.exists(VISITOR_DB_PATH):
        return []
    try:
        with open(VISITOR_DB_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        logger.exception("Failed to load visitor passes")
        return []
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and isinstance(data.get("visitors"), list):
        return data["visitors"]
    return []


def _save_visitors(visitors):
    os.makedirs(os.path.dirname(VISITOR_DB_PATH) or ".", exist_ok=True)
    with open(VISITOR_DB_PATH, "w", encoding="utf-8") as f:
        json.dump(visitors, f, indent=2)


def _generate_visitor_pass_id(visitors):
    existing = {str(v.get("id") or "").strip().lower() for v in visitors}
    base = datetime.datetime.now().strftime("VIS%Y%m%d%H%M%S")
    candidate = base
    counter = 1
    while candidate.lower() in existing:
        counter += 1
        candidate = f"{base}-{counter}"
    return candidate


def _auto_issue_visitor_pass(emp_name, timestamp_str, camera_id=None):
    if not _is_visitor_label(emp_name):
        return

    try:
        dt_obj = datetime.datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S")
    except ValueError:
        return
    name_key = str(emp_name or "").strip().lower()
    if not name_key:
        return

    with visitor_lock:
        last_seen = last_visitor_pass_by_name.get(name_key)
        if last_seen is not None:
            gap_seconds = (dt_obj - last_seen).total_seconds()
            if gap_seconds < EPISODE_GAP_SECONDS:
                last_visitor_pass_by_name[name_key] = dt_obj
                return

        visitors = _load_visitors()
        # If there is already an active pass for this visitor, do not open a new one.
        for visitor in visitors:
            if not isinstance(visitor, dict):
                continue
            vname = str(visitor.get("name") or "").strip().lower()
            if vname == name_key:
                status = str(visitor.get("status") or "").strip().lower()
                time_out = str(visitor.get("time_out") or "").strip()
                if status != "closed" and not time_out:
                    last_visitor_pass_by_name[name_key] = dt_obj
                    return

        pass_id = _generate_visitor_pass_id(visitors)
        valid_until = (dt_obj + datetime.timedelta(hours=VISITOR_AUTO_VALID_HOURS)).replace(microsecond=0)
        registry = _get_visitor_registry_record(emp_name) or {}
        company = str(registry.get("company") or "").strip()
        contact = str(registry.get("contact") or "").strip()
        id_proof = str(registry.get("id_proof") or "").strip()
        laptop_no = str(registry.get("laptop_no") or "").strip()
        mobile_qty = str(registry.get("mobile_qty") or "").strip()
        storage_details = str(registry.get("storage_details") or "").strip()
        person_to_visit = str(registry.get("person_to_visit") or "").strip()
        department = str(registry.get("department") or "").strip()
        purpose = str(registry.get("purpose") or "").strip()
        security_issued_by = str(registry.get("security_issued_by") or "").strip()
        employee_visited = str(registry.get("employee_visited") or "").strip() or person_to_visit

        visitors.append(
            {
                "id": pass_id,
                "name": str(emp_name),
                "company": company,
                "contact": contact,
                "id_proof": id_proof,
                "laptop_no": laptop_no,
                "mobile_qty": mobile_qty,
                "storage_details": storage_details,
                "purpose": purpose,
                "person_to_visit": person_to_visit,
                "department": department,
                "created_at": dt_obj.replace(microsecond=0).isoformat(),
                "valid_until": valid_until.isoformat(),
                "status": "active",
                "time_in": dt_obj.strftime("%H:%M:%S"),
                "time_out": "",
                "security_issued_by": security_issued_by,
                "employee_visited": employee_visited,
                "auto_issued": True,
                "source_camera": str(camera_id or ""),
            }
        )
        _save_visitors(visitors)
        last_visitor_pass_by_name[name_key] = dt_obj


def _close_visitor_pass_on_exit(emp_name, timestamp_str, camera_id=None):
    raw_name = str(emp_name or "").strip()
    if not raw_name:
        return False

    try:
        dt_obj = datetime.datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S")
    except ValueError:
        return False

    key = raw_name.lower()
    with visitor_lock:
        visitors = _load_visitors()
        if not visitors:
            return False

        matches = []
        for visitor in visitors:
            vid = str(visitor.get("id") or "").strip().lower()
            vname = str(visitor.get("name") or "").strip().lower()
            if key and (key == vid or key == vname):
                matches.append(visitor)

        if not matches:
            return False

        active = [
            v for v in matches
            if str(v.get("status") or "").strip().lower() != "closed"
            and not str(v.get("time_out") or "").strip()
        ]
        if not active:
            return False

        def sort_key(v):
            raw_created = v.get("created_at") or ""
            try:
                return datetime.datetime.fromisoformat(raw_created)
            except ValueError:
                return datetime.datetime.min

        active.sort(key=sort_key, reverse=True)
        target = active[0]
        target["time_out"] = dt_obj.strftime("%H:%M:%S")
        target["status"] = "closed"
        target["exit_camera"] = str(camera_id or "")
        target["closed_at"] = dt_obj.replace(microsecond=0).isoformat()
        target["exit_detected_at"] = dt_obj.replace(microsecond=0).isoformat()
        _save_visitors(visitors)
        return True


def initialize_shared_models():
    global embedder, model, encoder

    with model_load_lock:
        if embedder is None:
            start = time.time()
            embedder = FaceNet()
            logger.info("FaceNet loaded in %.3fs", time.time() - start)

        if model is None:
            start = time.time()
            with open("model/svm_model_160x160.pkl", "rb") as f:
                model = pickle.load(f)
            logger.info("SVM model loaded in %.3fs", time.time() - start)

        if encoder is None:
            start = time.time()
            with open("model/label_encoder.pkl", "rb") as f:
                encoder = pickle.load(f)
            logger.info("Label encoder loaded in %.3fs", time.time() - start)


def add_camera(
    camera_id=None,
    name="",
    source="",
    host="",
    username="",
    password="",
    port=554,
    path="Streaming/Channels/101",
    role="general",
    auto_start=False,
):
    source = (source or "").strip()
    if source:
        source_cfg = normalize_video_source(source)
    else:
        source_cfg = normalize_video_source(build_rtsp_source(host, username, password, port, path))

    raw_id = camera_id or name or "cam"
    role_clean = sanitize_camera_role(role)
    with workers_lock:
        cam_id = ensure_unique_camera_id(raw_id)
        worker = CameraWorker(
            camera_id=cam_id,
            source_cfg=source_cfg,
            display_name=name or cam_id,
            show_window=False,
            role=role_clean,
        )
        workers[cam_id] = {
            "worker": worker,
            "camera_id": cam_id,
            "name": worker.display_name,
            "source": str(source_cfg),
            "host": (host or "").strip(),
            "port": int(port) if str(port).isdigit() else 554,
            "path": (path or "").strip(),
            "username": (username or "").strip(),
            "has_password": bool(password),
            "role": role_clean,
            "created_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }

    if auto_start:
        worker.start()

    return get_camera_payload(cam_id)


def remove_camera(camera_id):
    cam_id = sanitize_camera_id(camera_id)
    with workers_lock:
        item = workers.get(cam_id)
        if item is None:
            raise KeyError(f"Camera not found: {cam_id}")
        worker = item["worker"]
        worker.stop()
        workers.pop(cam_id, None)


def start_camera(camera_id):
    cam_id = sanitize_camera_id(camera_id)
    with workers_lock:
        item = workers.get(cam_id)
        if item is None:
            raise KeyError(f"Camera not found: {cam_id}")
        worker = item["worker"]
    worker.start()
    return worker.build_status_payload()


def stop_camera(camera_id):
    cam_id = sanitize_camera_id(camera_id)
    with workers_lock:
        item = workers.get(cam_id)
        if item is None:
            raise KeyError(f"Camera not found: {cam_id}")
        worker = item["worker"]
    worker.stop()
    return worker.build_status_payload()


def get_camera_worker(camera_id):
    cam_id = sanitize_camera_id(camera_id)
    with workers_lock:
        item = workers.get(cam_id)
        if item is None:
            return None
        return item["worker"]


def get_camera_payload(camera_id):
    cam_id = sanitize_camera_id(camera_id)
    with workers_lock:
        item = workers.get(cam_id)
        if item is None:
            return None
        worker = item["worker"]
    base = {
        "camera_id": item["camera_id"],
        "name": item["name"],
        "source": redact_source(item["source"]),
        "host": item["host"],
        "port": item["port"],
        "path": item["path"],
        "username": item["username"],
        "has_password": item["has_password"],
        "role": item.get("role", "general"),
        "created_at": item["created_at"],
    }
    base.update(worker.build_status_payload())
    return base


def list_cameras():
    with workers_lock:
        ids = sorted(workers.keys())
    payload = []
    for cam_id in ids:
        item = get_camera_payload(cam_id)
        if item is not None:
            payload.append(item)
    return payload


def stop_all_cameras():
    with workers_lock:
        ids = list(workers.keys())
    for cam_id in ids:
        try:
            stop_camera(cam_id)
        except Exception:
            logger.exception("Failed to stop camera %s", cam_id)


def generate_mjpeg_stream(camera_id):
    worker = get_camera_worker(camera_id)
    if worker is None:
        raise KeyError(f"Camera not found: {camera_id}")
    return worker.generate_mjpeg_stream()


def build_status_payload(camera_id=None):
    if camera_id:
        item = get_camera_payload(camera_id)
        if item is None:
            raise KeyError(f"Camera not found: {camera_id}")
        return item

    cameras = list_cameras()
    running_count = sum(1 for c in cameras if c.get("running"))
    with attendance_lock:
        open_sessions = list(active_sessions.keys())
        row_count = len(attendance_rows)
    return {
        "total_cameras": len(cameras),
        "running_cameras": running_count,
        "open_sessions": open_sessions,
        "attendance_rows": row_count,
    }


def get_attendance_rows(limit=100):
    with attendance_lock:
        rows = list(attendance_rows)
    if isinstance(limit, int) and limit > 0:
        return rows[-limit:]
    return rows


# Backward-compatible wrappers for single-camera callers

def start_workers(source_override=None, show_window=False):
    source_cfg = normalize_video_source(source_override if source_override is not None else RTSP_URL)
    with workers_lock:
        item = workers.get("default")
    if item is None:
        add_camera(camera_id="default", name="default", source=str(source_cfg), auto_start=False)
    worker = get_camera_worker("default")
    worker.show_window = bool(show_window)
    worker.source_cfg = source_cfg
    worker.start()
    return source_cfg


def stop_workers():
    stop_all_cameras()


def workers_alive():
    with workers_lock:
        current = list(workers.values())
    return any(item["worker"].alive() for item in current)


def bootstrap():
    with runtime_bootstrap_lock:
        load_attendance_cache()


def main():
    parser = argparse.ArgumentParser(description="Run multi-camera tracking + attendance without Flask.")
    parser.add_argument("--source", default="", help="Camera source override for default camera.")
    parser.add_argument("--camera-id", default="default", help="Camera ID for standalone run.")
    parser.add_argument("--name", default="", help="Display name for the camera.")
    parser.add_argument("--no-window", action="store_true", help="Disable OpenCV preview window.")
    args = parser.parse_args()

    bootstrap()

    source_override = args.source.strip() if isinstance(args.source, str) else args.source
    if source_override == "":
        source_override = None

    source_cfg = normalize_video_source(source_override if source_override is not None else RTSP_URL)
    cam = add_camera(
        camera_id=args.camera_id,
        name=args.name or args.camera_id,
        source=str(source_cfg),
        auto_start=False,
    )
    worker = get_camera_worker(cam["camera_id"])
    worker.show_window = not args.no_window
    worker.start()

    logger.info("Standalone started for camera_id=%s source=%s", cam["camera_id"], redact_source(str(source_cfg)))

    try:
        while worker.alive() and not worker.stop_event.is_set():
            time.sleep(MAIN_HEALTHCHECK_INTERVAL_SECONDS)
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    finally:
        stop_all_cameras()


bootstrap()


if __name__ == "__main__":
    main()
