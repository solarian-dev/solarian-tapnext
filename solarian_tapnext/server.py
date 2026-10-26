"""TAPNext++ point tracker server: ``solarian run-service tapnext``.

One :class:`~solarian_tapnext.model.PointTracker`, one tracking session: the
model's recurrent state is the session, so every call runs on one inference
thread. Endpoints::

    POST /init  {image_jpeg, points}   -> {points, visible, visibility, inference_ms, session}
    POST /track {image_jpeg, session?} -> {points, visible, visibility, inference_ms}
    POST /reset {}                     -> {status}

``points`` go in as ``[[x, y], ...]`` pixels of the frame sent and come back
in the same order and pixels, one row per query. Each ``/init`` starts a new
``session``; a ``/track`` that names an older one is a 409, as another client
re-seeded the tracker and the points are its own.
"""

import argparse
import logging
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, Optional

import cv2
import numpy as np

from solarian.services import ServiceServer
from solarian.utils.wire import decode_arrays, encode_arrays
from solarian_tapnext.model import DEFAULT_RESOLUTION, PATCH, PointTracker, Tracked

logger = logging.getLogger(__name__)

#: Queries one session takes; each is a token next to the frame's patches.
MAX_POINTS = 1024


class NoTrackingSession(Exception):
    """``/track`` before ``/init``, after ``/reset``, or naming a session another ``/init`` replaced."""


def decode_jpeg(buffer: np.ndarray) -> np.ndarray:
    frame = cv2.imdecode(np.asarray(buffer, dtype=np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("cv2.imdecode failed")
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


class TapNextServer(ServiceServer):
    title = "TAPNext++"
    default_port = 8767

    def __init__(self, checkpoint: Optional[str] = None, resolution: int = DEFAULT_RESOLUTION):
        super().__init__(checkpoint=checkpoint, resolution=resolution)
        self.checkpoint = checkpoint
        self.resolution = resolution
        self.tracker: PointTracker | None = None
        self.session = ""
        self.tracking_active = False
        self._inference_thread = ThreadPoolExecutor(max_workers=1, thread_name_prefix="tapnext")

    def load(self) -> None:
        self.tracker = self._inference_thread.submit(PointTracker, self.checkpoint, self.resolution).result()
        self.checkpoint = str(self.tracker.checkpoint)

    def warm_up(self) -> None:
        self._inference_thread.submit(self.tracker.warm_up).result()

    @property
    def info(self) -> Dict[str, Any]:
        return {
            "checkpoint": self.checkpoint,
            "resolution": self.resolution,
            "device": None if self.tracker is None else str(self.tracker.device),
            "tracking_active": self.tracking_active,
            "max_points": MAX_POINTS,
        }

    @staticmethod
    def _reply(tracked: Tracked, t0: float) -> Dict[str, Any]:
        return {
            "points": tracked.points,
            "visible": tracked.visible,
            "visibility": tracked.visibility,
            "inference_ms": (time.perf_counter() - t0) * 1000.0,
        }

    def init(self, frame: np.ndarray, points: np.ndarray) -> Dict[str, Any]:
        return self._inference_thread.submit(self._init, frame, points).result()

    def _init(self, frame: np.ndarray, points: np.ndarray) -> Dict[str, Any]:
        t0 = time.perf_counter()
        try:
            tracked = self.tracker.seed(frame, points)
        except Exception:
            self.tracking_active = False
            raise
        self.tracking_active = True
        self.session = uuid.uuid4().hex
        return {**self._reply(tracked, t0), "session": self.session}

    def track(self, frame: np.ndarray, session: str | None = None) -> Dict[str, Any]:
        return self._inference_thread.submit(self._track, frame, session).result()

    def _track(self, frame: np.ndarray, session: str | None) -> Dict[str, Any]:
        if not self.tracking_active:
            raise NoTrackingSession("No active tracking session; call /init first")
        if session is not None and session != self.session:
            raise NoTrackingSession("Another client re-seeded the tracker; call /init again")
        t0 = time.perf_counter()
        return self._reply(self.tracker.track(frame), t0)

    def reset(self) -> None:
        def stop():
            self.tracking_active = False
            self.tracker.reset()

        self._inference_thread.submit(stop).result()

    def add_routes(self, app) -> None:
        from fastapi import HTTPException

        server = self

        # Sync handlers: FastAPI runs them on its thread pool, so a frame in
        # flight does not block /health.
        def frame_of(payload: dict) -> np.ndarray:
            try:
                return decode_jpeg(payload["image_jpeg"])
            except (KeyError, ValueError) as exc:
                raise HTTPException(status_code=400, detail=f"bad image_jpeg: {exc}")

        @app.post("/init")
        def init(body: dict):
            payload = decode_arrays(body)
            try:
                points = np.asarray(payload["points"], dtype=np.float32).reshape(-1, 2)
            except (KeyError, TypeError, ValueError) as exc:
                raise HTTPException(status_code=400, detail=f"`points` must be [[x, y], ...]: {exc}")
            if not 0 < len(points) <= MAX_POINTS:
                raise HTTPException(status_code=400, detail=f"1 to {MAX_POINTS} points, got {len(points)}")
            frame = frame_of(payload)
            try:
                result = server.init(frame, points)
            except Exception as exc:
                logger.exception("init failed")
                raise HTTPException(status_code=500, detail=f"init failed: {exc!r}")
            return encode_arrays(result)

        @app.post("/track")
        def track(body: dict):
            payload = decode_arrays(body)
            frame = frame_of(payload)
            try:
                result = server.track(frame, payload.get("session"))
            except NoTrackingSession as exc:
                raise HTTPException(status_code=409, detail=str(exc))
            except Exception as exc:
                logger.exception("track failed")
                raise HTTPException(status_code=500, detail=f"track failed: {exc!r}")
            return encode_arrays(result)

        @app.post("/reset")
        def reset(body: dict):
            server.reset()
            return {"status": "success"}

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--checkpoint",
            default=None,
            help="The weights: a path, or a filename in models/tapnext of the project "
            "(default the published tapnextpp_512.ckpt, downloaded when missing)",
        )
        parser.add_argument(
            "--resolution",
            type=int,
            default=DEFAULT_RESOLUTION,
            help=f"Square input the frame is resized to, a multiple of {PATCH}. "
            f"Below the checkpoint's own (default {DEFAULT_RESOLUTION}) the tracks lag the motion.",
        )

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "TapNextServer":
        if args.resolution <= 0 or args.resolution % PATCH:
            raise SystemExit(f"--resolution must be a positive multiple of {PATCH}, got {args.resolution}")
        return cls(checkpoint=args.checkpoint, resolution=args.resolution)


def main(argv: Optional[list] = None) -> None:
    TapNextServer.main(argv)


if __name__ == "__main__":
    main()
