"""A clicked point followed by TAPNext++, as a calculated observation field.

Declared on a camera's ``ModuleObs``::

    target_point = CalculatedField(
        inputs=[MyRobot.wrist_camera.camera.rgb_image],
        processor=TapNextPointProcessor,
        output=TrackedPointData(),
        conf=str(service_config_path("point_tracker")),
    )

and started with ``controller.start_service("wrist_camera", "target_point",
point=(x, y))``. Until then, and after ``stop_service``, the value is all
zeros. The UI draws it as a small box on the image with TAPNext's visibility.
"""

import logging
from typing import Optional, Tuple, Type

import cv2
import numpy as np
import requests

from solarian.core.data import BoundingBoxData
from solarian.core.fisheye import Box, lit_box
from solarian.services import ServiceProcessor

from solarian_tapnext.client import TapNextClient
from solarian_tapnext.model import Tracked

logger = logging.getLogger(__name__)


class TrackedPointData(BoundingBoxData):
    """``(x0, y0, x1, y1, visibility %)``: a square centred on the tracked point,
    in the source image's pixels. All zeros = no point."""

    shape: Optional[Tuple[int, ...]] = (5,)
    dtype: Type = np.int32

    def overlay_text(self) -> str | None:
        if not self.value[:4].any():
            return None
        visibility = int(self.value[4])
        return f"TAPNext {visibility}%" if visibility >= 50 else f"occluded {visibility}%"


class TapNextPointProcessor(ServiceProcessor):
    """Seeds TAPNext++ on ``params["point"]`` and follows that pixel frame to frame.

    The tracker sees ``crop`` of each frame, scaled so its long edge is
    ``resize``; the click goes in and the point comes out in the frame's own
    pixels. An occluded point keeps TAPNext's estimate of where it is.
    """

    def __init__(self, inputs, output, conf: str | None = None):
        super().__init__(inputs, output, conf)
        cfg = self.cfg or {}
        self.server_url = str(cfg.get("server_url", "http://localhost:8767"))
        crop = cfg.get("crop", "auto")
        # `auto` is found again on each click's frame and kept until the next.
        self.auto_crop = crop == "auto"
        self.crop: Box | None = None if crop in ("auto", "none", None) else tuple(int(v) for v in crop)
        resize = cfg.get("resize", 512)
        self.resize = int(resize) if resize is not None else None
        self.half_box = int(cfg.get("box_px", 24)) // 2
        self.jpeg_quality = int(cfg.get("jpeg_quality", 90))
        self.timeout = float(cfg.get("timeout", 5.0))
        self.init_timeout = float(cfg.get("init_timeout", 60.0))
        # Built on the worker thread: a requests session does not survive
        # the fork that puts a robot-side processor in its module process.
        self._client: TapNextClient | None = None
        #: The point in frame pixels and its visibility, None until seeded.
        self._point: np.ndarray | None = None
        self._visibility = 0.0
        self._scale = 1.0

    def _ensure_client(self) -> TapNextClient:
        if self._client is None:
            self._client = TapNextClient(
                self.server_url, timeout=self.timeout, jpeg_quality=self.jpeg_quality, token=self.token
            )
        return self._client

    def _view(self, shape: tuple[int, ...]) -> Box:
        h, w = shape[:2]
        return self.crop or (0, 0, w, h)

    def _to_send(self, frame_rgb: np.ndarray) -> np.ndarray:
        """The crop as BGR, its long edge at ``resize``; sets the scale it was resized by."""
        x0, y0, x1, y1 = self._view(frame_rgb.shape)
        bgr = cv2.cvtColor(np.asarray(frame_rgb[y0:y1, x0:x1], dtype=np.uint8), cv2.COLOR_RGB2BGR)
        h, w = bgr.shape[:2]
        self._scale = 1.0 if self.resize is None else self.resize / max(h, w)
        if self._scale != 1.0:
            bgr = cv2.resize(bgr, (max(1, round(w * self._scale)), max(1, round(h * self._scale))),
                             interpolation=cv2.INTER_AREA)
        return bgr

    def _to_sent(self, point, shape: tuple[int, ...]) -> np.ndarray:
        """A frame-pixel point in the sent image's pixels, held inside the crop."""
        x0, y0, x1, y1 = self._view(shape)
        x = min(max(float(point[0]), x0), x1 - 1)
        y = min(max(float(point[1]), y0), y1 - 1)
        return np.array([[(x - x0) * self._scale, (y - y0) * self._scale]], np.float32)

    def _keep(self, tracked: Tracked, shape: tuple[int, ...]) -> np.ndarray:
        x0, y0 = self._view(shape)[:2]
        self._point = tracked.points[0] / self._scale + (x0, y0)
        self._visibility = float(tracked.visibility[0])
        return self._output(shape)

    def _output(self, shape: tuple[int, ...]) -> np.ndarray:
        h, w = shape[:2]
        x, y = self._point
        r = self.half_box
        box = np.rint(np.clip([x - r, y - r, x + r, y + r], 0, [w, h, w, h]))
        return np.array([*box, round(100 * self._visibility)], dtype=np.int32)

    def on_start(self, frame, params: dict):
        client = self._ensure_client()
        if self.auto_crop:
            self.crop = lit_box([frame])
            logger.info("%s: TAPNext sees %s", self.label,
                        f"the frame cropped to {self.crop}" if self.crop else "the whole frame")
        bgr = self._to_send(frame)
        point = params["point"]
        tracked = client.init(bgr, self._to_sent(point, frame.shape), timeout=self.init_timeout)
        out = self._keep(tracked, frame.shape)
        logger.info("%s: following %s (%.0f ms)", self.label, tuple(point), client.last_inference_ms)
        return out

    def on_frame(self, frame):
        client = self._ensure_client()
        bgr = self._to_send(frame)
        try:
            tracked = client.track(bgr)
        except requests.HTTPError as exc:
            # 409: the server lost the session (restarted, or another client
            # seeded it). Re-seed where the point was last seen.
            if exc.response is None or exc.response.status_code != 409 or self._point is None:
                raise
            logger.warning("%s: tracker has no session; re-seeding at %s", self.label, tuple(self._point))
            tracked = client.init(bgr, self._to_sent(self._point, frame.shape), timeout=self.init_timeout)
        return self._keep(tracked, frame.shape)

    def on_stop(self) -> None:
        self._point = None
        self._ensure_client().reset()
