"""Client for :mod:`solarian_tapnext.server`.

Frames travel as JPEG bytes: a camera-rate stream from a robot over WiFi
cannot afford raw pixels.
"""

import cv2
import numpy as np

from solarian.services import ServiceClient
from solarian_tapnext.model import Tracked


def encode_jpeg(frame_bgr: np.ndarray, quality: int = 90) -> np.ndarray:
    ok, buf = cv2.imencode(".jpg", frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        raise RuntimeError("cv2.imencode('.jpg') failed")
    return buf.reshape(-1)


class TapNextClient(ServiceClient):
    """``init`` / ``track`` / ``reset`` on a BGR frame; points in that frame's pixels."""

    def __init__(self, server_url: str, timeout: float = 5.0, jpeg_quality: int = 90, token: str | None = None):
        super().__init__(server_url, timeout=timeout, token=token)
        self.jpeg_quality = jpeg_quality
        #: Model time of the last call as the server measured it, in ms.
        self.last_inference_ms = 0.0
        #: What the last ``init`` started; ``track`` names it so a re-seed by another client is a 409.
        self.session: str | None = None

    def _tracked(self, reply: dict) -> Tracked:
        self.last_inference_ms = float(reply["inference_ms"])
        return Tracked(
            np.asarray(reply["points"], dtype=np.float32).reshape(-1, 2),
            np.asarray(reply["visible"], dtype=bool).reshape(-1),
            np.asarray(reply["visibility"], dtype=np.float32).reshape(-1),
        )

    def init(self, frame_bgr: np.ndarray, points, timeout: float | None = None) -> Tracked:
        """Start following ``points`` (``(N, 2)`` pixels) from this frame."""
        reply = self.post(
            "/init",
            {
                "image_jpeg": encode_jpeg(frame_bgr, self.jpeg_quality),
                "points": np.asarray(points, dtype=np.float32).reshape(-1, 2),
            },
            timeout=timeout,
        )
        self.session = reply["session"]
        return self._tracked(reply)

    def track(self, frame_bgr: np.ndarray) -> Tracked:
        payload = {"image_jpeg": encode_jpeg(frame_bgr, self.jpeg_quality)}
        if self.session is not None:
            payload["session"] = self.session
        return self._tracked(self.post("/track", payload))

    def reset(self) -> None:
        self.post("/reset", {})
        self.session = None
