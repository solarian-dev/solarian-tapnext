"""TAPNext++ online: a set of points seeded on one frame and followed through the next ones.

TAPNext is recurrent: each frame updates a state that carries every query,
so the points to follow are fixed when tracking starts, and a new set is a
new :meth:`PointTracker.seed`. Frames are squashed to a square
``resolution`` before the patch embedding; queries and tracks live in a
square 256-pixel space whatever that resolution is.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import NamedTuple, Optional

import numpy as np

from solarian.utils.models import download_missing, models_dir

logger = logging.getLogger(__name__)

#: TAPNext++ fine-tuned at 512 on PointOdyssey and Kubric-1024 (Apache 2.0).
CHECKPOINT_NAME = "tapnextpp_512.ckpt"
CHECKPOINT_URL = "https://storage.googleapis.com/gresearch/tapnextpp/tapnextpp_512.ckpt"

#: The square coordinate space of queries and tracks.
MODEL_SIZE = 256

#: The resolution tapnextpp_512 was fine-tuned at. At 256 it is ~3x faster
#: (~25 ms against ~70 ms on an RTX 3090) but reports about half of each
#: frame's motion, so the tracks lag the scene.
DEFAULT_RESOLUTION = 512

#: TAPNext's patch size; ``resolution`` must be a multiple of it.
PATCH = 8


class Tracked(NamedTuple):
    """Each query point on the latest frame."""

    #: ``(N, 2)`` float32 ``[x, y]`` in the pixels of the frame given.
    points: np.ndarray
    #: ``(N,)`` bool.
    visible: np.ndarray
    #: ``(N,)`` float32, the model's probability that the point is visible.
    visibility: np.ndarray


def resolve_checkpoint(checkpoint: Optional[str]) -> Path:
    """The weights to load, downloaded first when it is the published file and missing.

    A bare filename lives in ``models_dir("tapnext")``; a path with a
    directory is used as given.
    """
    path = Path(checkpoint or CHECKPOINT_NAME).expanduser()
    if len(path.parts) == 1:
        path = models_dir("tapnext", path.name)
    if path.is_file():
        return path
    if path.name != CHECKPOINT_NAME:
        raise FileNotFoundError(f"{path} does not exist, and only the published {CHECKPOINT_NAME} is downloaded")
    return download_missing(path, CHECKPOINT_URL, "solarian-tapnext")


class PointTracker:
    """Not thread-safe: the recurrent state is the tracker's, so one caller at a time."""

    def __init__(self, checkpoint: Optional[str] = None, resolution: int = DEFAULT_RESOLUTION):
        if resolution <= 0 or resolution % PATCH:
            raise ValueError(f"resolution must be a positive multiple of {PATCH}, got {resolution}")
        import torch

        from solarian_tapnext.tapnext.tapnext_torch import TAPNext

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
        self.resolution = int(resolution)
        self.checkpoint = resolve_checkpoint(checkpoint)
        t0 = time.perf_counter()
        model = TAPNext(image_size=(MODEL_SIZE, MODEL_SIZE))
        state = torch.load(self.checkpoint, map_location="cpu", weights_only=True)
        state = state.get("state_dict", state)
        model.load_state_dict({k.removeprefix("tapnext."): v for k, v in state.items()})
        self.model = model.to(self.device).eval()
        self._state = None
        logger.info("TAPNext++ %s ready on %s at %d in %.1f s", self.checkpoint.name, self.device,
                    self.resolution, time.perf_counter() - t0)

    def _video(self, frame_rgb: np.ndarray):
        """``[1, 1, S, S, 3]`` in [-1, 1], resized on the device."""
        import torch
        import torch.nn.functional as F

        t = torch.from_numpy(np.ascontiguousarray(frame_rgb)).to(self.device)
        t = t.permute(2, 0, 1)[None].float()
        t = F.interpolate(t, size=(self.resolution, self.resolution), mode="bilinear", align_corners=False)
        return (t / 127.5 - 1.0).permute(0, 2, 3, 1)[None]

    def _step(self, frame_rgb: np.ndarray, query=None) -> Tracked:
        import torch

        h, w = frame_rgb.shape[:2]
        # fp16 as the TAPNext++ demo runs it on CUDA; MPS and CPU stay in fp32.
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16, enabled=self.device.type == "cuda"):
            tracks, _, visible_logits, self._state = self.model(
                video=self._video(frame_rgb), query_points=query, state=self._state,
            )
        # [1, 1, Q, 2] as (y, x) in model space.
        yx = tracks[0, 0].float().cpu().numpy()
        points = np.column_stack([yx[:, 1] * (w / MODEL_SIZE), yx[:, 0] * (h / MODEL_SIZE)]).astype(np.float32)
        logits = visible_logits[0, 0, :, 0].float().cpu().numpy()
        return Tracked(points, logits > 0, (1.0 / (1.0 + np.exp(-logits))).astype(np.float32))

    def seed(self, frame_rgb: np.ndarray, points_xy) -> Tracked:
        """Start following ``points_xy`` (``(N, 2)`` frame pixels) from ``frame_rgb``; drops the previous set."""
        import torch

        h, w = frame_rgb.shape[:2]
        xy = np.asarray(points_xy, dtype=np.float32).reshape(-1, 2)
        query = np.zeros((len(xy), 3), np.float32)
        query[:, 1] = xy[:, 1] * (MODEL_SIZE / h)
        query[:, 2] = xy[:, 0] * (MODEL_SIZE / w)
        self._state = None
        return self._step(frame_rgb, torch.from_numpy(query).to(self.device)[None])

    def track(self, frame_rgb: np.ndarray) -> Tracked:
        if self._state is None:
            raise RuntimeError("track before seed")
        return self._step(frame_rgb)

    def reset(self) -> None:
        self._state = None

    def warm_up(self, n_points: int = 65, steps: int = 3) -> None:
        frame = np.full((self.resolution, self.resolution, 3), 96, np.uint8)
        t0 = time.perf_counter()
        self.seed(frame, np.full((n_points, 2), self.resolution / 2.0))
        for _ in range(steps):
            self.track(frame)
        self.reset()
        logger.info("TAPNext++ warmed up in %.1f s", time.perf_counter() - t0)
