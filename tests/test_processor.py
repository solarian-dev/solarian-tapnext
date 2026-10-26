import time

import cv2
import numpy as np
import pytest
import requests

from solarian.core.data import ImageData

from solarian_tapnext import processor as processor_mod
from solarian_tapnext.model import Tracked
from solarian_tapnext.processor import TapNextPointProcessor, TrackedPointData

#: Sent-image pixels the fake moves the point by on each track.
STEP = np.array([5.0, 3.0], np.float32)


class FakeClient:
    """Answers in the pixels of the image it was sent, like the server."""

    def __init__(self):
        self.inits = []
        self.sent_shapes = []
        self.resets = 0
        self.last_inference_ms = 1.0
        self.visibility = 0.9
        self._point = None

    def _reply(self):
        return Tracked(self._point[None].copy(), np.array([self.visibility >= 0.5]),
                       np.array([self.visibility], np.float32))

    def init(self, frame_bgr, points, timeout=None):
        self.inits.append((frame_bgr.shape, np.asarray(points, np.float32).reshape(2).copy()))
        self._point = np.asarray(points, np.float32).reshape(2).copy()
        return self._reply()

    def track(self, frame_bgr):
        self.sent_shapes.append(frame_bgr.shape)
        self._point = self._point + STEP
        return self._reply()

    def reset(self):
        self.resets += 1


class Obs:
    """The two things a processor reads off a ModuleObs."""

    @classmethod
    def get_sensor_fields_spec(cls):
        return {}

    @classmethod
    def get_calculated_fields_spec(cls):
        return {}

    def __init__(self, image):
        self.image = image

    def get(self, name, default=None):
        return self.image


def _frame(h=400, w=800, lit=None):
    data = ImageData()
    data.value = np.zeros((h, w, 3), dtype=np.uint8)
    if lit is not None:
        cx, cy, r = lit
        cv2.circle(data.value, (cx, cy), r, (180, 180, 180), -1)
    data.timestamp = time.time()
    return data


def _wait(predicate, timeout=2.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.005)
    return False


@pytest.fixture
def proc(monkeypatch):
    fake = FakeClient()
    monkeypatch.setattr(processor_mod, "TapNextClient", lambda *a, **k: fake)
    p = TapNextPointProcessor(inputs=[ImageData()], output=TrackedPointData())
    p._input_field_names = ["image"]
    p.fake = fake
    return p


def _started(proc, point=(400, 200), **frame):
    proc.start({"point": point})
    proc(Obs(_frame(**frame)))
    assert _wait(lambda: proc._result is not None)


def test_idle_field_is_zero_and_never_calls_server(proc):
    out = proc(Obs(_frame()))
    assert out.value.tolist() == [0, 0, 0, 0, 0]
    assert proc.fake.inits == []


def test_click_goes_through_the_resize_and_the_box_is_centred_on_it(proc):
    _started(proc)
    (shape, point), = proc.fake.inits
    # The 800-wide frame is sent 512 wide.
    assert shape[:2] == (256, 512)
    assert point == pytest.approx((256.0, 128.0))
    out = proc(Obs(_frame()))
    assert out.value.dtype == np.int32
    assert out.value.tolist() == [388, 188, 412, 212, 90]


def test_tracked_motion_comes_back_in_frame_pixels(proc):
    _started(proc)
    proc(Obs(_frame()))
    assert _wait(lambda: len(proc.fake.sent_shapes) >= 1)
    assert _wait(lambda: proc._result[0][0] != 388)
    # One step of (5, 3) sent pixels is (7.8, 4.7) frame pixels at 512/800.
    assert proc._result[0].tolist() == [396, 193, 420, 217, 90]


def test_the_tracker_sees_the_lit_circle_and_the_point_comes_back_in_frame_pixels(proc):
    _started(proc, point=(450, 220), lit=(400, 200, 150))
    x0, y0, x1, y1 = proc.crop
    assert x0 < 260 and x1 > 540 and y1 <= 400
    (shape, point), = proc.fake.inits
    scale = 512 / max(x1 - x0, y1 - y0)
    assert max(shape[:2]) == 512
    assert point == pytest.approx(((450 - x0) * scale, (220 - y0) * scale), abs=1e-3)
    assert proc._result[0].tolist()[:4] == [438, 208, 462, 232]


def test_occluded_point_keeps_its_estimate_and_says_so(proc):
    proc.fake.visibility = 0.2
    _started(proc)
    value = proc._result[0]
    assert value.tolist() == [388, 188, 412, 212, 20]
    assert TrackedPointData(value=value).overlay_text() == "occluded 20%"
    assert TrackedPointData(value=np.array([1, 1, 5, 5, 87])).overlay_text() == "TAPNext 87%"
    assert TrackedPointData().overlay_text() is None


def test_lost_session_reseeds_at_the_last_point(proc):
    _started(proc)

    def no_session(frame_bgr):
        response = requests.Response()
        response.status_code = 409
        raise requests.HTTPError("409", response=response)

    proc.fake.track = no_session
    proc(Obs(_frame()))
    assert _wait(lambda: len(proc.fake.inits) == 2)
    assert proc.fake.inits[1][1] == pytest.approx((256.0, 128.0))


def test_stop_clears_the_point_and_resets_the_server(proc):
    _started(proc)
    proc.stop()
    assert proc(Obs(_frame())).value.tolist() == [0, 0, 0, 0, 0]
    assert _wait(lambda: proc.fake.resets == 1)
