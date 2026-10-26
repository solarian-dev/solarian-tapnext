import threading

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from solarian.utils.wire import encode_arrays
from solarian_tapnext.client import TapNextClient
from solarian_tapnext.model import PointTracker, Tracked
from solarian_tapnext.server import MAX_POINTS, TapNextServer


class FakeTracker:
    """Each point drifts one pixel right per frame; records the thread of every call."""

    def __init__(self):
        self.threads = []
        self.points = None
        self.error: Exception | None = None

    def _tracked(self):
        n = len(self.points)
        return Tracked(self.points.copy(), np.ones(n, bool), np.full(n, 0.9, np.float32))

    def seed(self, frame, points):
        self.threads.append(threading.get_ident())
        self.points = np.asarray(points, np.float32).copy()
        return self._tracked()

    def track(self, frame):
        self.threads.append(threading.get_ident())
        if self.error is not None:
            raise self.error
        self.points[:, 0] += 1.0
        return self._tracked()

    def reset(self):
        self.points = None


@pytest.fixture
def server():
    s = TapNextServer()
    s.tracker = FakeTracker()
    s.loaded = True
    return s


def _jpeg(shape=(48, 64, 3)):
    ok, buf = cv2.imencode(".jpg", np.zeros(shape, dtype=np.uint8))
    assert ok
    return buf.reshape(-1)


def _post(client, path, **payload):
    return client.post(path, json=encode_arrays({"image_jpeg": _jpeg(), **payload}))


class Routed:
    """The client's requests session, answered by a TestClient."""

    def __init__(self, http):
        self.http = http

    def post(self, url, data, headers, timeout):
        reply = self.http.post(url.split(":8767")[1], content=data, headers=headers)
        reply.ok = reply.is_success
        return reply


def test_tracker_calls_share_one_thread(server):
    frame = np.zeros((8, 8, 3), np.uint8)
    server.init(frame, np.array([[4, 4]], np.float32))
    caller = threading.Thread(target=server.track, args=(frame,))
    caller.start()
    caller.join()
    first, second = server.tracker.threads
    assert first == second != threading.get_ident()


def test_points_come_back_in_order_and_follow_the_frames(server):
    with TestClient(server.build_app()) as http:
        client = TapNextClient("http://gpu:8767")
        client._http = Routed(http)
        seeded = client.init(np.zeros((48, 64, 3), np.uint8), [[10, 20], [30, 5]])
        tracked = client.track(np.zeros((48, 64, 3), np.uint8))
    np.testing.assert_allclose(seeded.points, [[10, 20], [30, 5]])
    np.testing.assert_allclose(tracked.points, [[11, 20], [31, 5]])
    assert tracked.visible.tolist() == [True, True]
    np.testing.assert_allclose(tracked.visibility, 0.9)


def test_a_new_init_turns_away_the_old_session_and_reset_ends_tracking(server):
    with TestClient(server.build_app()) as client:
        assert _post(client, "/track").status_code == 409
        first = _post(client, "/init", points=np.array([[4, 4]], np.float32)).json()["session"]
        second = _post(client, "/init", points=np.array([[5, 5]], np.float32)).json()["session"]
        assert _post(client, "/track", session=first).status_code == 409
        assert _post(client, "/track", session=second).status_code == 200
        assert client.post("/reset", json={}).status_code == 200
        assert _post(client, "/track", session=second).status_code == 409


def test_points_are_required_and_bounded(server):
    with TestClient(server.build_app()) as client:
        assert _post(client, "/init").status_code == 400
        assert _post(client, "/init", points=np.zeros((0, 2), np.float32)).status_code == 400
        assert _post(client, "/init", points=np.zeros((MAX_POINTS + 1, 2), np.float32)).status_code == 400
        assert _post(client, "/init", points=[[1, 2, 3]]).status_code == 400


def test_a_model_error_is_500_with_its_message(server):
    server.tracker.error = RuntimeError("out of memory")
    with TestClient(server.build_app()) as client:
        assert _post(client, "/init", points=np.array([[4, 4]], np.float32)).status_code == 200
        reply = _post(client, "/track")
    assert reply.status_code == 500
    assert "out of memory" in reply.json()["detail"]


class EchoModel:
    """TAPNext's call signature; tracks each query where it was put, half of them visible."""

    def __call__(self, video, query_points=None, state=None):
        import torch

        if query_points is None:
            query_points = state
        yx = query_points[:, None, :, 1:]
        logits = torch.where(torch.arange(yx.shape[2]) % 2 == 0, 3.0, -3.0)[None, None, :, None]
        return yx, None, logits, query_points


def test_queries_and_tracks_map_between_frame_pixels_and_the_model_square():
    pytest.importorskip("torch")
    tracker = PointTracker.__new__(PointTracker)
    import torch

    tracker.device = torch.device("cpu")
    tracker.resolution = 16
    tracker.model = EchoModel()
    tracker._state = None
    frame = np.zeros((480, 640, 3), np.uint8)
    points = np.array([[0, 0], [320, 240], [639, 100]], np.float32)
    seeded = tracker.seed(frame, points)
    tracked = tracker.track(frame)
    for out in (seeded, tracked):
        np.testing.assert_allclose(out.points, points, atol=1e-3)
        assert out.visible.tolist() == [True, False, True]
        assert out.visibility[0] > 0.9 and out.visibility[1] < 0.1
