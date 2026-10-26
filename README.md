# solarian-tapnext

Follow points clicked on a camera image, frame by frame, with
[TAPNext++](https://tap-next-plus-plus.github.io/) (Google DeepMind's online,
recurrent Tracking-Any-Point model). Unlike a box or mask tracker it follows
the pixel itself: a point on a table stays on that point of the table as the
camera moves, which is what the place skill needs.

It is a Solarian *service*: a server that holds the model on a GPU host, and a
client (`TapNextClient`) that a skill or processor calls with each frame.

## Install

On every machine that calls the tracker (the client needs only numpy and OpenCV):

```bash
solarian install solarian-tapnext
```

On the GPU host, also the server side:

```bash
pip install -e 'plugins/solarian-tapnext[server]'
```

The model code is vendored in `solarian_tapnext/tapnext/` (Apache 2.0, from
[tapnet](https://github.com/google-deepmind/tapnet)) because the `tapnet`
package pulls in jax.

## Serve

```bash
solarian run-service point_tracker   # the conf's server block
solarian run-service tapnext         # the service's defaults
```

The first start downloads the published `tapnextpp_512.ckpt` (2.5 GB) into the
project's `models/tapnext/` (`$SOLARIAN_MODELS_DIR` when set). Options:
`--checkpoint` (a bare filename is looked up there; a path is used as is),
`--resolution N` (the square the frame is resized to, a multiple of 8;
default 512), `--port` (default 8767).

At 512 on an RTX 3090 a step takes about 70 ms on the server and 76 ms round
trip from a laptop over the LAN, for 1 point or 65. Keep the checkpoint's 512:
at 256 a step is about 22 ms, but the tracks follow only about half of each
frame's motion (a frame shifted 64 px in small steps reads as about 33 px).

## Use

```python
from solarian_tapnext.client import TapNextClient

client = TapNextClient(conf.server_url, timeout=conf.timeout, token=conf.token)
tracked = client.init(frame_bgr, [[512, 400], [530, 400]], timeout=conf.init_timeout)
while running:
    tracked = client.track(next_frame_bgr)
    tracked.points      # (N, 2) float32, pixels of the frame you sent, in seed order
    tracked.visible     # (N,) bool
    tracked.visibility  # (N,) float32 in [0, 1]
client.reset()
```

Points are fixed at `init`: to add or move one, `init` again with the whole
set. The server holds one tracking session; `init` from another client
replaces it, and the first client's next `track` gets a 409, which
`ServiceClient` raises. Re-seed on it.

## Observation field

`TapNextPointProcessor` follows one clicked point as a calculated field:

```python
from solarian_tapnext.processor import TapNextPointProcessor, TrackedPointData

class WristCameraObs(ModuleObs):
    target_point = CalculatedField(
        inputs=[MyRobot.wrist_camera.camera.rgb_image],
        processor=TapNextPointProcessor,
        output=TrackedPointData(),
        conf="conf/services/point_tracker.yaml",
    )
```

`controller.start_service("wrist_camera", "target_point", point=(x, y))` seeds
it at a pixel of the camera image; `stop_service` clears it. The value is
`(x0, y0, x1, y1, visibility %)`, a `box_px` square centred on the point in the
camera's pixels, all zeros while idle. The UI draws it like any bounding box,
labelled `TAPNext N%`, or `occluded N%` below 50 (the box then stays at
TAPNext's estimate of where the point is). When another client takes the
server's session, the processor re-seeds at the last point.

The so106 controller binds it to `v`: press `v`, click the wrist image, and
teleop the arm to watch the box stay on the point; `b` clears it.

## HTTP

| Route | Body | Reply |
|---|---|---|
| `POST /init` | `image_jpeg`, `points` (N, 2) | `points`, `visible`, `visibility`, `inference_ms`, `session` |
| `POST /track` | `image_jpeg`, `session` (optional) | same; 409 without a session or for a stale one |
| `POST /reset` | | `{}` |
| `GET /info` | | checkpoint, resolution, device, `tracking_active`, `max_points` |

Arrays travel through `solarian.services.encode_arrays`. `init` takes 1 to
1024 points; anything else is a 400.

## Conf

`templates/services/point_tracker.yaml` is copied to
`conf/services/point_tracker.yaml`. It reads `deployment.tapnext_url`, and
holds the server block plus the caller's keys: `crop` (`auto` crops each
seed's frame to the fisheye's lit circle with `solarian.core.fisheye.lit_box`,
so the scene fills the model's square input; `none`; or `[x0, y0, x1, y1]`),
`resize` (the long edge sent, in pixels), `box_px`, `jpeg_quality`, `timeout`,
`init_timeout`, and `execution_place` for the field.

## Tests

```bash
python -m pytest -q plugins/solarian-tapnext/tests
```

They run the server's routes over a fake model and the processor over a fake
client, without torch weights.
