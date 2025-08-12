import asyncio
import base64
import os
import json
import cv2
import numpy as np
from pathlib import Path
import websockets
from dpvo.config import cfg
from dpvo.plot_utils import plot_trajectory, save_output_for_COLMAP, save_ply
from evo.core.trajectory import PoseTrajectory3D
from evo.tools import file_interface


def encode_frame(frame):
    """Encode a frame as a base64 string."""
    _, buffer = cv2.imencode(".jpg", frame)
    return base64.b64encode(buffer).decode("utf-8")


def load_calibration(calib_path):
    """Load camera calibration from file"""
    calib = np.loadtxt(calib_path, delimiter=" ")
    fx, fy, cx, cy = calib[:4]

    K = np.eye(3)
    K[0, 0] = fx
    K[0, 2] = cx
    K[1, 1] = fy
    K[1, 2] = cy

    return calib, K, np.array([fx, fy, cx, cy])


def load_video_frames(video_path, calib, K, stride=1, skip=0):
    """Load all video frames sequentially"""
    print(f"Loading video {video_path}...")

    assert os.path.exists(video_path), video_path
    cap = cv2.VideoCapture(video_path)

    frames = []
    t = 0

    # Skip initial frames
    for _ in range(skip):
        ret, image = cap.read()
        if not ret:
            break

    while True:
        # Skip frames according to stride
        for _ in range(stride):
            ret, image = cap.read()
            if not ret:
                break

        if not ret:
            break

        # Apply undistortion if calibration has distortion coefficients
        if len(calib) > 4:
            image = cv2.undistort(image, K, calib[4:])

        # Resize image (same as in video_stream)
        image = cv2.resize(image, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
        h, w, _ = image.shape
        image = image[: h - h % 16, : w - w % 16]

        # Adjust intrinsics for resized image
        intrinsics = np.array(
            [calib[0] * 0.5, calib[1] * 0.5, calib[2] * 0.5, calib[3] * 0.5]
        )

        frames.append((t, image, intrinsics))
        t += 1
    frames.append((-1, image, intrinsics))  # End marker
    cap.release()
    print(f"Loaded {len(frames)} frames")
    return frames


def load_image_frames(imagedir, calib, K, stride=1, skip=0):
    """Load all image frames sequentially"""
    print(f"Loading images from {imagedir}...")

    calib_array = np.array([calib[0], calib[1], calib[2], calib[3]])

    from itertools import chain

    img_exts = ["*.png", "*.jpeg", "*.jpg"]
    image_list = sorted(chain.from_iterable(Path(imagedir).glob(e) for e in img_exts))[
        skip::stride
    ]
    assert os.path.exists(imagedir), imagedir

    frames = []
    for t, imfile in enumerate(image_list):
        image = cv2.imread(str(imfile))
        if len(calib) > 4:
            image = cv2.undistort(image, K, calib[4:])

        h, w, _ = image.shape
        image = image[: h - h % 16, : w - w % 16]

        intrinsics = calib_array
        frames.append((t, image, intrinsics))

    print(f"Loaded {len(frames)} frames")
    return frames


async def test_dpvo_websocket(args, cfg):
    calib, K, intrinsics = load_calibration(args.calib)

    cfg.calib = calib.tolist()

    if os.path.isdir(args.imagedir):
        frames = load_image_frames(args.imagedir, calib, K, args.stride, args.skip)
    else:
        frames = load_video_frames(args.imagedir, calib, K, args.stride, args.skip)

    if len(frames) == 0:
        print("No frames loaded. Exiting.")
        return

    cfg_uri = "ws://localhost:8000/dpvo/config"
    slam_uri = "ws://localhost:8000/dpvo/slam"
    frame_count = 0

    # Send configuration to the DPVO server
    async with websockets.connect(cfg_uri) as websocket:
        await websocket.send(json.dumps(cfg))
        response = await websocket.recv()
        print("Configuration response:", response)

    # Start the SLAM process
    print(">>> Starting DPVO SLAM...")

    async with websockets.connect(slam_uri) as websocket:
        print("Connected to DPVO WebSocket server URI:", slam_uri)

        # wait for session info
        session_info = await websocket.recv()
        session_data = json.loads(session_info)
        print("Session started with ID:", session_data.get("session_id", "unknown"))

        try:
            for t, image, intrinsics in frames:
                if t < 0:
                    # End of stream marker
                    payload = {
                        "type": "end_stream",
                        "intrinsics": intrinsics.tolist(),
                    }
                    await websocket.send(json.dumps(payload))
                else:
                    payload = {
                        "image": encode_frame(image),
                        "intrinsics": intrinsics.tolist(),
                        "timestamp": t,
                    }
                await websocket.send(json.dumps(payload))

                response = await websocket.recv()
                response_data = json.loads(response)

                if response_data["type"] == "slam_result":
                    if t < 0:
                        frame_count += 1
                        poses = response_data["poses"]
                        timestamps = response_data["timestamps"]
                        is_initialized = response_data["is_initialized"]
                        points = response_data.get("points", [])
                        colors = response_data.get("colors", [])
                        metrics = response_data.get("metrics", None)
                        print(f"SLAM Completed:"
                            f"Last Pose: {poses[-1] if poses else 'N/A'}, "
                            f"Is Initialized: {is_initialized}, "
                            f"Points: {len(points)}, "
                            f"Colors: {len(colors)}, "
                            f"Metrics: {metrics if metrics else 'N/A'}")
                    else:
                        frame_count += 1
                        pose = response_data["pose"]
                        is_initialized = response_data["is_initialized"]
                        points = response_data.get("points", [])
                        colors = response_data.get("colors", [])
                        metrics = response_data.get("metrics", None)
                        # print(
                        #     f"Frame {t} : Pose: {pose}, "
                        #     f"Is Initialized: {is_initialized}, "
                        #     f"Points: {len(points)}, "
                        #     f"Colors: {len(colors)}, "
                        #     f"Metrics: {metrics if metrics else 'N/A'}"
                        # )
                elif response_data["type"] == "error":
                    print(f"Error in frame {t}: {response_data['message']}")
                    break

            print(f"Processed {frame_count} frames.")
        except websockets.exceptions.ConnectionClosed:
            print("WebSocket connection closed")
        except websockets.exceptions.InvalidStatusCode as e:
            print(f"WebSocket connection failed with status: {e.status_code}")
            if e.status_code == 401:
                print("Authentication failed - check your API key")
            elif e.status_code == 403:
                print("Access forbidden - API key may not have required permissions")
        except KeyboardInterrupt:
            print("\nTracking interrupted by user")
        except Exception as e:
            print(f"Error: {e}")


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--network", type=str, default="dpvo.pth")
    parser.add_argument("--imagedir", type=str)
    parser.add_argument("--calib", type=str)
    parser.add_argument("--name", type=str, help="name your run", default="result")
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--skip", type=int, default=0)
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--timeit", action="store_true")
    parser.add_argument("--viz", action="store_true")
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--opts", nargs="+", default=[])
    parser.add_argument("--save_ply", action="store_true")
    parser.add_argument("--save_colmap", action="store_true")
    parser.add_argument("--save_trajectory", action="store_true")
    parser.add_argument(
        "--user_id", type=str, default="default_user", help="User ID for saving results"
    )
    args = parser.parse_args()

    cfg.merge_from_file(args.config)
    cfg.merge_from_list(args.opts)

    # merge save_trajectory option into config
    cfg.user_id = args.user_id
    cfg.map_name = args.name
    cfg.SAVE_TRAJECTORY = args.save_trajectory
    cfg.SAVE_PLY = args.save_ply
    cfg.SAVE_COLMAP = args.save_colmap
    cfg.VIZ = args.viz
    cfg.PLOT = args.plot

    print("Running DPVO Stateless with config...")
    print(cfg)

    # print("Running with config...")
    # print(cfg)

    asyncio.run(test_dpvo_websocket(args, cfg))


if __name__ == "__main__":
    main()
