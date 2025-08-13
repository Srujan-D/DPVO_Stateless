from __future__ import annotations

import asyncio
import base64
import time
import json
import uuid
from typing import Dict, Optional, Tuple
from pathlib import Path

import cv2
import numpy as np
import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from ray import serve

from dpvo.config import cfg
from dpvo.dpvo import DPVO
from dpvo.patchgraph import PatchGraph
from dpvo.plot_utils import plot_trajectory, save_output_for_COLMAP, save_ply
from evo.core.trajectory import PoseTrajectory3D
from evo.tools import file_interface
from dpvo import lietorch


class DPVOStatelessSLAM:
    """Lightweight functional API around DPVO suitable for micro-services."""

    def __init__(self, cfg, device: str | None = None, model_path: str | None = None):
        self.config = cfg
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_path = model_path or "dpvo.pth"

        # Initialize network (will be loaded in first call)
        self.network = None
        self._n_frames = 0
        self._total_compute_time = 0.0

    def _load_network(self):
        """Lazy load the network on first use"""
        if self.network is None:
            from dpvo.net import VONet
            from collections import OrderedDict

            # Load network from checkpoint file
            state_dict = torch.load(self.model_path, map_location=self.device)
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                if "update.lmbda" not in k:
                    new_state_dict[k.replace("module.", "")] = v

            self.network = VONet()
            self.network.load_state_dict(new_state_dict)
            self.network.to(self.device)
            self.network.eval()

            self.network = torch.compile(self.network)

    def _extract_state(self, slam: DPVO) -> Dict:
        state = {
            # Core state
            "n": slam.n,
            "m": slam.m,
            "counter": slam.counter,
            "is_initialized": slam.is_initialized,
            "tlist": slam.tlist.copy(),
            "ran_global_ba": slam.ran_global_ba.copy(),
            # Image dimensions
            "ht": slam.ht,
            "wd": slam.wd,
            # PatchGraph state (main SLAM state)
            "pg_tstamps": slam.pg.tstamps_[: slam.n + 1].copy(),
            "pg_intrinsics": slam.pg.intrinsics_[: slam.n + 1].detach().clone(),
            "pg_poses": slam.pg.poses_[: slam.n + 1].detach().clone(),
            "pg_patches": slam.pg.patches_[: slam.n + 1].detach().clone(),
            "pg_points": slam.pg.points_[: slam.n * slam.M].detach().clone(),
            "pg_colors": slam.pg.colors_[: slam.n + 1].detach().clone(),
            "pg_index": slam.pg.index_[: slam.n + 1].detach().clone(),
            "pg_index_map": slam.pg.index_map_[: slam.n + 1].detach().clone(),
            # Factor graph state (active)
            "pg_ii": slam.pg.ii.detach().clone(),
            "pg_jj": slam.pg.jj.detach().clone(),
            "pg_kk": slam.pg.kk.detach().clone(),
            "pg_net": slam.pg.net.cpu().numpy(),
            # Inactive factors
            "pg_ii_inac": slam.pg.ii_inac.detach().clone(),
            "pg_jj_inac": slam.pg.jj_inac.detach().clone(),
            "pg_kk_inac": slam.pg.kk_inac.detach().clone(),
            "pg_weight_inac": slam.pg.weight_inac.detach().clone(),
            "pg_target_inac": slam.pg.target_inac.detach().clone(),
            # Memory buffers
            "imap": slam.imap_.cpu().numpy(),
            "gmap": slam.gmap_.cpu().numpy(),
            "fmap1": slam.fmap1_.cpu().numpy(),
            "fmap2": slam.fmap2_.cpu().numpy(),
            # Trajectory state
            "pg_delta": slam.pg.delta.copy(),
            # Loop closure state
            "last_global_ba": getattr(slam, "last_global_ba", -1000),
            "pmem": slam.pmem,
            "mem": slam.mem,
            # Bundle adjustment state
            "ran_global_ba": slam.ran_global_ba.copy(),
            "last_global_ba": slam.last_global_ba,
        }

        # Long term loop closure state -- not implemented as of now in the stateless version
        # Had trouble reproducing in the original demo
        if hasattr(slam, "long_term_lc"):
            state["has_long_term_lc"] = True
        else:
            state["has_long_term_lc"] = False

        return state

    def _deserialize_state(self, state: Dict) -> DPVO:
        slam = DPVO.__new__(DPVO)
        slam.cfg = self.config
        slam.device = self.device

        slam.enable_timing = False

        slam.M = slam.cfg.PATCHES_PER_FRAME
        slam.N = slam.cfg.BUFFER_SIZE
        slam.DIM = self.network.DIM  # DPVO feature dimension from net.py
        slam.RES = self.network.RES  # DPVO resolution from net.py
        slam.P = self.network.P  # DPVO patch size from net.py

        if slam.cfg.MIXED_PRECISION:
            slam.kwargs = kwargs = {"device": "cuda", "dtype": torch.half}
        else:
            slam.kwargs = kwargs = {"device": "cuda", "dtype": torch.float}

        slam.pg = PatchGraph(slam.cfg, slam.P, slam.DIM, slam.kwargs)

        slam.n = state["n"]
        slam.m = state["m"]
        slam.counter = state["counter"]
        slam.is_initialized = state["is_initialized"]
        slam.tlist = state["tlist"].copy()
        slam.ran_global_ba = state["ran_global_ba"].copy()
        slam.ht = state["ht"]
        slam.wd = state["wd"]

        # PatchGraph state
        slam.pg.tstamps_[: slam.n + 1] = state["pg_tstamps"]
        slam.pg.intrinsics_[: slam.n + 1] = state["pg_intrinsics"]
        slam.pg.poses_[: slam.n + 1] = state["pg_poses"]
        slam.pg.patches_[: slam.n + 1] = state["pg_patches"]
        slam.pg.points_[: slam.n * slam.M] = state["pg_points"]
        slam.pg.colors_[: slam.n + 1] = state["pg_colors"]
        slam.pg.index_[: slam.n + 1] = state["pg_index"]
        slam.pg.index_map_[: slam.n + 1] = state["pg_index_map"]

        # factor graph state
        slam.pg.ii = state["pg_ii"]
        slam.pg.jj = state["pg_jj"]
        slam.pg.kk = state["pg_kk"]
        slam.pg.net = torch.from_numpy(state["pg_net"]).to(
            slam.device, non_blocking=True
        )

        # inactive factors
        slam.pg.ii_inac = state["pg_ii_inac"]
        slam.pg.jj_inac = state["pg_jj_inac"]
        slam.pg.kk_inac = state["pg_kk_inac"]
        slam.pg.weight_inac = state["pg_weight_inac"]
        slam.pg.target_inac = state["pg_target_inac"]

        # trajectory state
        slam.pg.delta = state["pg_delta"].copy()

        if slam.cfg.MIXED_PRECISION:
            kwargs = {"device": slam.device, "dtype": torch.half}
        else:
            kwargs = {"device": slam.device, "dtype": torch.float}

        # memory buffers
        slam.imap_ = torch.from_numpy(state["imap"]).to(**kwargs)
        slam.gmap_ = torch.from_numpy(state["gmap"]).to(**kwargs)
        slam.fmap1_ = torch.from_numpy(state["fmap1"]).to(**kwargs)
        slam.fmap2_ = torch.from_numpy(state["fmap2"]).to(**kwargs)

        # loop closure state
        slam.last_global_ba = state["last_global_ba"]
        slam.pmem = state["pmem"]
        slam.mem = state["mem"]

        slam.network = self.network
        slam.pyramid = (slam.fmap1_, slam.fmap2_)
        slam.viewer = None

        # Reinitialize loop closure -- not implemented in stateless version as of now
        if state.get("has_long_term_lc", False):
            try:
                slam.load_long_term_loop_closure()
            except:
                slam.cfg.CLASSIC_LOOP_CLOSURE = False

        return slam

    def save_slam_files(
        self, poses, timestamps, points, colors, intrinsics, H=480, W=640
    ):
        """Save SLAM results to files"""
        slam_dir = Path(f"slam_results/{self.config.user_id}/{self.config.map_name}")
        slam_dir.mkdir(parents=True, exist_ok=True)
        trajectory = PoseTrajectory3D(
            positions_xyz=np.array(poses)[:, :3],
            orientations_quat_wxyz=np.array(poses)[:, [6, 3, 4, 5]],
            timestamps=np.array(timestamps),
        )
        if self.config.SAVE_PLY:
            save_ply(slam_dir / "trajectory.ply", points=points, colors=colors)

        if self.config.SAVE_COLMAP:
            save_output_for_COLMAP(
                slam_dir / "colmap", trajectory, points, colors, *intrinsics, H, W
            )

        if self.config.SAVE_TRAJECTORY:
            file_interface.write_tum_trajectory_file(
                slam_dir / "trajectory.txt",
                trajectory,
            )
        if self.config.PLOT:
            plot_trajectory(
                trajectory,
                title=f"DPVO Trajectory Prediction for {self.config.map_name}",
                filename=f"trajectory_plots/{self.config.map_name}.pdf",
            )

    # ------------------------------------------------------------------
    #  Public API
    # ------------------------------------------------------------------
    @torch.no_grad()
    def run(
        self,
        frame: np.ndarray,
        intrinsics: np.ndarray,
        timestamp: float,
        *,
        state: Optional[Dict] = None,
    ) -> Tuple[Dict, Dict]:
        """
        Run DPVO on a single frame.

        Args:
            frame: RGB image as numpy array (H, W, 3)
            intrinsics: Camera intrinsics [fx, fy, cx, cy]
            timestamp: Frame timestamp
            state: Previous state (None for first frame)

        Returns:
            new_state: Updated state dictionary
            result: Results dictionary with pose and metrics
        """
        try:
            self._load_network()

            start_t = time.perf_counter()

            # For last frame, timestamp is -1 -- terminate the SLAM session
            if timestamp < 0:
                if state is None:
                    raise ValueError("Negative timestamp requires an initial state")
                else:
                    # time1 = time.perf_counter()
                    slam = self._deserialize_state(state)
                    # time2 = time.perf_counter() - time1
                    # print(f"Deserialized SLAM state in {time2:.4f} seconds")
                    points = slam.pg.points_.cpu().numpy()[: slam.m]
                    colors = slam.pg.colors_.view(-1, 3).cpu().numpy()[: slam.m]

                    poses, timestamps = slam.terminate()

                    frame_time = time.perf_counter() - start_t
                    self._n_frames += 1
                    self._total_compute_time += frame_time
                    self.save_slam_files(
                        poses, timestamps, points, colors, intrinsics, slam.ht, slam.wd
                    )
                    return state, {
                        "pose": poses[-1].tolist(),
                        # "points": points.tolist(),
                        # "colors": colors.tolist(),
                        "timestamp": timestamps[-1],
                        # # "intrinsics": intrinsics.tolist(),
                        # "frame_number": slam.n,
                        # # "poses": poses.tolist(),
                        # # "timestamps": timestamps.tolist(),
                        "metrics": {
                            "frame_time_sec": frame_time,
                            "avg_fps": (
                                self._n_frames / self._total_compute_time
                                if self._total_compute_time > 0
                                else 0
                            ),
                            "total_frames": self._n_frames,
                        },
                    }

            frame_tensor = torch.from_numpy(frame).permute(2, 0, 1).to(self.device)
            intrinsics_tensor = torch.from_numpy(intrinsics).to(self.device)

            if state is None:
                slam = DPVO(
                    self.config,
                    self.network,
                    ht=frame_tensor.shape[1],
                    wd=frame_tensor.shape[2],
                    viz=False,
                )
                slam(timestamp, frame_tensor, intrinsics_tensor)
                new_state = self._extract_state(slam)

            else:
                # Subsequent frames
                if timestamp >= 0:
                    # time1 = time.perf_counter()
                    slam = self._deserialize_state(state)
                    # time2 = time.perf_counter() - time1
                    # print(f"Deserialized SLAM state in {time2:.4f} seconds")
                    slam(timestamp, frame_tensor, intrinsics_tensor)
                    # time1 = time.perf_counter()
                    new_state = self._extract_state(slam)
                    # time2 = time.perf_counter() - time1
                    # print("extract_state time:", time2)

            if slam.n - slam.last_global_ba >= slam.cfg.GLOBAL_OPT_FREQ:
                slam.append_factors(*slam.pg.edges_loop())
                slam.last_global_ba = slam.n

            pose = lietorch.SE3(slam.pg.poses_[slam.n - 1]).inv().data.cpu().numpy()

            # points = slam.pg.points_.cpu().numpy()[: slam.m]
            # colors = slam.pg.colors_.view(-1, 3).cpu().numpy()[: slam.m]

            timestamp = slam.tlist[slam.n - 1]

            frame_time = time.perf_counter() - start_t
            self._n_frames += 1
            self._total_compute_time += frame_time

            return new_state, {
                "pose": pose.tolist(),
                "timestamp": timestamp,
                # "points": points.tolist(),
                # "colors": colors.tolist(),
                # # "intrinsics": intrinsics_tensor.cpu().numpy().tolist(),
                # "frame_number": slam.n,
                "metrics": {
                    "frame_time_sec": frame_time,
                    "avg_fps": (
                        self._n_frames / self._total_compute_time
                        if self._total_compute_time > 0
                        else 0
                    ),
                    "total_frames": self._n_frames,
                },
            }
        except Exception as e:
            import traceback

            traceback.print_exc()
            raise RuntimeError(f"DPVO processing error: {e}") from e


###############################################################################
# FastAPI + Ray Serve deployment
###############################################################################

app = FastAPI(title="DPVO WebSocket SLAM Service", version="0.1")


def _b64_to_ndarray(b64_str: str) -> np.ndarray:
    """Convert base64 encoded image to numpy array"""
    try:
        img_bytes = base64.b64decode(b64_str)
        img_array = np.frombuffer(img_bytes, dtype=np.uint8)
        frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError("OpenCV failed to decode image")
        return frame
    except Exception as exc:
        raise ValueError(f"Failed to decode frame: {exc}") from exc


@serve.deployment(num_replicas=1, ray_actor_options={"num_gpus": 1})
@serve.ingress(app)
class DPVOSLAMService:
    """WebSocket-based DPVO SLAM service with automatic session management."""

    def __init__(self):
        # Use absolute path to model in repo
        self.model_path = "dpvo.pth"
        self.cfg = cfg  # TODO: does not seem elegant, check if there's a better way to handle config

        ## Not Initializing here since config is not available yet
        # self.slam = DPVOStatelessSLAM(
        #     config_path=config_path,
        #     model_path=model_path
        # )

        self.sessions: Dict[str, Dict] = {}  # session_id -> slam state

    def initialize_slam(self):
        self.slam = DPVOStatelessSLAM(cfg=self.cfg, model_path=self.model_path)

    @app.websocket("/config")
    async def websocket_config(self, websocket: WebSocket):
        """
        WebSocket endpoint to receive configuration.
        This is a placeholder for future configuration updates.
        """
        await websocket.accept()
        try:
            message = await websocket.receive_text()
            config_data = json.loads(message)
            if hasattr(self.cfg, "merge_from_dict"):
                self.cfg.merge_from_dict(config_data)
            else:
                for k, v in config_data.items():
                    setattr(self.cfg, k, v)
            print("Configuration updated:", self.cfg)
            # Initialize SLAM with new config
            self.initialize_slam()
            await websocket.send_text("Configuration updated successfully")
        except Exception as e:
            await websocket.send_text(f"Error updating config: {str(e)}")

    @app.websocket("/slam")
    async def websocket_endpoint(self, websocket: WebSocket):
        """
        WebSocket endpoint for real-time SLAM.

        Protocol:
        - Connect to /slam (no session_id needed)
        - First message: {"frame": "base64...", "intrinsics": [fx, fy, cx, cy], "timestamp": 0.0}
        - Subsequent: {"frame": "base64...", "timestamp": 1.0}
        - Response: {"pose": [tx, ty, tz, qx, qy, qz, qw]}
        """

        await websocket.accept()

        # Generate new session_id for every connection
        session_id = f"cortex_ws_{uuid.uuid4().hex[:8]}"
        print(f"Creating new DPVO SLAM session: {session_id}")

        try:
            # Send initial session info
            await websocket.send_json(
                {"type": "session_info", "session_id": session_id, "status": "ready"}
            )

            frame_count = 0
            session_start_time = time.time()

            while True:
                # Receive message from client
                try:
                    message_text = await websocket.receive_text()
                    message = json.loads(message_text)
                except json.JSONDecodeError as e:
                    await websocket.send_json(
                        {"type": "error", "message": f"Invalid JSON: {e}"}
                    )
                    continue

                if "type" in message and message["type"] == "end_stream":
                    intrinsics = np.array(message["intrinsics"], dtype=np.float32)
                    state, result = await asyncio.to_thread(
                        self.slam.run,
                        None,
                        intrinsics,
                        -1,
                        state=self.sessions.get(session_id),
                    )
                    self.sessions[session_id] = state
                    response = {
                        "type": "slam_result",
                        "session_id": session_id,
                        # "frame_number": frame_count,
                        "pose": result["pose"],
                        "timestamp": result["timestamp"],
                        # "points": result.get("points", []),
                        # "colors": result.get("colors", []),
                        # # "poses": result["poses"],
                        # # "timestamps": result["timestamps"],
                        "metrics": result.get("metrics", {}),
                    }

                    await websocket.send_json(response)
                    print(f"Session {session_id} completed with {frame_count} frames.")
                    break

                # Validate message format
                if "image" not in message:
                    await websocket.send_json(
                        {"type": "error", "message": "Missing 'image' field in message"}
                    )
                    continue

                try:
                    # Decode image
                    image = _b64_to_ndarray(message["image"])
                    frame_count += 1

                    # Get intrinsics and timestamp
                    if "intrinsics" not in message:
                        await websocket.send_json(
                            {
                                "type": "error",
                                "message": "Missing 'intrinsics' field in message",
                            }
                        )
                        continue
                    intrinsics = np.array(message["intrinsics"], dtype=np.float32)

                    if "timestamp" not in message:
                        timestamp = frame_count
                    else:
                        timestamp = float(message["timestamp"])

                    # Process frame
                    if session_id not in self.sessions:
                        # First frame
                        state, result = await asyncio.to_thread(
                            self.slam.run, image, intrinsics, timestamp
                        )
                        self.sessions[session_id] = state

                        print(f"Session {session_id}: Initialized SLAM")

                    else:
                        # Subsequent frames
                        state_in = self.sessions[session_id]
                        state, result = await asyncio.to_thread(
                            self.slam.run, image, intrinsics, timestamp, state=state_in
                        )
                        self.sessions[session_id] = state

                    # Prepare response
                    response = {
                        "type": "slam_result",
                        "session_id": session_id,
                        "frame_number": frame_count,
                        "pose": result["pose"],
                        "timestamp": result["timestamp"],
                        # "points": result.get("points", []),
                        # "colors": result.get("colors", []),
                        # # "intrinsics": result["intrinsics"],
                        "metrics": {
                            **result["metrics"],
                            "session_fps": frame_count
                            / (time.time() - session_start_time),
                            "total_frames": frame_count,
                        },
                    }

                    await websocket.send_json(response)

                except Exception as e:
                    error_response = {
                        "type": "error",
                        "session_id": session_id,
                        "message": f"Processing error: {str(e)}",
                    }
                    import traceback

                    traceback.print_exc()
                    await websocket.send_json(error_response)
                    print(f"Session {session_id} processing error: {e}")

        except WebSocketDisconnect:
            print(f"WebSocket disconnected for session: {session_id}")
        except Exception as e:
            print(f"WebSocket error for session {session_id}: {e}")
        finally:
            # Clean up session when connection ends
            await self._cleanup_session(session_id)

    async def _cleanup_session(self, session_id: str):
        """Clean up session state"""
        if session_id in self.sessions:
            del self.sessions[session_id]

    @app.get("/health")
    async def health(self):
        return {"status": "ok"}


# Ray Serve deployment handle
deployment = DPVOSLAMService.bind()


if __name__ == "__main__":
    serve.start(detached=True, http_options={"host": "0.0.0.0", "port": 8000})
    serve.run(deployment, route_prefix="/dpvo")
    print("Running on http://0.0.0.0:8000/dpvo  (docs → /docs)")
    try:
        while True:
            time.sleep(30)
    except KeyboardInterrupt:
        serve.shutdown()
