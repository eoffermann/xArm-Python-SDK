"""
xArm Gradio Controller — Drag/Teach Recorder and Looping Playback
===================================================================
This script creates a Gradio web app for controlling an xArm robot in drag/teach mode, recording trajectories,
and looping playback. It connects to an xArm controller over IP, manages modes and safety (collision and teach
sensitivity), and automatically starts recording when it detects joint motion and stops/saves the trajectory
after the arm has been idle for a configurable timeout. The UI lets you toggle drag/teach mode, adjust
sensitivities and playback speed, and start/stop continuous playback of the recorded trajectory on a background
thread. A small CLI wrapper configures the Gradio server (host, port, share), motion detection parameters,
default robot IP, and trajectory filename before launching the app.

"""
from __future__ import annotations

import argparse
import math
import threading
import time
from dataclasses import dataclass
from typing import Optional, Tuple, List

import gradio as gr

try:
    from xarm.wrapper import XArmAPI
except Exception:  # pragma: no cover
    XArmAPI = None  # type: ignore


# ------------------------------
# Configuration Dataclasses
# ------------------------------

@dataclass
class DetectConfig:
    """
    Parameters for movement-based record detection.

    Attributes:
        motion_threshold: Joint-space Euclidean threshold to declare motion start.
        idle_timeout: Seconds with motion below threshold to stop recording.
        report_hz: Expected report frequency for smoothing logic. Informational only.
    """

    motion_threshold: float = 0.015  # radians
    idle_timeout: float = 1.0  # seconds
    report_hz: float = 100.0


@dataclass
class AppState:
    """
    Mutable state shared with the UI.

    Attributes:
        ip: Last entered IP address.
        connected: Whether a session is active.
        teach_mode: Whether controller is in drag/teach mode.
        recording: Whether the controller is currently recording.
        last_status: Short status string for display.
        traj_filename: Name of the saved trajectory on controller storage.
        playback_speed: UI playback speed; mapped to valid double_speed values.
        collision_sensitivity: Collision sensitivity level 0-5.
        teach_sensitivity: Drag/teach sensitivity level 0-5 (if supported by firmware).
    """

    ip: str = ""
    connected: bool = False
    teach_mode: bool = False
    recording: bool = False
    last_status: str = "idle"
    traj_filename: str = "gradio_record"
    playback_speed: float = 1.0
    collision_sensitivity: int = 3
    teach_sensitivity: int = 3


# ------------------------------
# Controller
# ------------------------------

class XArmGradioController:
    """
    Owns the xArm session, callbacks, recording, and playback threads.

    Public methods return concise status strings for UI display.
    """

    def __init__(self, detect_cfg: DetectConfig) -> None:
        self.detect_cfg = detect_cfg
        self.arm: Optional[XArmAPI] = None
        self.state = AppState()
        self._cb_registered = False
        self._lock = threading.RLock()

        # Motion detection variables
        self._last_q: Optional[List[float]] = None
        self._last_motion_t: float = 0.0

        # Recording flags
        self._record_started = False

        # Playback control
        self._play_thread: Optional[threading.Thread] = None
        self._stop_play_evt = threading.Event()

    # ---- connection ----

    def connect(self, ip: str) -> Tuple[str, bool]:
        """
        Connect to the xArm controller.

        Args:
            ip: IPv4 address of the robot controller.

        Returns:
            Tuple of status string and connection boolean.
        """
        if XArmAPI is None:
            return ("xarm-python-sdk not installed", False)

        with self._lock:
            if self.arm is not None:
                return ("already connected", True)
            try:
                arm = XArmAPI(ip, is_radian=True)
                self.arm = arm
                self.state.ip = ip
                # Bringup
                arm.motion_enable(True)
                arm.set_mode(0)
                arm.set_state(0)
                # Collision sensitivity default
                try:
                    arm.set_collision_sensitivity(self.state.collision_sensitivity)
                except Exception:
                    pass
                # Optional teach sensitivity
                if hasattr(arm, "set_teach_sensitivity"):
                    try:
                        arm.set_teach_sensitivity(self.state.teach_sensitivity)
                    except Exception:
                        pass
                self.state.connected = True
                self.state.last_status = "connected"
                return ("connected", True)
            except Exception as e:  # pragma: no cover
                self.arm = None
                self.state.connected = False
                return (f"connect failed: {e}", False)

    def disconnect(self) -> Tuple[str, bool]:
        """
        Disconnect and cleanup.

        Returns:
            Tuple of status string and False since disconnected.
        """
        with self._lock:
            self._stop_playback_thread()
            if self.arm is not None and self.state.recording:
                try:
                    # stop without saving if mid-gesture
                    self.arm.stop_record_trajectory(filename=None)
                except Exception:
                    pass
            if self.arm is not None and self._cb_registered:
                try:
                    self.arm.register_report_callback(lambda *_args, **_kw: None)
                except Exception:
                    pass
                self._cb_registered = False
            if self.arm is not None and self.state.teach_mode:
                try:
                    self.arm.set_mode(0)
                    self.arm.set_state(0)
                except Exception:
                    pass
            try:
                if self.arm is not None:
                    self.arm.disconnect()
            except Exception:
                pass
            finally:
                self.arm = None
                self.state.connected = False
                self.state.teach_mode = False
                self.state.recording = False
                self.state.last_status = "disconnected"
                return ("disconnected", False)

    # ---- teach mode ----

    def toggle_teach(self, enable: bool) -> str:
        """
        Enter or exit drag/teach mode.

        Args:
            enable: True to enter teach mode. False to exit.

        Returns:
            Status string.
        """
        with self._lock:
            if self.arm is None:
                return "not connected"
            try:
                if enable:
                    self.arm.set_mode(2)
                    self.arm.set_state(0)
                    self._ensure_report_callback()
                    self.state.teach_mode = True
                    self.state.last_status = "waiting"
                else:
                    if self.state.recording:
                        try:
                            # stop without save on explicit exit; auto-save happens via idle
                            self.arm.stop_record_trajectory(filename=None)
                        except Exception:
                            pass
                        self.state.recording = False
                        self._record_started = False
                    self.arm.set_mode(0)
                    self.arm.set_state(0)
                    if self._cb_registered:
                        try:
                            self.arm.register_report_callback(lambda *_args, **_kw: None)
                        except Exception:
                            pass
                        self._cb_registered = False
                    self.state.teach_mode = False
                    self.state.last_status = "idle"
                return self.state.last_status
            except Exception as e:
                return f"teach toggle failed: {e}"

    # ---- collision sensitivity ----

    def set_collision_sensitivity(self, level: int) -> str:
        """
        Set collision sensitivity 0–5.

        Args:
            level: Integer in [0, 5]. Higher is more sensitive.

        Returns:
            Status string.
        """
        with self._lock:
            if self.arm is None:
                return "not connected"
            level = max(0, min(5, int(level)))
            try:
                code = self.arm.set_collision_sensitivity(level)
                self.state.collision_sensitivity = level
                if hasattr(self.arm, "save_conf"):
                    try:
                        self.arm.save_conf()
                    except Exception:
                        pass
                return f"collision:{level} (code {code})"
            except Exception as e:
                return f"collision set failed: {e}"

    # ---- teach sensitivity ----

    def set_teach_sensitivity(self, level: int) -> str:
        """
        Set manual drag/teach sensitivity 0-5, if supported by firmware.

        Args:
            level: Integer in [0, 5]. Higher is lighter drag.

        Returns:
            Status string.
        """
        with self._lock:
            if self.arm is None:
                return "not connected"
            level = max(0, min(5, int(level)))
            if not hasattr(self.arm, "set_teach_sensitivity"):
                return "teach sensitivity unsupported"
            try:
                code = self.arm.set_teach_sensitivity(level)
                self.state.teach_sensitivity = level
                return f"teach_sens:{level} (code {code})"
            except Exception as e:
                return f"teach sensitivity set failed: {e}"

    # ---- playback ----

    def start_playback(self, ui_speed: float) -> str:
        """
        Begin looping playback on a background thread.

        Args:
            ui_speed: UI slider value 0-5, snapped to {1,2,4}.

        Returns:
            Status string.
        """
        with self._lock:
            if self.arm is None:
                return "not connected"
            if self._play_thread and self._play_thread.is_alive():
                return "playback already running"
            self.state.playback_speed = max(0.0, min(5.0, float(ui_speed)))
            self._stop_play_evt.clear()
            self._play_thread = threading.Thread(target=self._playback_loop, name="playback", daemon=True)
            self._play_thread.start()
            self.state.last_status = "play:start"
            return self.state.last_status

    def stop_playback(self) -> str:
        """
        Stop looping playback quickly.

        Returns:
            Status string.
        """
        with self._lock:
            self._stop_playback_thread()
            self.state.last_status = "play:stop"
            return self.state.last_status

    # ---- internals ----

    def _ensure_report_callback(self) -> None:
        arm = self.arm
        if arm is None:
            return
        if self._cb_registered:
            return
        try:
            arm.register_report_callback(self._on_report)
            self._cb_registered = True
        except Exception:
            self._cb_registered = False

    @staticmethod
    def _snap_double_speed(ui_speed: float) -> int:
        """
        Map a continuous UI speed to valid double_speed {1,2,4}.

        Args:
            ui_speed: UI slider value.

        Returns:
            One of 1, 2, 4.
        """
        if ui_speed >= 3.0:
            return 4
        if ui_speed >= 1.5:
            return 2
        return 1

    def _on_report(self, data: dict) -> None:
        """
        Robot state callback used for motion detection.

        Args:
            data: Report dictionary from SDK. Expected keys include 'angles'.
        """
        try:
            # accept multiple possible keys from SDK variants
            q = (data.get("angles") or
                 data.get("joint_angles") or
                 data.get("angle") or
                 data.get("joints"))
            if not q:
                return
            now = time.time()
            moved = False

            if self._last_q is not None:
                # Euclidean norm of joint deltas
                dq2 = 0.0
                for a, b in zip(q, self._last_q):
                    d = float(a) - float(b)
                    dq2 += d * d
                # if report rate is high, per-tick deltas can be tiny; also check max-abs per joint
                dist = math.sqrt(dq2)
                max_abs = max(abs(float(a) - float(b)) for a, b in zip(q, self._last_q))
                moved = (dist >= self.detect_cfg.motion_threshold) or (max_abs >= (self.detect_cfg.motion_threshold * 0.6))

            self._last_q = list(q)

            if not self.state.teach_mode:
                # reflect idle when not teaching
                self.state.last_status = "idle"
                return

            if moved:
                self._last_motion_t = now
                if not self._record_started:
                    try:
                        if self.arm is not None:
                            self.arm.start_record_trajectory()
                        self.state.recording = True
                        self._record_started = True
                        self.state.last_status = "recording"
                    except Exception:
                        self.state.last_status = "record:start failed"
            else:
                if self._record_started and (now - self._last_motion_t) >= self.detect_cfg.idle_timeout:
                    # Attempt atomic stop+save
                    try:
                        if self.arm is not None:
                            self.arm.stop_record_trajectory(filename=self.state.traj_filename)
                        self.state.recording = False
                        self._record_started = False
                        self.state.last_status = "idle"
                    except Exception:
                        self.state.last_status = "record:save failed"
                    # Auto-exit teach mode after saving attempt
                    try:
                        if self.arm is not None:
                            self.arm.set_mode(0)
                            self.arm.set_state(0)
                            if self._cb_registered:
                                try:
                                    self.arm.register_report_callback(lambda *_args, **_kw: None)
                                except Exception:
                                    pass
                                self._cb_registered = False
                        self.state.teach_mode = False
                    except Exception:
                        pass
        except Exception:
            pass

    def _playback_loop(self) -> None:
        arm = self.arm
        if arm is None:
            return
        filename = self.state.traj_filename
        double_speed = self._snap_double_speed(self.state.playback_speed)

        # Ensure not in teach and preload the trajectory
        try:
            arm.set_mode(0)
            arm.set_state(0)
            arm.load_trajectory(filename)
        except Exception:
            # If preload fails, loop will attempt playback anyway
            pass

        while not self._stop_play_evt.is_set():
            try:
                # Blocking single playback iteration
                arm.playback_trajectory(times=1, filename=filename, wait=True, double_speed=double_speed)
            except Exception:
                # Backoff if controller rejects the request
                if self._stop_play_evt.wait(0.2):
                    break
            # Allow responsive stop between iterations
            if self._stop_play_evt.wait(0.05):
                break

        # Attempt to leave the controller in motion-ready state
        try:
            arm.set_state(0)
        except Exception:
            pass

    def _stop_playback_thread(self) -> None:
        if self._play_thread and self._play_thread.is_alive():
            self._stop_play_evt.set()
            self._play_thread.join(timeout=1.0)
        self._play_thread = None


# ------------------------------
# Gradio UI Assembly
# ------------------------------

def build_ui(ctrl: XArmGradioController) -> gr.Blocks:
    """
    Construct the Gradio Blocks UI.

    Args:
        ctrl: Controller instance.

    Returns:
        Configured `gr.Blocks` app.
    """
    with gr.Blocks(title="xArm Gradio Controller", theme="gradio/soft") as demo:
        gr.Markdown("# xArm Gradio Controller\nControl drag/teach, record motion by moving the arm, then loop playback.")

        with gr.Row():
            ip_in = gr.Textbox(label="Robot IP", placeholder="192.168.1.221", value=ctrl.state.ip)
            conn_state = gr.Label(value="disconnected", label="Connection")
        with gr.Row():
            btn_connect = gr.Button("Connect", variant="primary")
            btn_disconnect = gr.Button("Disconnect")

        with gr.Row():
            teach_toggle = gr.Checkbox(value=False, label="Drag/Teach Mode")
            teach_state = gr.Label(value="teach:off", label="Mode")

        with gr.Row():
            collision = gr.Slider(0, 5, value=ctrl.state.collision_sensitivity, step=1, label="Collision Sensitivity (0-5)")
            teach_sens = gr.Slider(0, 5, value=ctrl.state.teach_sensitivity, step=1, label="Teach Sensitivity (0-5)")
            speed = gr.Slider(0, 5, value=ctrl.state.playback_speed, step=0.1, label="Playback Speed (snaps to 1×/2×/4×)")

        with gr.Row():
            btn_play = gr.Button("Play", variant="primary")
            btn_stop = gr.Button("Stop")

        status = gr.Textbox(label="Status", value=ctrl.state.last_status)

        # periodic UI poll to reflect backend state (waiting/recording/idle and teach toggle)
        def _poll():
            if ctrl.state.teach_mode:
                s = "recording" if ctrl.state.recording else "waiting"
            else:
                s = "idle"
            ctrl.state.last_status = s
            return s, ctrl.state.teach_mode, ("teach:on" if ctrl.state.teach_mode else "teach:off")

        timer = gr.Timer(0.5)
        timer.tick(_poll, None, [status, teach_toggle, teach_state])

        # ---- wiring ----
        def _on_connect(ip: str):
            msg, ok = ctrl.connect(ip)
            return msg, ("connected" if ok else "disconnected")

        def _on_disconnect():
            msg, _ = ctrl.disconnect()
            return msg, "disconnected", False, "teach:off"

        def _on_teach_toggle(flag: bool):
            msg = ctrl.toggle_teach(flag)
            return msg, msg

        def _on_collision(val: int):
            msg = ctrl.set_collision_sensitivity(int(val))
            return msg

        def _on_speed(val: float):
            ctrl.state.playback_speed = float(val)
            ds = XArmGradioController._snap_double_speed(ctrl.state.playback_speed)
            return f"speed:{val:.2f} (double_speed={ds})"

        def _on_teach_sens(val: int):
            msg = ctrl.set_teach_sensitivity(int(val))
            return msg

        def _on_play(speed_val: float):
            msg = ctrl.start_playback(speed_val)
            return msg

        def _on_stop():
            msg = ctrl.stop_playback()
            return msg

        btn_connect.click(_on_connect, inputs=[ip_in], outputs=[status, conn_state])
        btn_disconnect.click(_on_disconnect, inputs=None, outputs=[status, conn_state, teach_toggle, teach_state])

        teach_toggle.change(_on_teach_toggle, inputs=[teach_toggle], outputs=[teach_state, status])

        collision.change(_on_collision, inputs=[collision], outputs=[status])
        teach_sens.change(_on_teach_sens, inputs=[teach_sens], outputs=[status])
        speed.change(_on_speed, inputs=[speed], outputs=[status])

        btn_play.click(_on_play, inputs=[speed], outputs=[status])
        btn_stop.click(_on_stop, inputs=None, outputs=[status])

    return demo


# ------------------------------
# CLI Entrypoint
# ------------------------------

def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for the Gradio app.

    Returns:
        Parsed namespace.
    """
    parser = argparse.ArgumentParser(description="xArm Gradio Controller: drag/teach recorder and playback")
    parser.add_argument("--host", default="0.0.0.0", help="Gradio host bind address")
    parser.add_argument("--port", type=int, default=7860, help="Gradio TCP port")
    parser.add_argument("--share", action="store_true", help="Enable Gradio share")
    parser.add_argument("--theme", default="gradio/soft", help="Gradio theme")
    parser.add_argument("--motion-threshold", type=float, default=0.015, help="Joint-space motion threshold in radians")
    parser.add_argument("--idle-timeout", type=float, default=1.0, help="Idle timeout (s) to stop recording")
    parser.add_argument("--ip", default="", help="Robot controller IP; if omitted, use UI")
    parser.add_argument("--traj", default="gradio_record", help="Trajectory base name to save and play")
    return parser.parse_args()


def main() -> None:
    """Launch the Gradio app.

    Ensure `xarm-python-sdk` and `gradio` are installed. Connect the PC and xArm controller on the same LAN.
    """
    args = parse_args()
    detect_cfg = DetectConfig(motion_threshold=args.motion_threshold, idle_timeout=args.idle_timeout)
    ctrl = XArmGradioController(detect_cfg)
    if args.ip:
        ctrl.state.ip = args.ip
    if args.traj:
        ctrl.state.traj_filename = args.traj
    demo = build_ui(ctrl)
    demo.launch(server_name=args.host, server_port=args.port, share=args.share)


if __name__ == "__main__":  # pragma: no cover
    main()
