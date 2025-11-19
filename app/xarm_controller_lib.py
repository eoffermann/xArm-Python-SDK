from __future__ import annotations

"""
xArm controller wrapper: point-to-point motion along selectable tool axis.

This module defines a thread-safe XArmController wrapper around the uFactory xArm 850 SDK
that exposes a UI-friendly API for connection, drag/teach mode, and oscillatory point-to-point
motion along a selectable tool axis. It handles connecting to the robot by IP, enabling motion,
caching SDK and TCP velocity/acceleration limits, and switching controller modes safely. When
point-to-point motion is started, it reads the current TCP pose as the center, computes two
endpoints offset along the chosen tool axis (x, y, or z) using a roll-pitch-yaw-to-rotation-matrix 
conversion, and then runs a background loop that bounces between those endpoints at
speeds/accelerations derived from percentage sliders. Motion can be stopped at any time, which
signals the thread to exit, puts the arm back into position mode, and returns it to the saved
starting pose, with all operations reporting concise status strings for a Gradio UI.

"""

import math
import threading
from dataclasses import dataclass
from typing import Optional, Tuple, List, Literal

import gradio as gr

try:
    from xarm.wrapper import XArmAPI  # type: ignore
except Exception:  # pragma: no cover
    XArmAPI = None  # type: ignore

from xarm_state_lib import ArmState

AxisLiteral = Literal["x", "y", "z"]

SPEED_MULTIPLIER = 4.0
ACCEL_MULTIPLIER = 4.0

class XArmController:
    """
    Thin controller for uFactory xArm 850 used by the Gradio app.

    This class wraps the xArm SDK usage in a thread-safe, UI-friendly API.

    Notes:
        - Requires official uFactory SDK: `pip install xarm` if a conda package is not available.
        - Tested on Windows with Python 3.10.11. SDK: xarm >= 1.12 (XArmAPI).
    """

    def __init__(self) -> None:
        self._arm: Optional[XArmAPI] = None
        self.state = ArmState()
        self._lock = threading.RLock()
        self._stop_event = threading.Event()
        self._motion_thread: Optional[threading.Thread] = None

    # -------------------- Helpers --------------------
    def _check(self, code: int, ok_msg: str = "OK", fail_msg: str = "Error") -> Tuple[bool, str]:
        """
        Normalize SDK return code handling.

        Args:
            code: Return code from an SDK call.
            ok_msg: Message to return when code == 0.
            fail_msg: Base message to return otherwise.

        Returns:
            Tuple[bool, str]: (success, message).
        """
        if code == 0:
            return True, ok_msg
        # Attempt to fetch and clean error/warn codes where possible
        extra = []
        try:
            if self._arm is not None and hasattr(self._arm, "get_err_warn_code"):
                _c, err, warn = self._arm.get_err_warn_code()  # type: ignore[attr-defined]
                extra.append(f"err={err}")
                extra.append(f"warn={warn}")
                # Best-effort cleanup
                if hasattr(self._arm, "clean_error"):
                    try:
                        self._arm.clean_error()  # type: ignore[attr-defined]
                    except Exception:
                        pass
                if hasattr(self._arm, "clean_warn"):
                    try:
                        self._arm.clean_warn()  # type: ignore[attr-defined]
                    except Exception:
                        pass
        except Exception:
            pass
        return False, f"{fail_msg} (code {code}{', ' + ', '.join(extra) if extra else ''})"

    def _set_mode_and_ready(self, mode: int) -> Tuple[bool, str]:
        """
        Set controller mode then ready state.

        Args:
            mode: Target mode number.

        Returns:
            Tuple[bool, str]: (success, message)
        """
        if self._arm is None:
            return False, "SDK not connected"
        try:
            ok, msg = self._check(self._arm.set_mode(mode), ok_msg=f"mode={mode}", fail_msg="set_mode failed")
            if not ok:
                return False, msg
            ok, msg = self._check(self._arm.set_state(0), ok_msg="ready", fail_msg="set_state failed")
            return ok, msg
        except Exception as e:  # Safety net
            return False, f"mode/state exception: {e}"

    def _rpy_to_rot(self, rx: float, ry: float, rz: float, is_radian: bool) -> list[list[float]]:
        """
        Compute a 3x3 rotation matrix from roll, pitch, yaw.

        Uses intrinsic XYZ (Rx * Ry * Rz). Converts from degrees if needed.

        Args:
            rx: Roll.
            ry: Pitch.
            rz: Yaw.
            is_radian: True if angles are radians.

        Returns:
            list[list[float]]: 3x3 rotation matrix mapping tool frame to base.
        """
        if not is_radian:
            rx, ry, rz = math.radians(rx), math.radians(ry), math.radians(rz)
        cx, sx = math.cos(rx), math.sin(rx)
        cy, sy = math.cos(ry), math.sin(ry)
        cz, sz = math.cos(rz), math.sin(rz)
        # R = Rz * Ry * Rx (extrinsic) == Rx * Ry * Rz (intrinsic XYZ)
        R00 = cz * cy
        R01 = cz * sy * sx - sz * cx
        R02 = cz * sy * cx + sz * sx
        R10 = sz * cy
        R11 = sz * sy * sx + cz * cx
        R12 = sz * sy * cx - cz * sx
        R20 = -sy
        R21 = cy * sx
        R22 = cy * cx
        return [[R00, R01, R02], [R10, R11, R12], [R20, R21, R22]]

    @staticmethod
    def _tool_axis_col(axis: AxisLiteral) -> int:
        """
        Map tool-axis label to rotation-matrix column index.

        Args:
            axis: One of "x", "y", or "z".

        Returns:
            int: Column index in R representing the tool axis resolved in base.

        Raises:
            ValueError: If axis is not one of the allowed values.
        """
        a = str(axis).lower()
        if a == "x":
            return 0
        if a == "y":
            return 1
        if a == "z":
            return 2
        raise ValueError("axis must be 'x', 'y', or 'z'")

    # -------------------- Connection --------------------
    def connect(self, ip: str) -> str:
        """
        Connect to the robot at the provided IP.

        Args:
            ip: IPv4 address of the xArm controller.

        Returns:
            str: A short human-readable status string for the UI.
        """
        with self._lock:
            self.state.ip = ip.strip() or self.state.ip
            if XArmAPI is None:
                self.state.last_error = (
                    "xArm SDK is not installed. Run: pip install xarm"
                )
                return self.state.last_error
            if self.state.connected:
                return f"Already connected to {self.state.ip}"
            try:
                self._arm = XArmAPI(port=self.state.ip)
                # Cache unit preference if available
                try:
                    if hasattr(self._arm, "is_radian"):
                        self.state.is_radian = bool(getattr(self._arm, "is_radian"))
                except Exception:
                    pass
                # Prepare for motion
                ok, msg = self._check(self._arm.motion_enable(True), ok_msg="motors on", fail_msg="motion_enable failed")
                if not ok:
                    self.state.last_error = msg
                    return msg
                ok, msg = self._set_mode_and_ready(0)
                if not ok:
                    self.state.last_error = msg
                    return msg
                # Query TCP linear velocity/acceleration if supported
                try:
                    if hasattr(self._arm, "get_tcp_maxlin_vel"):
                        c_v, vmax = self._arm.get_tcp_maxlin_vel()  # type: ignore[attr-defined]
                        if c_v == 0 and vmax is not None:
                            self.state.max_tcp_lin_vel_mmps = float(vmax)
                    if hasattr(self._arm, "get_tcp_maxlin_acc"):
                        c_a, amax = self._arm.get_tcp_maxlin_acc()  # type: ignore[attr-defined]
                        if c_a == 0 and amax is not None:
                            self.state.max_tcp_lin_acc_mmps2 = float(amax)
                except Exception:
                    pass
                # Cache version if available
                try:
                    code, ver = self._arm.get_version()
                    if code == 0:
                        self.state.sdk_version = str(ver)
                except Exception:
                    pass
                self.state.connected = True
                self.state.last_error = "Connected"
                return f"Connected to {self.state.ip}"
            except Exception as e:
                self._arm = None
                self.state.connected = False
                self.state.last_error = f"Connect failed: {e}"
                return self.state.last_error

    def disconnect(self) -> str:
        """
        Disconnect from the robot, stopping any active motion first.

        Returns:
            str: Status string describing the result of the operation.
        """
        # Phase 1: stop outside to avoid holding lock during join
        self.stop_motion()
        with self._lock:
            try:
                if self._arm is not None:
                    try:
                        self._arm.disconnect()
                    finally:
                        self._arm = None
                self.state.connected = False
                self.state.teach_enabled = False
                self.state.last_error = "Disconnected"
                return "Disconnected"
            except Exception as e:
                self.state.last_error = f"Disconnect error: {e}"
                return self.state.last_error

    # -------------------- Teach/Drag --------------------
    def toggle_teach(self) -> Tuple[str, str]:
        """
        Toggle joint teaching (drag) mode on/off.

        Returns:
            Tuple[str, str]: (status_text, draggable_label) for UI display.
        """
        with self._lock:
            if not self.state.connected or self._arm is None:
                return ("Not connected", "Not Draggable")
            try:
                if self.state.playing:
                    # Stop current motion before switching modes
                    # Do not block the UI in the lock; signal and exit
                    pass
            except Exception:
                pass
        # Stop motion outside lock
        if self.state.playing:
            self.stop_motion()
        with self._lock:
            try:
                if not self.state.teach_enabled:
                    ok, msg = self._set_mode_and_ready(2)
                    if not ok:
                        # Try to park in position mode for safety
                        self._set_mode_and_ready(0)
                        self.state.teach_enabled = False
                        return (f"Teach enable failed: {msg}", "Not Draggable")
                    self.state.teach_enabled = True
                    return ("Teach enabled", "Draggable")
                else:
                    ok, msg = self._set_mode_and_ready(0)
                    if not ok:
                        self.state.teach_enabled = True
                        return (f"Teach disable failed: {msg}", "Draggable")
                    self.state.teach_enabled = False
                    return ("Teach disabled", "Not Draggable")
            except Exception as e:
                self.state.last_error = f"Teach toggle failed: {e}"
                # Best effort to indicate status
                return (self.state.last_error, "Not Draggable")

    # -------------------- Point‑to‑Point Motion --------------------
    def _p2p_loop(self, amp_mm: float, speed_percent: float, accel_percent: float, axis: AxisLiteral) -> None:
        """
        Bounce between two endpoints along the selected tool axis.

        Endpoints are computed by taking the saved center pose and applying a
        tool-frame offset of [±amp/2, 0, 0], [0, ±amp/2, 0], or [0, 0, ±amp/2]
        depending on `axis`, transformed into base coordinates via the current
        RPY. The orientation is held constant.

        Args:
            amp_mm: Peak-to-peak amplitude in millimeters.
            speed_percent: 0–100% of max TCP linear speed.
            accel_percent: 0–100% of max TCP linear acceleration.
            axis: Tool axis for motion. One of "x", "y", "z".
        """
        assert self._arm is not None
        vmax = float(self.state.max_tcp_lin_vel_mmps) * SPEED_MULTIPLIER
        amax = float(self.state.max_tcp_lin_acc_mmps2) * ACCEL_MULTIPLIER

        pct_v = max(0.0, min(100.0, speed_percent)) / 100.0
        pct_a = max(0.0, min(100.0, accel_percent)) / 100.0
        #v_target = max(1.0, pct_v * vmax)
        #a_target = max(10.0, pct_a * amax)
        v_target = pct_v * vmax
        a_target = pct_a * amax
        
        # Endpoints around the saved center pose, along selected tool axis resolved in base
        pose = list(self.state.init_pose or [])
        if not pose or len(pose) < 6:
            return
        x, y, z, rx, ry, rz = pose

        # Determine unit mode for angles
        if hasattr(self._arm, "is_radian"):
            try:
                is_radian = bool(getattr(self._arm, "is_radian"))
            except Exception:
                is_radian = self.state.is_radian
        else:
            is_radian = self.state.is_radian

        R = self._rpy_to_rot(rx, ry, rz, is_radian)
        col = self._tool_axis_col(axis)
        # tool axis in base = selected column of rotation matrix
        fx, fy, fz = R[0][col], R[1][col], R[2][col]
        norm = max(1e-9, (fx * fx + fy * fy + fz * fz) ** 0.5)
        fx, fy, fz = fx / norm, fy / norm, fz / norm
        half = max(0.0, amp_mm) / 2.0
        p_low = [x - fx * half, y - fy * half, z - fz * half, rx, ry, rz]
        p_high = [x + fx * half, y + fy * half, z + fz * half, rx, ry, rz]

        self._arm.set_tcp_jerk(50000)

        current_target_high = True
        while not self._stop_event.is_set():
            target = p_high if current_target_high else p_low
            try:
                # set_position accepts speed and mvacc per SDK
                self._arm.set_position(
                    target[0], target[1], target[2], target[3], target[4], target[5],
                    speed=v_target, mvacc=a_target, wait=True
                )
            except Exception:
                break
            current_target_high = not current_target_high

    def play_point_to_point(self, amplitude_cm: float, speed_percent: float, accel_percent: float, axis: AxisLiteral = "z") -> str:
        """
        Start point-to-point motion along a selected tool axis.

        Saves the current pose, computes tool-axis endpoints in base frame,
        and loops until stop.

        Args:
            amplitude_cm: Peak-to-peak amplitude in centimeters.
            speed_percent: 0–100% of max TCP speed.
            accel_percent: 0–100% of max TCP acceleration.
            axis: Tool axis for motion. One of "x", "y", or "z". Defaults to "z".

        Returns:
            str: Status string for UI.
        """
        with self._lock:
            if not self.state.connected or self._arm is None:
                return "Not connected"
            if self.state.playing:
                return "Already playing"
            try:
                # Save center pose
                code, pose = self._arm.get_position()
                if code != 0 or pose is None:
                    return f"Failed to read TCP pose (code {code})"
                self.state.init_pose = list(pose)

                # Ensure we are in position mode and ready
                ok, msg = self._set_mode_and_ready(0)
                if not ok:
                    return f"Failed to enter position mode: {msg}"

                self._stop_event.clear()
                self.state.playing = True
                self.state.last_play_speed_pct = float(max(0.0, min(100.0, speed_percent)))
                amp_mm = max(0.0, min(200.0, float(amplitude_cm) * 10.0))
                self._motion_thread = threading.Thread(
                    target=self._p2p_loop,
                    args=(amp_mm, float(speed_percent), float(accel_percent), axis),
                    daemon=True,
                )
                self._motion_thread.start()
                return "Point-to-point motion started"
            except Exception as e:
                self.state.playing = False
                self.state.last_error = f"Play error: {e}"
                return self.state.last_error

    def stop_motion(self) -> str:
        """
        Stop any active motion and return to the saved starting pose.

        Returns:
            str: Human-readable status string for the UI.
        """
        # Phase 1: signal stop and extract thread under lock
        thread: Optional[threading.Thread] = None
        with self._lock:
            if not self.state.connected or self._arm is None:
                return "Not connected"
            try:
                if self.state.playing:
                    self._stop_event.set()
                    thread = self._motion_thread
                self.state.playing = False
            except Exception as e:
                self.state.last_error = f"Stop error: {e}"
                return self.state.last_error
        # Phase 2: join without holding the lock
        if thread and thread.is_alive():
            thread.join(timeout=2.0)
        # Phase 3: finish shutdown under lock
        with self._lock:
            try:
                # Ensure controller is in position mode after stopping
                if self._arm is not None:
                    self._set_mode_and_ready(0)

                # If we have a saved pose, return to it in position mode
                if self._arm is not None and self.state.init_pose is not None:
                    vmax = float(self.state.max_tcp_lin_vel_mmps)
                    pct = max(0.0, min(100.0, self.state.last_play_speed_pct)) / 100.0
                    ret_speed = max(20.0, min(vmax, vmax * pct))  # mm/s

                    x, y, z, rx, ry, rz = list(self.state.init_pose)
                    code = self._arm.set_position(x, y, z, rx, ry, rz, speed=ret_speed, wait=True)
                    ok, msg = self._check(code, ok_msg="returned", fail_msg="set_position failed")
                    if not ok:
                        return f"Stopped, return pose error: {msg}"
                return "Stopped"
            except Exception as e:
                self.state.last_error = f"Stop error: {e}"
                return self.state.last_error
