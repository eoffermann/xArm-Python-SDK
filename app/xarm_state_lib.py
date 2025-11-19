from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass
from typing import Optional, Tuple, List

import gradio as gr

try:
    from xarm.wrapper import XArmAPI  # type: ignore
except Exception:  # pragma: no cover
    XArmAPI = None  # type: ignore

@dataclass
class ArmState:
    """Holds connection and runtime state for the xArm UI.

    Attributes:
        connected: Whether the SDK connection is active.
        ip: Current target IP address.
        teach_enabled: Whether joint teaching (drag) mode is enabled.
        playing: Whether the sine motion is active.
        init_pose: Saved TCP pose [x, y, z, roll, pitch, yaw] from get_position().
        sdk_version: Firmware/API version string if available.
        last_error: Last human‑readable error/status message.
        max_tcp_lin_vel_mmps: Controller‑reported TCP max linear velocity (mm/s).
        max_tcp_lin_acc_mmps2: Controller‑reported TCP max linear acceleration (mm/s^2).
        last_play_speed_pct: Speed percentage used at last Play for use on return‑to‑pose.
        is_radian: Whether the SDK expects radians for orientation.
    """

    connected: bool = False
    ip: str = "192.168.1.221"
    teach_enabled: bool = False
    playing: bool = False
    init_pose: Optional[List[float]] = None
    sdk_version: Optional[str] = None
    last_error: str = ""
    max_tcp_lin_vel_mmps: float = 200.0
    max_tcp_lin_acc_mmps2: float = 1000.0
    last_play_speed_pct: float = 50.0
    is_radian: bool = False

