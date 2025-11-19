"""
Gradio UI for uFactory xArm 850 Oscillation

This script defines a Gradio web UI for controlling a uFactory xArm 850 robot via a higher-level
XArmController wrapper. Through the browser, you can connect/disconnect to the robot by IP, toggle
drag/teach mode (making the arm draggable or not), and start or stop a point-to-point oscillation
motion whose amplitude (in cm), speed, and acceleration (as percentages of TCP max) are set with
sliders. The UI displays connection and teach-mode status labels plus a status textbox showing
controller messages, while a small CLI entrypoint lets you choose the default robot IP and Gradio
server bind/port options before launching the app.

"""
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

from xarm_controller_lib import XArmController



# -------------------- Gradio App --------------------
_controller = XArmController()


def ui_connect(ip: str) -> tuple[str, str, str]:
    """Gradio handler to connect to the robot.

    Args:
        ip: IP address from the UI textbox.

    Returns:
        Tuple of (status, connection badge, teach status badge).
    """
    status = _controller.connect(ip)
    conn_badge = (
        f"Connected ({_controller.state.sdk_version or 'SDK'})"
        if _controller.state.connected
        else "Disconnected"
    )
    teach_badge = "Draggable" if _controller.state.teach_enabled else "Not Draggable"
    return status, conn_badge, teach_badge


def ui_disconnect() -> tuple[str, str, str]:
    """Gradio handler to disconnect the robot.

    Returns:
        Tuple of (status, connection badge, teach status badge).
    """
    status = _controller.disconnect()
    return status, "Disconnected", "Not Draggable"


def ui_toggle_teach() -> tuple[str, str]:
    """Gradio handler to toggle teaching mode.

    Returns:
        A tuple of (status_text, draggable_label).
    """
    status, label = _controller.toggle_teach()
    return status, label


def ui_play(amplitude_cm: float, speed_percent: float, accel_percent: float) -> str:
    """Gradio handler to start point‑to‑point motion.

    Args:
        amplitude_cm: Amplitude in centimeters.
        speed_percent: Speed percentage.

    Returns:
        Status string.
    """
    return _controller.play_point_to_point(amplitude_cm, speed_percent, accel_percent)


def ui_stop() -> str:
    """Gradio handler to stop the motion and return to the initial pose.

    Returns:
        Status string.
    """
    return _controller.stop_motion()


def build_interface(default_ip: str = "192.168.1.221") -> gr.Blocks:
    """Construct the Gradio Blocks UI.

    Args:
        default_ip: Default IP to prefill in the textbox.

    Returns:
        A ready-to-launch gradio.Blocks app.
    """
    with gr.Blocks(title="xArm 850 — Connect • Teach • P2P Motion") as demo:
        gr.Markdown("# xArm 850 — Connect • Teach • Point-to-Point Motion")
        with gr.Row():
            ip = gr.Textbox(label="Robot IP", value=default_ip, interactive=True)
            btn_connect = gr.Button("Connect", variant="primary")
            btn_disconnect = gr.Button("Disconnect")
        with gr.Row():
            teach_toggle = gr.Button("Toggle Drag/Teach")
            teach_status = gr.Label(value="Not Draggable", label="Teach Status")
            conn_status = gr.Label(value="Disconnected", label="Connection")
        with gr.Row():
            amp = gr.Slider(
                minimum=0.0,
                maximum=20.0,
                step=0.1,
                value=5.0,
                label="Amplitude (cm)",
            )
            spd = gr.Slider(minimum=0, maximum=100, step=1, value=50, label="Speed (% of TCP max)")
            acc = gr.Slider(minimum=0, maximum=100, step=1, value=50, label="Accel (% of TCP max)")
        with gr.Row():
            play = gr.Button("Play", variant="primary")
            stop = gr.Button("Stop")
        status_out = gr.Textbox(label="Status", value="", interactive=False)

        # Wiring
        btn_connect.click(ui_connect, inputs=[ip], outputs=[status_out, conn_status, teach_status])
        btn_disconnect.click(ui_disconnect, outputs=[status_out, conn_status, teach_status])
        teach_toggle.click(ui_toggle_teach, outputs=[status_out, teach_status])
        play.click(ui_play, inputs=[amp, spd, acc], outputs=[status_out])
        stop.click(ui_stop, outputs=[status_out])
    return demo


def main() -> None:
    """CLI entrypoint to launch the Gradio UI.

    Command-line options:
        --ip: Default robot IP to prefill in the UI (default 192.168.1.221).
        --server-name: Host to bind (default 127.0.0.1).
        --server-port: Port to bind (default 7860).
        --share: If passed, enables Gradio share link.
    """
    import argparse

    parser = argparse.ArgumentParser(description="xArm 850 Gradio UI: Teach & Sine Motion")
    parser.add_argument("--ip", default="192.168.1.221", help="Default robot IP for the textbox")
    parser.add_argument("--server-name", default="127.0.0.1", help="Host/IP to bind the Gradio server")
    parser.add_argument("--server-port", type=int, default=7860, help="Port to bind the Gradio server")
    parser.add_argument("--share", action="store_true", help="Enable external share link (use with caution)")
    args = parser.parse_args()

    demo = build_interface(default_ip=args.ip)
    demo.launch(server_name=args.server_name, server_port=args.server_port, share=args.share)


if __name__ == "__main__":
    main()
