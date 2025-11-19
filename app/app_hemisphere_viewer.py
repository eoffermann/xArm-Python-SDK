# app_hemisphere_viewer.py

"""
Gradio app to generate and visualize 3D points on the surface of a hemisphere,
with the option to download the point cloud as a .ply file.

Platform: Windows
Environment: Anaconda (Python 3.10.11)
Rendering: Plotly for 3D visualization
"""

import argparse
import numpy as np
import gradio as gr
import plotly.graph_objects as go
import tempfile
import os


def generate_hemisphere_points(radius: float,
                                lat_start: float,
                                lat_end: float,
                                lat_step: float,
                                lon_start: float,
                                lon_end: float,
                                lon_step: float) -> np.ndarray:
    """Generate Cartesian coordinates for points on a hemisphere.

    Args:
        radius (float): Radius of the hemisphere.
        lat_start (float): Starting latitude in degrees.
        lat_end (float): Ending latitude in degrees.
        lat_step (float): Step size for latitude.
        lon_start (float): Starting longitude in degrees.
        lon_end (float): Ending longitude in degrees.
        lon_step (float): Step size for longitude.

    Returns:
        np.ndarray: Array of shape (N, 3) with 3D point coordinates.
    """
    lat_range = np.radians(np.arange(lat_start, lat_end + 1e-6, lat_step))
    lon_range = np.radians(np.arange(lon_start, lon_end + 1e-6, lon_step))

    points = []
    for lat in lat_range:
        for lon in lon_range:
            x = radius * np.cos(lat) * np.cos(lon)
            y = radius * np.cos(lat) * np.sin(lon)
            z = radius * np.sin(lat)
            points.append([x, y, z])

    return np.array(points)


def plot_points_3d(points: np.ndarray) -> go.Figure:
    """Create a 3D scatter plot of points using Plotly.

    Args:
        points (np.ndarray): Array of shape (N, 3) with 3D coordinates.

    Returns:
        go.Figure: Plotly figure object.
    """
    fig = go.Figure(data=[
        go.Scatter3d(
            x=points[:, 0],
            y=points[:, 1],
            z=points[:, 2],
            mode='markers',
            marker=dict(size=3, color='blue')
        )
    ])
    fig.update_layout(scene=dict(
        xaxis_title='X',
        yaxis_title='Y',
        zaxis_title='Z'
    ))
    return fig


def export_to_ply(points: np.ndarray) -> str:
    """Export 3D points to a .ply file.

    Args:
        points (np.ndarray): Array of shape (N, 3).

    Returns:
        str: Path to the temporary .ply file.
    """
    header = f"""ply
format ascii 1.0
element vertex {len(points)}
property float x
property float y
property float z
end_header
"""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".ply", mode="w") as f:
        f.write(header)
        for pt in points:
            f.write(f"{pt[0]} {pt[1]} {pt[2]}\n")
        return f.name


def update_view(radius, lat_start, lat_end, lat_step,
                lon_start, lon_end, lon_step):
    points = generate_hemisphere_points(radius, lat_start, lat_end, lat_step,
                                        lon_start, lon_end, lon_step)
    fig = plot_points_3d(points)
    count = len(points)
    ply_path = export_to_ply(points)
    return fig, count, ply_path


def launch_gradio():
    """Launch the Gradio UI."""
    with gr.Blocks() as demo:
        gr.Markdown("## 3D Hemisphere Point Cloud Generator")
        with gr.Row():
            radius = gr.Slider(0.1, 10.0, value=1.0, label="Radius")
        with gr.Row():
            lat_start = gr.Slider(-90, 0, value=-45, step=1, label="Latitude Start")
            lat_end = gr.Slider(0, 90, value=45, step=1, label="Latitude End")
            lat_step = gr.Slider(1, 15, value=5, step=1, label="Latitude Step")
        with gr.Row():
            lon_start = gr.Slider(-90, 0, value=-60, step=1, label="Longitude Start")
            lon_end = gr.Slider(0, 90, value=60, step=1, label="Longitude End")
            lon_step = gr.Slider(1, 15, value=5, step=1, label="Longitude Step")

        with gr.Row():
            count_display = gr.Number(label="Number of Points", interactive=False)
        with gr.Row():
            plot = gr.Plot(label="3D Viewport")
        with gr.Row():
            download_btn = gr.File(label="Download .ply")

        inputs = [radius, lat_start, lat_end, lat_step,
                  lon_start, lon_end, lon_step]

        gr.Button("Generate").click(fn=update_view, inputs=inputs,
                                     outputs=[plot, count_display, download_btn])

    demo.launch()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Launch the 3D Gradio hemisphere point viewer.")
    args = parser.parse_args()
    launch_gradio()


if __name__ == "__main__":
    main()