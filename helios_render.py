#!/usr/bin/env python3
"""
Helios Data Visualizer

Installation:
1. Create a virtual environment:
   python -m venv venv
   source venv/bin/activate  # On Windows: venv/Scripts/activate

2. Install dependencies:
   pip install -r requirements.txt

3. Install FFmpeg (required for video generation):
   - Windows: Download from https://ffmpeg.org/download.html and add to PATH
   - macOS: brew install ffmpeg
   - Linux: sudo apt install ffmpeg

Usage:
   python helios_render.py --csv cmi-detect-behavior-with-sensor-data/train.csv --subject SUBJ_032761 --gesture "Wave hello"
   python helios_render.py --csv cmi-detect-behavior-with-sensor-data/train.csv --subject SUBJ_032761 --gesture "Neck - pinch skin" --sequence_index 3
"""

import os
import numpy as np
import pandas as pd
import pyvista as pv
from scipy.spatial.transform import Rotation as R
import argparse
from tqdm import tqdm
import pathlib
import time
import matplotlib.pyplot as plt
import tempfile
import subprocess

WINDOW_SIZE = [1280, 720]
VIDEO_SIZE = (16, 9)
DPI = 120
TOF_GRID_SIZE = (8, 8)
NUM_TOF_SENSORS = 5
NUM_THERMOPILE_SENSORS = 5

COLORS = {
    "background": [0.05, 0.05, 0.07],
    "background_top": [0.12, 0.12, 0.16],
    "mesh": "#e0e0e8",
    "text": "#ffffff",
    "text_secondary": "#e0e0e8",
    "text_tertiary": "#c0c0d0",
    "title_bg": "#1a1a2e",
    "figure_bg": "#0a0a12",
    "spine": "#4a4a6a",
}


def create_transform_matrix(rotation_matrix):
    """Create 4x4 transformation matrix from 3x3 rotation matrix."""
    transform_matrix = np.eye(4)
    transform_matrix[:3, :3] = rotation_matrix
    return transform_matrix


def extract_tof_data(row, sensor_num):
    """Extract ToF sensor data and reshape to 8x8 grid."""
    tof_data = []
    for i in range(64):
        column_name = f"tof_{sensor_num}_v{i}"
        if column_name in row.index:
            value = row[column_name]
            tof_data.append(np.nan if value == -1 else float(value))
        else:
            tof_data.append(np.nan)
    return np.array(tof_data).reshape(TOF_GRID_SIZE)


def load_data(csv_path, filters=None, use_cache=True):
    """Load and filter data with optional caching."""
    csv_path = pathlib.Path(csv_path)
    cache_dir = pathlib.Path("./data_cache")
    cache_dir.mkdir(exist_ok=True)
    cache_file = cache_dir / f"{csv_path.stem}.parquet"

    if use_cache and cache_file.exists():
        print(f"Loading data from cache: {cache_file}")
        start_time = time.time()
        df = pd.read_parquet(cache_file)
        print(f"Loaded from cache in {time.time() - start_time:.2f} seconds")
    else:
        print(f"Loading data from {csv_path}...")
        start_time = time.time()
        df = pd.read_csv(csv_path)
        print(f"Loaded CSV in {time.time() - start_time:.2f} seconds")

        if use_cache:
            print(f"Creating cache file: {cache_file}")
            df.to_parquet(cache_file, index=False)

    # Apply filters
    if filters:
        for column, value in filters.items():
            if value:
                df = df[df[column] == value]
                if df.empty:
                    raise ValueError(f"No data found for {column} = {value}")
                print(f"Filtered for {column}: {value}")

    unique_sequence_ids = df["sequence_id"].unique()
    if len(unique_sequence_ids) == 0:
        raise ValueError("No valid sequences found in the dataset")

    print(f"Found {len(unique_sequence_ids)} sequences")
    return df, unique_sequence_ids


def load_arm_mesh():
    """Load arm mesh from file."""
    mesh_path = "arm.obj"
    if not os.path.exists(mesh_path):
        raise FileNotFoundError(f"Required arm mesh file not found: {mesh_path}")

    print(f"Loading arm mesh from: {mesh_path}")
    return pv.read(mesh_path)


def check_ffmpeg():
    """Check if FFmpeg is available."""
    try:
        subprocess.run(
            ["ffmpeg", "-version"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        )
        return True
    except (subprocess.SubprocessError, FileNotFoundError):
        return False


def prepare_sequence_data(df, sequence_id):
    """Prepare sequence data for rendering."""
    sequence_df = df[df["sequence_id"] == sequence_id].copy()
    sequence_df = sequence_df.sort_values(by=["sequence_counter"])

    sequence_data = []
    for _, row in sequence_df.iterrows():
        quat = [float(row[col]) for col in ["rot_w", "rot_x", "rot_y", "rot_z"]]
        acc = [float(row[col]) for col in ["acc_x", "acc_y", "acc_z"]]

        sequence_data.append(
            {
                "row": row,
                "sequence_counter": int(row["sequence_counter"]),
                "row_id": row["row_id"],
                "quat": quat,
                "acc": acc,
                "gesture": row["gesture"],
                "behavior": row["behavior"],
                "orientation": row["orientation"],
                "phase": row["phase"],
                "sequence_type": row["sequence_type"],
            }
        )

    return sequence_data, sequence_df["gesture"].iloc[0]


def setup_plotter():
    """Setup PyVista plotter with consistent lighting and camera."""
    plotter = pv.Plotter(notebook=False, off_screen=True, window_size=WINDOW_SIZE)
    plotter.set_background(COLORS["background"], top=COLORS["background_top"])
    plotter.add_axes(interactive=True, line_width=2)
    plotter.camera_position = "yz"
    plotter.camera.up = [0, -1, 0]

    # Remove default lights and add custom lighting
    plotter.remove_all_lights()
    light_configs = [
        {"position": (0, 10, 10), "color": [1, 1, 1], "intensity": 0.7},
        {"position": (10, -5, 0), "color": [0.9, 0.9, 1], "intensity": 0.5},
        {"position": (-10, -10, -10), "color": [0.7, 0.7, 0.8], "intensity": 0.3},
    ]

    for config in light_configs:
        light = pv.Light(
            position=config["position"],
            focal_point=(0, 0, 0),
            color=config["color"],
            intensity=config["intensity"],
        )
        plotter.add_light(light)

    return plotter


def create_sensor_subplot(fig, frame_data, sensor_num, subplot_pos):
    """Create a single ToF sensor visualization."""
    ax = plt.subplot(subplot_pos)
    ax.set_facecolor(COLORS["figure_bg"])

    tof_data = extract_tof_data(frame_data["row"], sensor_num)
    ax.imshow(tof_data, cmap=plt.cm.plasma, vmin=0, vmax=254, interpolation="bilinear")
    ax.set_title(
        f"ToF {sensor_num}",
        fontsize=10,
        color=COLORS["text_secondary"],
        fontweight="medium",
    )
    ax.set_xticks([])
    ax.set_yticks([])

    for spine in ax.spines.values():
        spine.set_edgecolor(COLORS["spine"])
        spine.set_linewidth(0.8)


def create_thermopile_subplot(fig, frame_data, subplot_pos):
    """Create thermopile sensor visualization."""
    ax = plt.subplot(subplot_pos)
    ax.set_facecolor(COLORS["figure_bg"])

    # Extract thermopile data
    therm_data = []
    for j in range(1, NUM_THERMOPILE_SENSORS + 1):
        col_name = f"thm_{j}"
        if col_name in frame_data["row"].index:
            therm_data.append(float(frame_data["row"][col_name]))
        else:
            therm_data.append(np.nan)

    y_pos = np.arange(NUM_THERMOPILE_SENSORS)

    # Normalize temperatures for color mapping (20-40°C range)
    temp_min, temp_max = 20, 40
    normalized_temps = [
        (temp - temp_min) / (temp_max - temp_min) if not np.isnan(temp) else 0.5
        for temp in therm_data
    ]

    # colors based on temperature values
    colors = plt.cm.magma([max(0, min(1, norm_temp)) for norm_temp in normalized_temps])

    ax.barh(y_pos, therm_data, color=colors, height=0.7, edgecolor="none")

    ax.set_xlim(20, 40)  # Set min and max temperature range in Celsius

    ax.set_yticks(y_pos)
    ax.set_yticklabels(
        [f"{j + 1}" for j in range(NUM_THERMOPILE_SENSORS)],
        fontsize=10,
        color=COLORS["text_secondary"],
    )
    ax.set_title(
        "Thermopile", fontsize=10, color=COLORS["text_secondary"], fontweight="medium"
    )
    ax.set_xlabel("°C", fontsize=10, color=COLORS["text_tertiary"])
    ax.tick_params(
        axis="both", which="major", labelsize=9, colors=COLORS["text_tertiary"]
    )
    ax.grid(axis="x", linestyle="--", alpha=0.15, color=COLORS["text_tertiary"])

    for spine in ax.spines.values():
        spine.set_edgecolor(COLORS["spine"])
        spine.set_linewidth(0.8)


def create_frame_info_text(frame_data):
    """Generate frame information text."""
    return (
        f"Subject: {frame_data['row']['subject']} | "
        f"Gesture: {frame_data['gesture']} | "
        f"Orientation: {frame_data['orientation']} | "
        f"Phase: {frame_data['phase']} | "
        f"Counter: {frame_data['sequence_counter']}"
    )


def render_frame_with_sensors(
    plotter, mesh, original_points, frame_data, frame_index, temp_dir
):
    """Render a single frame with sensor data."""
    # transform mesh
    mesh.points = original_points.copy()
    rot = R.from_quat(frame_data["quat"])
    transform = create_transform_matrix(rot.as_matrix())
    mesh.transform(transform, inplace=True)

    # render 3D view
    plotter.clear_actors()
    plotter.add_mesh(
        mesh,
        color=COLORS["mesh"],
        specular=0.3,
        specular_power=5,
        ambient=0.5,
        diffuse=0.6,
        smooth_shading=False,
    )
    pv_img = plotter.screenshot(return_img=True)

    # create composite figure
    fig = plt.figure(figsize=VIDEO_SIZE, dpi=DPI, facecolor=COLORS["figure_bg"])
    gs = plt.GridSpec(2, 1, height_ratios=[2, 1], figure=fig)

    ax_3d = plt.subplot(gs[0])
    ax_3d.imshow(pv_img)
    ax_3d.set_xticks([])
    ax_3d.set_yticks([])
    ax_3d.axis("off")
    ax_3d.set_title(
        create_frame_info_text(frame_data),
        fontsize=11,
        color=COLORS["text"],
        backgroundcolor=COLORS["title_bg"],
        pad=10,
        fontweight="medium",
    )

    # sensor subplots
    gs_sensors = plt.GridSpec(1, 6, wspace=0.3, figure=fig)
    gs_sensors.update(top=0.45, bottom=0.05, left=0.05, right=0.95)

    for j in range(NUM_TOF_SENSORS):
        create_sensor_subplot(fig, frame_data, j + 1, gs_sensors[0, j])

    create_thermopile_subplot(fig, frame_data, gs_sensors[0, 5])

    # save frame
    frame_file = os.path.join(temp_dir, f"frame_{frame_index:05d}.png")
    plt.savefig(
        frame_file,
        facecolor=COLORS["figure_bg"],
        bbox_inches="tight",
        pad_inches=0.1,
        dpi=DPI,
    )
    plt.close(fig)
    return frame_file


def create_video_from_frames(frame_files, output_file, framerate, temp_dir):
    """Combine frames into video using FFmpeg."""
    print(f"Combining frames into video: {output_file}")
    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)

    ffmpeg_cmd = [
        "ffmpeg",
        "-y",
        "-framerate",
        str(framerate),
        "-i",
        os.path.join(temp_dir, "frame_%05d.png"),
        "-c:v",
        "libx264",
        "-profile:v",
        "high",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        "-vf",
        "scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:(oh-ih)/2,setsar=1:1",
        output_file,
    ]

    try:
        subprocess.run(ffmpeg_cmd, check=True)
        print(f"Video successfully created: {output_file}")

        # Cleanup
        for file in frame_files:
            os.remove(file)
        os.rmdir(temp_dir)

    except subprocess.SubprocessError as e:
        print(f"Error creating video with FFmpeg: {e}")
        print(f"Individual frames are saved in {temp_dir}")


def render_animation(sequence_data, output_file, framerate=10):
    """Main rendering function with sensor visualizations."""
    if not check_ffmpeg():
        raise RuntimeError(
            "FFmpeg is required but not found in system PATH. Please install FFmpeg."
        )

    mesh = load_arm_mesh()
    original_points = mesh.points.copy()
    temp_dir = tempfile.mkdtemp()
    print(f"Creating composite frames in: {temp_dir}")

    plotter = setup_plotter()
    frame_files = []

    print(f"Rendering {len(sequence_data)} frames...")
    for i, frame_data in enumerate(tqdm(sequence_data, desc="Creating frames")):
        frame_file = render_frame_with_sensors(
            plotter, mesh, original_points, frame_data, i, temp_dir
        )
        frame_files.append(frame_file)

    plotter.close()
    create_video_from_frames(frame_files, output_file, framerate, temp_dir)


def main():
    parser = argparse.ArgumentParser(
        description="Render 3D motion data animation from sensor data"
    )
    parser.add_argument("--csv", required=True, help="Path to the CSV data file")
    parser.add_argument("--subject", help="Subject ID to filter (e.g., SUBJ_059520)")
    parser.add_argument("--gesture", help="Gesture description to filter")
    parser.add_argument("--behavior", help="Behavior description to filter")
    parser.add_argument("--phase", help="Phase to filter")
    parser.add_argument("--orientation", help="Orientation to filter")
    parser.add_argument(
        "--output_dir", default="./outputs", help="Output directory for videos"
    )
    parser.add_argument(
        "--framerate", type=int, default=10, help="Frame rate for output video"
    )
    parser.add_argument(
        "--sequence_index",
        type=int,
        default=0,
        help="Index of the sequence to render (if multiple sequences match)",
    )
    parser.add_argument(
        "--no-cache", action="store_true", help="Disable parquet caching"
    )

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    if not check_ffmpeg():
        raise RuntimeError(
            "FFmpeg is required but not found in system PATH. Please install it:\n"
            "  - Windows: Download from https://ffmpeg.org/download.html and add to PATH\n"
            "  - macOS: Run 'brew install ffmpeg' (requires Homebrew)\n"
            "  - Linux: Run 'sudo apt install ffmpeg' or equivalent for your distro"
        )

    # Prepare filters
    filters = {
        "subject": args.subject,
        "gesture": args.gesture,
        "behavior": args.behavior,
        "phase": args.phase,
        "orientation": args.orientation,
    }

    df, unique_sequence_ids = load_data(args.csv, filters, use_cache=not args.no_cache)

    # Select sequence
    if args.sequence_index >= len(unique_sequence_ids):
        print(
            f"Warning: Selected index {args.sequence_index} out of range. Using first sequence."
        )
        selected_sequence_id = unique_sequence_ids[0]
    else:
        selected_sequence_id = unique_sequence_ids[args.sequence_index]

    sequence_data, gesture_name = prepare_sequence_data(df, selected_sequence_id)

    # output filename
    subject = (
        args.subject
        or df.loc[df["sequence_id"] == selected_sequence_id, "subject"].iloc[0]
    )
    safe_gesture_name = gesture_name.replace(" - ", "_").replace(" ", "_")
    output_file = os.path.join(
        args.output_dir, f"{subject}_{safe_gesture_name}_{selected_sequence_id}.mp4"
    )

    render_animation(sequence_data, output_file, args.framerate)

    print("Animation complete!")
    print(f"Output file: {output_file}")


if __name__ == "__main__":
    main()
