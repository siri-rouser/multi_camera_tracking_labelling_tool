#!/usr/bin/env python3
# coding: utf-8

"""
Visualize inter-camera time transition KDE.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity


def load_data(json_path, direction):
    with open(json_path, "r") as f:
        data_dict = json.load(f)
    if direction not in data_dict:
        raise ValueError(f"Direction '{direction}' not found in JSON keys: {list(data_dict.keys())}")
    data = np.array(data_dict[direction], dtype=float)
    if data.size == 0:
        raise ValueError(f"No data for direction {direction}")
    return data


def plot_kde(data, bandwidth=5.0,  direction=""):
    """
    Args:
        data: np.ndarray of frame gaps
        bandwidth: KDE bandwidth in frame units
        fps: frames per second (convert to seconds)
        direction: for plot title
    """
    # Convert to seconds
    data_sec = data 
    kde = KernelDensity(kernel="gaussian", bandwidth=bandwidth)
    kde.fit(data_sec.reshape(-1, 1))

    # Evaluate KDE on a grid
    x_d = np.linspace(0, 200, 200).reshape(-1, 1)
    log_density = kde.score_samples(x_d)
    density = np.exp(log_density)

    # Plot
    plt.figure(figsize=(8, 4))
    plt.fill_between(x_d.flatten(), density, alpha=0.5, color="skyblue")
    plt.plot(data_sec, np.full_like(data_sec, -0.005), "|k", markeredgewidth=1)
    plt.title(f"KDE of Time Transition ({direction})")
    plt.xlabel("Time Transition (seconds)")
    plt.ylabel("Density")
    plt.grid(True, alpha=0.3)
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Visualize KDE of camera time transitions.")
    parser.add_argument("json_path", type=str, help="Path to MTC_time_transition_GT.json")
    parser.add_argument("--direction", type=str, required=True, help="Transition direction (e.g., '1_to_2')")
    parser.add_argument("--bandwidth", type=float, default=5.0, help="KDE bandwidth (frames)")
    args = parser.parse_args()

    data = load_data(Path(args.json_path), args.direction)
    plot_kde(data, bandwidth=args.bandwidth,  direction=args.direction)


if __name__ == "__main__":
    main()

# example usage: python CLM_vis.py MTC_time_transition_GT.json --direction 2_to_3