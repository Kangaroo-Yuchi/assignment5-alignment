#!/usr/bin/env python3
"""
General-purpose experiment plotting for GRPO and similar RL training runs.

Usage:
    # Fetch from RunPod and plot (auto-syncs remote -> local):
    python plot_experiments.py --remote 38.80.152.249:30405 /workspace/assignment5-alignment/grpo_comparison

    # Plot from already-downloaded local data:
    python plot_experiments.py grpo_comparison

    # Plot specific local experiment dirs:
    python plot_experiments.py grpo_comparison/reinforce_with_baseline grpo_comparison/no_baseline

    # Custom output path and title:
    python plot_experiments.py --remote 38.80.152.249:30405 /workspace/grpo_comparison -o my_plot.png -t "My Experiment"

Directory structure expected (remote or local):
    grpo_comparison/
        reinforce_with_baseline/
            rl_evaluation_results_step_10.jsonl
            rl_evaluation_results_step_20.jsonl
            ...
        no_baseline/
            rl_evaluation_results_step_10.jsonl
            ...
"""

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.dpi": 150,
})

# Consistent color palette for experiments
COLORS = [
    "#2196F3",  # blue
    "#FF5722",  # deep orange
    "#4CAF50",  # green
    "#9C27B0",  # purple
    "#FF9800",  # orange
    "#00BCD4",  # cyan
    "#E91E63",  # pink
    "#795548",  # brown
]

MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]

SSH_KEY = "~/.ssh/id_ed25519_cs336"
SSH_USER = "root"


def scp_from_remote(host: str, port: int, remote_path: str, local_path: Path):
    """scp JSONL files from a remote host to a local directory."""
    local_path.mkdir(parents=True, exist_ok=True)

    ssh_opts = ["-p", str(port), "-i", SSH_KEY, "-o", "StrictHostKeyChecking=no"]
    remote = f"{SSH_USER}@{host}"

    # List remote subdirectories and top-level jsonl files
    print(f"Syncing from {remote}:{port}:{remote_path} -> {local_path}")
    ls_cmd = ["ssh", *ssh_opts, remote,
              f"find {remote_path} -name '*.jsonl' -printf '%P\\n'"]
    result = subprocess.run(ls_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"ssh ls stderr:\n{result.stderr}", file=sys.stderr)
        sys.exit(1)

    remote_files = [f for f in result.stdout.strip().split("\n") if f]
    if not remote_files:
        print("No .jsonl files found on remote.")
        return

    # Create local subdirectories and scp each file (skip if already exists locally)
    new_files = []
    for rel in remote_files:
        local_file = local_path / rel
        if local_file.exists():
            continue
        new_files.append(rel)
        local_file.parent.mkdir(parents=True, exist_ok=True)

    if not new_files:
        print(f"All {len(remote_files)} files already present locally.")
        return

    print(f"Downloading {len(new_files)} new file(s) ({len(remote_files) - len(new_files)} already cached)...")
    scp_opts = ["-P", str(port), "-i", SSH_KEY, "-o", "StrictHostKeyChecking=no"]
    for rel in new_files:
        scp_cmd = ["scp", *scp_opts,
                    f"{remote}:{remote_path}/{rel}",
                    str(local_path / rel)]
        r = subprocess.run(scp_cmd, capture_output=True, text=True)
        if r.returncode != 0:
            print(f"scp failed for {rel}: {r.stderr}", file=sys.stderr)
    print(f"Sync complete.")


def parse_experiment_dir(exp_dir: Path) -> dict | None:
    """
    Parse a directory of JSONL evaluation files into a dict of metrics per step.

    Returns:
        {
            "name": str,
            "steps": [10, 20, ...],
            "accuracy": [0.45, 0.52, ...],
            "format_reward": [0.8, 0.85, ...],
            "reward": [0.4, 0.5, ...],
        }
    """
    step_pattern = re.compile(r"rl_evaluation_results_(?:lr_\S+_)?step_(\d+)\.jsonl")

    step_metrics = {}
    for f in sorted(exp_dir.iterdir()):
        m = step_pattern.match(f.name)
        if not m:
            continue
        step = int(m.group(1))

        total_accuracy = 0.0
        total_format = 0.0
        total_reward = 0.0
        count = 0
        wall_clock_time = None
        mean_entropy = None
        mean_response_length = None

        with open(f) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                # First line may be a metadata record
                if obj.get("_meta"):
                    wall_clock_time = obj.get("wall_clock_time")
                    mean_entropy = obj.get("mean_entropy")
                    mean_response_length = obj.get("mean_response_length")
                    continue
                scores = obj["scores"]
                total_accuracy += 1.0 if scores["answer_reward"] > 0 else 0.0
                total_format += scores["format_reward"]
                total_reward += scores["reward"]
                count += 1

        if count > 0:
            step_metrics[step] = {
                "accuracy": total_accuracy / count,
                "format_reward": total_format / count,
                "reward": total_reward / count,
                "wall_clock_time": wall_clock_time,
                "entropy": mean_entropy,
                "response_length": mean_response_length,
            }

    if not step_metrics:
        return None

    sorted_steps = sorted(step_metrics.keys())
    return {
        "name": exp_dir.name.replace("_", " "),
        "steps": sorted_steps,
        "accuracy": [step_metrics[s]["accuracy"] for s in sorted_steps],
        "format_reward": [step_metrics[s]["format_reward"] for s in sorted_steps],
        "reward": [step_metrics[s]["reward"] for s in sorted_steps],
        "wall_clock_time": [step_metrics[s]["wall_clock_time"] for s in sorted_steps],
        "entropy": [step_metrics[s]["entropy"] for s in sorted_steps],
        "response_length": [step_metrics[s]["response_length"] for s in sorted_steps],
    }


def discover_experiments(paths: list[Path]) -> list[dict]:
    """
    Given a list of paths, discover experiments.
    If a path is a directory containing JSONL files, treat it as one experiment.
    If a path is a parent directory containing subdirectories with JSONL files,
    treat each subdirectory as an experiment.
    """
    experiments = []
    for p in paths:
        if not p.is_dir():
            print(f"Warning: {p} is not a directory, skipping.")
            continue

        # Check if this directory itself has JSONL files
        jsonl_files = list(p.glob("rl_evaluation_results_*.jsonl"))
        if jsonl_files:
            exp = parse_experiment_dir(p)
            if exp:
                experiments.append(exp)
        else:
            # Check subdirectories
            for sub in sorted(p.iterdir()):
                if sub.is_dir():
                    exp = parse_experiment_dir(sub)
                    if exp:
                        experiments.append(exp)

    return experiments


def plot_experiments(experiments: list[dict], output_path: Path, title: str = None):
    """
    Plot side-by-side metric curves for multiple experiments.
    Row 1 (vs training step): accuracy, format_reward, reward, entropy, response_length
    Row 2 (vs wall-clock time): same metrics, if wall-clock data is available
    """
    metric_labels = {
        "accuracy": "Answer Accuracy",
        "format_reward": "Format Reward",
        "reward": "Overall Reward",
        "entropy": "Mean Token Entropy",
        "response_length": "Mean Response Length (tokens)",
    }

    # Check which extra metrics have data
    active_metrics = ["accuracy", "format_reward", "reward"]
    for m in ["entropy", "response_length"]:
        if any(any(v is not None for v in exp.get(m, [])) for exp in experiments):
            active_metrics.append(m)

    # Check if any experiment has wall-clock data
    has_wallclock = any(
        any(t is not None for t in exp.get("wall_clock_time", []))
        for exp in experiments
    )

    n_metrics = len(active_metrics)
    n_rows = 2 if has_wallclock else 1
    fig, axes = plt.subplots(n_rows, n_metrics, figsize=(5 * n_metrics, 5 * n_rows), squeeze=False)

    def _plot_row(axes_row, x_key, xlabel):
        for ax, metric in zip(axes_row, active_metrics):
            for i, exp in enumerate(experiments):
                color = COLORS[i % len(COLORS)]
                marker = MARKERS[i % len(MARKERS)]
                if x_key == "steps":
                    xs = exp["steps"]
                else:
                    xs = exp.get("wall_clock_time", [None] * len(exp["steps"]))
                ys = exp.get(metric, [None] * len(exp["steps"]))
                pairs = [(x, y) for x, y in zip(xs, ys) if x is not None and y is not None]
                if not pairs:
                    continue
                px, py = zip(*pairs)
                ax.plot(px, py, marker=marker, color=color, label=exp["name"],
                        markersize=5, linewidth=2, alpha=0.9)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(metric_labels[metric])
            ax.set_title(metric_labels[metric])
            ax.legend(loc="best")
            ax.grid(True, alpha=0.3)
            if metric in ("accuracy", "format_reward", "reward"):
                ax.set_ylim(0, 1.05)

    _plot_row(axes[0], "steps", "Training Step")
    if has_wallclock:
        _plot_row(axes[1], "wall_clock_time", "Wall-Clock Time (s)")

    suptitle = title or "GRPO Training: Experiment Comparison"
    fig.suptitle(suptitle, fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(output_path, dpi=150, facecolor="white")
    plt.close(fig)
    print(f"Saved plot to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Plot evaluation metrics from GRPO experiment results.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "paths",
        nargs="+",
        type=str,
        help="Remote path (with --remote) or local directory/directories to plot.",
    )
    parser.add_argument(
        "--remote",
        type=str,
        default=None,
        metavar="HOST:PORT",
        help="RunPod SSH endpoint as HOST:PORT (e.g. 38.80.152.249:30405). "
             "Data is rsynced to a local mirror before plotting.",
    )
    parser.add_argument(
        "--local-dir",
        type=Path,
        default=None,
        help="Local directory to sync remote data into (default: ./grpo_comparison)",
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=None,
        help="Output path for the plot image (default: <data_dir>/comparison.png)",
    )
    parser.add_argument(
        "-t", "--title",
        type=str,
        default=None,
        help="Plot title",
    )

    args = parser.parse_args()

    if args.remote:
        # Parse host:port
        parts = args.remote.split(":")
        if len(parts) != 2:
            print("Error: --remote must be HOST:PORT (e.g. 38.80.152.249:30405)")
            sys.exit(1)
        host, port = parts[0], int(parts[1])

        # Sync each remote path to local mirror
        local_paths = []
        for remote_path in args.paths:
            # Default local dir: use the basename of the remote path
            if args.local_dir:
                local_dir = args.local_dir
            else:
                local_dir = Path(Path(remote_path).name)
            scp_from_remote(host, port, remote_path, local_dir)
            local_paths.append(local_dir)
    else:
        local_paths = [Path(p) for p in args.paths]

    experiments = discover_experiments(local_paths)
    if not experiments:
        print("No experiment data found. Check that directories contain "
              "rl_evaluation_results_step_*.jsonl files.")
        return

    print(f"Found {len(experiments)} experiment(s):")
    for exp in experiments:
        print(f"  - {exp['name']} ({len(exp['steps'])} steps: {exp['steps'][0]}..{exp['steps'][-1]})")

    output = args.output or local_paths[0] / "comparison.png"
    plot_experiments(experiments, output, title=args.title)


if __name__ == "__main__":
    main()