"""Render the preserved diagnostic observations, without a performance claim.

Run with Python and Matplotlib: python docs/render-pilot-figure.py
The training environment does not need Matplotlib.
"""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    folder = Path(__file__).resolve().parent
    report = json.loads((folder / "recovery-pilot-report.json").read_text())
    setting = report["supervised_setting"]
    plt.rcParams.update({"font.size": 10, "svg.hashsalt": report["run_id"]})
    fig, (timing, memory) = plt.subplots(1, 2, figsize=(10, 4.7))
    seconds = [setting["cold_step_seconds"], *setting["warmed_seconds"]]
    labels = ["First update\n(includes compilation)", "Warmed update 2", "Warmed update 3"]
    bars = timing.barh(labels, seconds, color=["#7c8994", "#276a8c", "#276a8c"])
    timing.invert_yaxis()
    timing.set_xlim(0, max(seconds) * 1.25)
    timing.set_xlabel("Synchronized update time (seconds)")
    timing.set_title("Supervised diagnostic, microbatch 1", fontsize=11)
    for bar, value in zip(bars, seconds):
        timing.text(value + 12, bar.get_y() + bar.get_height() / 2, f"{value:.1f}", va="center")

    devices = setting["measured_peak_memory"]
    peaks = [d["peak_bytes"] / 2**30 for d in devices]
    limit = devices[0]["limit_bytes"] / 2**30
    memory.bar([f"TPU {d['device']}" for d in devices], peaks, color="#276a8c")
    memory.axhline(limit, color="#3b4349", linewidth=1.2, label="Reported memory limit")
    memory.axhline(limit * .85, color="#7c8994", linestyle="--", label="15% headroom threshold")
    memory.set_ylim(0, limit * 1.2)
    memory.set_ylabel("Observed peak device memory (GiB)")
    memory.set_xlabel("Device")
    memory.set_title("Minimum measured headroom: 39.8%", fontsize=11)
    memory.legend(frameon=False, fontsize=8, loc="upper left")
    for axis in (timing, memory):
        axis.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Qwen3-1.7B: three finite supervised pilot updates", fontsize=13)
    fig.text(.02, .02, "Source: owned TPU run 20261007-220801, October 7, 2026; live operator observations.\n"
             "The larger-batch pilot was incomplete. These diagnostics establish no held-out or RL improvement.", fontsize=8)
    fig.tight_layout(rect=(0, .12, 1, .93))
    output = folder / "figures"
    output.mkdir(exist_ok=True)
    fig.savefig(output / "recovery-pilot.png", dpi=180)
    fig.savefig(output / "recovery-pilot.svg", metadata={"Date": None})
    svg = output / "recovery-pilot.svg"
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")
    plt.close(fig)


if __name__ == "__main__":
    main()
