"""Draw exactly two event-window figures from the fixed simulator experiments."""
from pathlib import Path
import hashlib
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import ticker


def load(root, name, duration, event):
    meta = json.loads((root / f"{name}.json").read_text(encoding="utf-8"))
    if (meta["status"] != "completed" or meta["completed_steps"] != round(duration / 1e-6)
            or abs(meta["event_time_s"] - event) > 1e-12):
        raise ValueError(f"{name}: incomplete or wrong simulation case")
    data = np.genfromtxt(root / f"{name}.csv", delimiter=",", names=True)
    if not all(np.isfinite(data[n]).all() for n in data.dtype.names):
        raise ValueError(f"{name}: CSV contains nonfinite values")
    if not np.all(np.diff(data["time_s"]) > 0):
        raise ValueError(f"{name}: timestamps must increase")
    if abs(data["time_s"][-1] - duration) > 1e-12:
        raise ValueError(f"{name}: wrong final time")
    return data


def select(data, start, end):
    return (data["time_s"] >= start - 1e-12) & (data["time_s"] < end - 1e-12)


def canvas(title, start, end, event, event_label):
    fig, ax = plt.subplots(figsize=(12, 5.9))
    fig.subplots_adjust(left=.09, right=.98, bottom=.25, top=.82)
    fig.suptitle(title, x=.09, y=.97, ha="left", fontsize=17, fontweight="bold")
    ax.set_title("Time step: 1 us | PWM: 10 kHz | Fundamental: 60 Hz (inferred from the paper plot)",
                 loc="left", fontsize=10, color="#526173", pad=14)
    ax.set(xlim=(start, end), ylim=(-7000, 7000), xlabel="Time / s", ylabel="Instantaneous voltage / V")
    ax.yaxis.set_major_locator(ticker.MultipleLocator(2000))
    ax.xaxis.set_major_locator(ticker.MultipleLocator(.02))
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
    ax.grid(True, color="#d9e0e7", linewidth=.7)
    ax.set_axisbelow(True)
    ax.axvline(event, color="#596575", ls="--", lw=1, zorder=1)
    ax.text(event+.001, -6650, event_label, fontsize=9, color="#384759",
            bbox=dict(facecolor="white", edgecolor="none", alpha=.9, pad=3))
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    return fig, ax


def main(root):
    plt.rcParams.update({"axes.unicode_minus": False, "font.size": 11, "lines.linewidth": 1.1})
    bus = load(root, "fig16", .6, .5)
    ac = load(root, "fig19", 10.2, 10.133)
    ctrl = np.genfromtxt(root / "fig19_control.csv", names=True, delimiter=",")
    if not all(np.isfinite(ctrl[n]).all() for n in ctrl.dtype.names):
        raise ValueError("PI control CSV contains nonfinite values")
    if not np.any((ctrl["time_s"] < 10.133) & (ctrl["reference_V"] == 6000)):
        raise ValueError("Missing 6000 V pre-event reference")
    if not np.all(ctrl["reference_V"][ctrl["time_s"] >= 10.133-1e-12] == 3000):
        raise ValueError("Wrong 3000 V post-event reference")

    w = select(bus, .47, .580001)
    before = select(bus, .47, .49)
    after = select(bus, .55, .58)
    bp = float(bus["bus_pos_V"][after].mean())
    bn = float(bus["bus_neg_V"][after].mean())

    fig, (ax_bus, ax_ac) = plt.subplots(
        2, 1,
        figsize=(24, 10),
        sharex=False,
        gridspec_kw={"height_ratios": [1.08, 1.0]}
    )
    fig.subplots_adjust(left=.055, right=.985, bottom=.11, top=.90, hspace=.28)

    fig.suptitle(
        "Fig. 16 scenario - DC bus voltage reduction",
        x=.09, y=.965, ha="left", fontsize=17, fontweight="bold"
    )

    ax_bus.set_title(
        "Time step: 1 us | PWM: 10 kHz | Fundamental: 60 Hz",
        loc="left", fontsize=10, color="#526173", pad=12
    )

    ax_bus.plot(
        bus["time_s"][w], bus["bus_pos_V"][w],
        color="#b51f23", lw=1.1, label="Positive DC bus"
    )
    ax_bus.plot(
        bus["time_s"][w], bus["bus_neg_V"][w],
        color="#a76316", lw=1.1, label="Negative DC bus"
    )
    ax_bus.set_ylabel("DC bus voltage / V")
    ax_bus.set_xlim(.47, .58)
    ax_bus.set_ylim(-7000, 7000)
    ax_bus.yaxis.set_major_locator(ticker.MultipleLocator(2000))
    ax_bus.xaxis.set_major_locator(ticker.MultipleLocator(.01))
    ax_bus.xaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
    ax_bus.axhline(0, color="#89939d", lw=.6)
    ax_bus.axvline(.5, color="#596575", ls="--", lw=1)
    ax_bus.legend(loc="upper right", ncol=2, fontsize=10)

    ax_bus.text(
        .501, -6500,
        "0.500 s: DC bus target +/-6 kV to +/-2 kV",
        fontsize=9, color="#384759",
        bbox=dict(facecolor="white", edgecolor="none", alpha=.9, pad=3)
    )

    ax_bus.text(
        .471, 5900,
        f"After event: +{bp:.1f} V / {bn:.1f} V",
        fontsize=9, color="#384759",
        bbox=dict(facecolor="white", edgecolor="none", alpha=.9, pad=3)
    )

    for key, color, label in (
            ("va_V", "#1765cb", "Phase A"),
            ("vb_V", "#228347", "Phase B"),
            ("vc_V", "#7a4fb3", "Phase C"),
    ):
        ax_ac.plot(
            bus["time_s"][w], bus[key][w],
            color=color, lw=.35, alpha=.68, label=label
        )

    ax_ac.set(
        xlim=(.47, .58),
        ylim=(-7000, 7000),
        xlabel="Time / s",
        ylabel="Phase voltage / V"
    )
    ax_ac.yaxis.set_major_locator(ticker.MultipleLocator(2000))
    ax_ac.xaxis.set_major_locator(ticker.MultipleLocator(.01))
    ax_ac.xaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
    ax_ac.axhline(0, color="#89939d", lw=.6)
    ax_ac.axvline(.5, color="#596575", ls="--", lw=1)
    ax_ac.legend(loc="upper right", ncol=3, fontsize=9)
    ax_ac.grid(True, color="#d9e0e7", lw=.7)
    ax_ac.set_axisbelow(True)
    ax_bus.grid(True, color="#d9e0e7", lw=.7)
    ax_bus.set_axisbelow(True)

    for ax in (ax_bus, ax_ac):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.text(
        .09, .075,
        "Open-loop bus control; raw switching ripple retained without smoothing.",
        fontsize=9, color="#526173"
    )

    fig.savefig(root / "fig16_dc_bus.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    w = select(ac, 10.09, 10.200001)
    wc = select(ctrl, 10.09, 10.200001)

    settled = select(ctrl, 10.17, 10.2)
    amplitude = float(ctrl["filtered_amplitude_V"][settled].mean())
    mod = float(ctrl["modulation"][settled].mean())

    fig, (ax_v, ax_c) = plt.subplots(
        2, 1,
        figsize=(12, 8.4),
        sharex=True,
        gridspec_kw={"height_ratios": [1.15, 1.0]}
    )
    fig.subplots_adjust(left=.09, right=.91, bottom=.13, top=.87, hspace=.25)

    fig.suptitle(
        "Fig. 19 scenario - Inverter 1 PI voltage reduction",
        x=.09, y=.965, ha="left", fontsize=17, fontweight="bold"
    )

    ax_v.set_title(
        "Time step: 1 us | PWM: 10 kHz | Fundamental: 60 Hz",
        loc="left", fontsize=10, color="#526173", pad=12
    )

    for key, color, label in (
            ("va_V", "#b51f23", "Phase A"),
            ("vb_V", "#228347", "Phase B"),
            ("vc_V", "#1765cb", "Phase C"),
    ):
        ax_v.plot(
            ac["time_s"][w], ac[key][w],
            color=color, lw=.45, alpha=.75, label=label
        )

    ax_v.set(
        ylim=(-7000, 7000),
        ylabel="Phase voltage / V"
    )
    ax_v.yaxis.set_major_locator(ticker.MultipleLocator(2000))
    ax_v.axhline(0, color="#89939d", lw=.6)
    ax_v.axvline(10.133, color="#596575", ls="--", lw=1)
    ax_v.legend(loc="upper right", ncol=3, fontsize=9)
    ax_v.grid(True, color="#d9e0e7", lw=.7)
    ax_v.set_axisbelow(True)

    ax_v.text(
        10.134, -6500,
        "10.133 s: reference amplitude 6 kV -> 3 kV",
        fontsize=9, color="#384759",
        bbox=dict(facecolor="white", edgecolor="none", alpha=.9, pad=3)
    )

    ax_c.plot(
        ctrl["time_s"][wc], ctrl["reference_V"][wc],
        color="#596575", ls="--", lw=1.2, label="Reference amplitude"
    )
    ax_c.plot(
        ctrl["time_s"][wc], ctrl["filtered_amplitude_V"][wc],
        color="#b51f23", lw=1.2, label="Filtered amplitude"
    )
    ax_c.set(
        xlim=(10.09, 10.2),
        ylim=(0, 7000),
        xlabel="Time / s",
        ylabel="Amplitude / V"
    )
    ax_c.yaxis.set_major_locator(ticker.MultipleLocator(1000))
    ax_c.axvline(10.133, color="#596575", ls="--", lw=1)
    ax_c.grid(True, color="#d9e0e7", lw=.7)
    ax_c.set_axisbelow(True)

    ax_mod = ax_c.twinx()
    ax_mod.plot(
        ctrl["time_s"][wc], ctrl["modulation"][wc],
        color="#1765cb", lw=1.0, label="Modulation index"
    )
    ax_mod.set_ylim(0, 1.05)
    ax_mod.set_ylabel("Modulation")

    lines1, labels1 = ax_c.get_legend_handles_labels()
    lines2, labels2 = ax_mod.get_legend_handles_labels()
    ax_c.legend(
        lines1 + lines2,
        labels1 + labels2,
        loc="upper right",
        fontsize=9
    )

    for ax in (ax_v, ax_c):
        ax.spines["top"].set_visible(False)
    ax_v.spines["right"].set_visible(False)

    fig.text(
        .09, .075,
        f"Settled values: filtered amplitude {amplitude:.1f} V; "
        f"modulation {mod:.4f}. PI gains are reconstructed.",
        fontsize=9, color="#526173"
    )

    fig.savefig(root / "fig19_pi_voltage.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # Validate all displayed samples fit the axes; never hide a large transient by clipping.
    for data, start, end, keys in ((bus, .47, .58, ("bus_pos_V", "bus_neg_V", "va_V", "vb_V")),
                                  (ac, 10.09, 10.2, ("va_V", "vb_V", "vc_V"))):
        mask = select(data, start, end+1e-6)
        if any(np.max(np.abs(data[key][mask])) >= 7000 for key in keys):
            raise ValueError("Displayed voltage exceeds plot limits; expand axes before delivery")
    report = {
    "dt_s": 1e-6,
    "fig16_steps": 600000,
    "fig19_steps": 10200000,
              "all_saved_values_finite": True, "no_plot_clipping": True,
              "bus_before_positive_mean_V": float(bus["bus_pos_V"][before].mean()),
              "bus_after_positive_mean_V": bp, "bus_after_negative_mean_V": bn,
              "pi_after_filtered_amplitude_mean_V": amplitude, "pi_after_modulation_mean": mod,
              "pi_settings_from_paper": False,
              "csv_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in root.glob("*.csv")}}
    (root / "validation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    for name in ("fig16_dc_bus.png", "fig19_pi_voltage.png"):
        print(root / name)


if __name__ == "__main__":
    here = Path(__file__).resolve().parent
    candidates = [here / "results", here / "build/Release/results",
                  here / "build/results", here / "build/Debug/results"]
    found = [p for p in candidates if (p / "fig16.json").exists() and (p / "fig19.json").exists()]
    if not found:
        raise SystemExit("CSV files not found. Build and run simulate_pi without arguments first.")
    # The script copied beside the executable always uses that executable's results.
    # Running the source script also supports the standard CMake output locations.
    root = found[0]
    print(f"Reading simulation results: {root}")
    main(root)
