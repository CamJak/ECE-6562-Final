# HT2TF Radar/Camera Fusion Simulation

Monte Carlo simulation of heterogeneous track-to-track fusion (HT2TF) between an
active radar (range + azimuth) and a passive IR/EO camera (azimuth only), using
the cross-covariance LMMSE fusion approach of Yang, Bar-Shalom, and Willett
(2019). Includes a configurable camera "blackout" sector to test fusion
performance when the passive sensor is periodically denied.

## Project Structure

```
project/
├── Sim.py              # Configuration + entry point (run this)
├── Functions.py        # Simulation loop, tracking/fusion glue, plotting
└── Classes/
    ├── Target.py        # Ground-truth target motion model
    ├── Radar.py          # Radar sensor + local Kalman tracker
    ├── Camera.py          # Camera sensor + local Kalman tracker
    └── FusionManager.py    # HT2TF fusion center (cross-covariance LMMSE)
```

No `__init__.py` is required in `Classes/` — Python 3's implicit namespace
packages handle it, as long as you run from the project root (see below).

## Requirements

- Python 3.8+
- `numpy`
- `matplotlib`

```bash
pip install numpy matplotlib
```

## How to Run

Run `Sim.py` **from the project root directory** (the one containing
`Classes/`), since `Functions.py` imports via `from Classes.Target import *`
and similar — these paths are resolved relative to your current working
directory, not the location of the script itself.

```bash
cd project/
python Sim.py
```

By default this runs **both** experiments back-to-back:

1. `DEBUG[0]` — Control simulation (camera always visible)
2. `DEBUG[1]` — Sensor blackout simulation (180° camera dropout sector)

Each experiment runs `num_runs = 300` independent Monte Carlo trials of
`sim_steps = 250` steps each (`dt = 0.1` s, so 25 s of simulated track history
per trial). Expect roughly **25–35 seconds per experiment** (~1 minute total)
on a typical laptop — the runtime is dominated by the Python-level Monte Carlo
loop, not by the per-step linear algebra.

To run only one experiment, set `DEBUG = [1, 0]` or `DEBUG = [0, 1]` near the
top of `Sim.py`.

## What It Returns

For each experiment, three `matplotlib` figures are opened in sequence via
`plt.show()`. Because `plt.show()` blocks by default, **close each figure
window to advance to the next one** (6 figures total across both
experiments):

| # | Plot | Shows |
|---|------|-------|
| 1 | Position RMSE | Radar-only vs. fused track position error over time, averaged over all trials |
| 2 | Actual vs. Predicted RMSE | Actual RMSE overlaid with each tracker's self-reported uncertainty, $\sqrt{\operatorname{tr}(\mathbf{P}_{\text{pos}})}$ — tests whether the filter's covariance is trustworthy, not just how accurate the estimate is |
| 3 | NEES (log scale) | Normalized Estimation Error Squared, a statistical consistency check. Ideal value is 2 (dashed line), with a shaded 95% consistency band. Plotted on a log axis because rare, large transient spikes (mainly in the first ~1 s of track life, before the cross-covariance has converged) would otherwise dominate a linear plot |

None of the figures are saved to disk automatically. To save them instead of
(or in addition to) displaying them, call `plt.savefig("name.png")` right
before each `plt.show()` in `Sim.py`, or reuse the `run_monte_carlo()` /
`plot_rmse()` / `plot_rmse_with_uncertainty()` / `plot_nees()` functions
directly in your own script.

### Reading the plots

- **RMSE plots:** the fused (blue) curve should sit below the radar-only
  (orange) curve — that gap is the fusion benefit. In the blackout
  experiment, expect that gap to shrink substantially once the target's
  bearing enters the blacked-out sector.
- **NEES plots:** values should mostly track near the dashed line at 2.
  Sharp spikes early in the trajectory are expected (see the project report's
  Discussion section for why) and are not a sign the whole run failed — check
  where the bulk of the curve sits, not just the peaks.

## Optional: Single-Run GUI Mode

Setting `GUI_ON = True` in `Sim.py` switches from batch Monte Carlo
statistics to a live animated view of a single run — radar and camera
positions, the ground-truth target, local radar/camera tracks, the fused
track, and (if a blackout is configured) the shaded blackout sector. Useful
for a qualitative sanity check or a demo, but not used for the quantitative
results in the report, since it only shows one trial at a time.

## Known Limitations

- Single-target only — no clutter, false alarms, or data association.
- The fused track's NEES exhibits occasional severe transient spikes early in
  track life, traced to the linearized cross-covariance's Jacobian becoming
  ill-conditioned near degenerate target geometries (e.g. near-zero relative
  velocity). This affects a small fraction of samples (~1–2%) and is
  discussed in detail in the project report.