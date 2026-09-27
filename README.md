# CityPrior — location memory from infrastructure cameras as a prior for autonomous driving

> *The vehicle understands its surroundings. The city understands the place.*

A robotaxi sees what is around it **now**. A fixed infrastructure camera watches the **same place** for hours,
days and months. This repository tests, on real data, whether that long-term *location memory* contains
information a vehicle arriving for the first time does not have, and what it would cost to send it.

This is proof-of-concept step 1 of a larger project (live infrastructure perception + location memory +
prediction → planning prior). Here we answer the most basic question first:
**does memory of a place predict what happens there later?**

## Data

[TGSIM Foggy Bottom](https://data.transportation.gov/Automobiles/Third-Generation-Simulation-Data-TGSIM-Foggy-Botto/brzy-6zfh)
(FHWA / USDOT, public domain): **12 stationary 4K infrastructure cameras** covering four intersections in
Washington, DC; 2 hours (15:00–17:00), 10 Hz, 2.48 M trajectory points after cleaning,
4,930 vehicles, 11,521 pedestrian tracks, including one SAE L3 automated test vehicle.

Protocol: memory is built from the **first 79 minutes** and evaluated on the **last 40 minutes** it has never seen,
plus blocked 3-fold cross-validation (each 40-minute block held out once).
Every score compares the memory with the **location-agnostic prior**, which is all a newcomer has
(uniform over the road area, or area-wide pooled statistics).

## Results

| Question | Memory | No memory | Evidence |
|---|---|---|---|
| Where will pedestrians step onto the road? | **73%** of held-out entries fall in the top 10% of road area; **+2.27 bits/entry** [2.22, 2.31] | 10% (uniform) | fig. 2, CV folds 2.27–2.44 bits |
| …outside crosswalks only (jaywalking, between parked cars) | **42%** in top 10% of area; +0.84 bits/entry [0.72, 0.97] | 10% | fig. 2 |
| Which lane segments produce hard braking (≤ −3.5 m/s²)? | AUC **0.85**, Brier skill **+0.08** (CV +0.085…+0.115) | pooled rate 5.7% | fig. 3a, calibrated |
| Which way will a vehicle that changes direction be heading in 3 s? | **59%** correct, P(true direction) = 0.52 | 19%, 0.33 | fig. 4a, +0.59 bits/sample |
| 3 s trajectory error of turning vehicles (> 30°) | **5.22 m** (−18% [−16, −20]) | 6.38 m (constant velocity) | fig. 4b |
| Pedestrians on/entering a car's path hidden by other vehicles | 12% of encounters hidden at first, 5% give the infrastructure ≥ 1 s head start (upper bound; buses/trucks only: 0.9% / 0.3%) | — | fig. 5, 3,042 virtual-ego encounters |
| Bandwidth for one car | memory tile **2.7 KB once** + live objects **~12 kbit/s** | raw video ≈ 180 Mbit/s (12 × 4K) | fig. 6 |

**Honest negatives, and why they are informative**

* **Vehicle–pedestrian conflicts (TTC ≤ 1.5 s)**: only 36 in 2 hours (8 in the test window). The per-crosswalk
  prior is **indistinguishable from the pooled rate** (Brier skill −0.005). Rare safety events need
  weeks to months of memory, which is the argument for long-term infrastructure memory, not against it.
* **Longitudinal error dominates vehicle prediction** (3.2 m of 3.3 m at 3 s): stop-and-go at signals.
  Memory of a place cannot know the current signal phase, so the flow prior tuned on all samples does nothing
  (β = 0). This is where **live** infrastructure data (SPaT) must complement memory.
* **A single deterministic rollout** cannot use a multimodal prior well: memory knows "30% turn right here"
  but must commit to one path. The probabilistic metric (fig. 4a) is the right use; the planner should receive
  the distribution, not one guess.
* **Pedestrian direction changes** are only weakly place-dependent in this scene (+0.10 bits/sample; the
  deterministic steering hurt turning pedestrians and was tuned away).
* **Learning curve not saturated** (5 → 79 min: 1.46 → 2.27 bits): more history is still paying off.

![Location memory](results/figures/fig1_location_memory.png)
![Held-out emergence](results/figures/fig2_emergence_heldout.png)
![Region rates](results/figures/fig3_region_rates_calibration.png)
![Prediction](results/figures/fig4_prediction.png)
![Occlusion](results/figures/fig5_occlusion_virtual_ego.png)
![Message](results/figures/fig6_robotaxi_message.png)

## What the memory holds

Sufficient statistics, not raw trajectories, and no identities:

| Layer | Per | Content | Used for |
|---|---|---|---|
| occupancy, exposure | 1 m cell × class | agent-seconds, distinct passages | normalising every rate by exposure |
| flow field | cell × class × 8 heading bins | count, mean direction, mean speed (a discrete [CLiFF map](https://arxiv.org/abs/2309.07066)) | motion prior |
| manoeuvre transitions | cell × heading now × heading in 3 s | counts | "which way next" distribution |
| speed | cell | vehicle p50 / p85 | typical-speed prior |
| pedestrian emergence | cell | first appearance on the road (after fragment stitching) | hidden-agent prior for occlusion-aware planning |
| hard braking, conflicts | cell and lane segment | events + exposure → empirical-Bayes Beta posterior with 90% interval | risk layer with uncertainty |

The whole 200 × 230 m area packs into **276 KB raw / 20 KB compressed** (6 uint8 layers).

## Method notes

* **Cleaning**: pedestrian track fragments (tracker re-identification) are stitched when a track starts ≤ 1 s
  after another ended within 2 m of its constant-velocity extrapolation (15,307 → 14,237); tracks < 1 s are dropped.
* **Hard braking**: derivative of 0.5 s-smoothed speed ≤ −3.5 m/s² for ≥ 0.5 s at ≥ 3 m/s
  (the raw Kalman accelerations spike to ±28 m/s²).
* **Conflicts**: constant-velocity TTC with a 3-disc vehicle footprint and 0.3 m pedestrian disc, 3 s horizon.
* **Rates**: per-passage event probability per lane segment, Beta prior strength fitted by beta-binomial
  marginal likelihood (shrinks small segments towards the pooled rate).
* **Virtual-ego occlusion**: every moving vehicle is treated as a robotaxi; a pedestrian ahead (≤ 40 m) on or
  heading into its straight-line path (±2.5 m within 4 s) is *hidden* if the 2-D ray from the roof-front crosses
  another vehicle's footprint. Bounds: all vehicles (upper) vs buses/trucks only (lower).
* Prediction hyper-parameters are tuned on a separate validation split (memory 0–60 min, samples 60–80 min).

## Limitations

* **2 hours, one sunny weekday afternoon**, one site. No weather, lighting or day-of-week conditioning is possible
  yet; results are a lower bound on what months of memory would give, and an upper bound on transfer to other sites.
* The dataset contains road users **only inside the road polygons** (no sidewalks) and **no buildings**, so occlusion
  and infrastructure lead time are strongly **underestimated**: real cameras would see pedestrians on the
  sidewalk before they step out.
* Detection and tracking come from the dataset (Kalman-filtered); infrastructure perception errors are not modelled.
* Occlusion is 2-D; a roof sensor sees over low cars (hence the two bounds).
* Bandwidth reference for 4K video (15 Mbit/s per stream) is an order-of-magnitude figure, not a measurement.
* Memory learned from human drivers is a **prior**, not a policy: "most drivers do X here" ≠ "X is safe".

## Reproduce

```bash
python3 -m venv .venv && .venv/bin/pip install -e ".[dev]"
.venv/bin/python -m cityprior.pipeline      # downloads ~340 MB on first run, then ~30 s on a laptop
.venv/bin/python -m pytest -q
```

Outputs: `results/metrics.json`, `results/figures/*.png`.

## Layout

```
src/cityprior/
  data.py        download, load, fragment stitching, flicker filter
  geometry.py    grid, ray–box intersection
  events.py      hard braking, pedestrian emergence, TTC conflicts
  memory.py      LocationMemory: layers, empirical-Bayes rates, flow queries, footprint
  evaluate.py    held-out scores vs the no-memory prior, bootstrap CIs
  prediction.py  constant velocity vs memory-steered rollouts
  occlusion.py   virtual-robotaxi line-of-sight analysis
  message.py     corridor message for one vehicle + byte accounting
  plots.py       figures
  pipeline.py    end-to-end run
tests/           geometry, TTC, braking detector, memory
```

## Next steps

1. **Closed-loop simulation** (occluded pedestrian scenarios): a planner receiving the emergence prior vs a
   worst-case occlusion-aware baseline, measured as a safety ↔ trip-time Pareto frontier; stale/wrong-prior and
   spoofing experiments; emergence rates calibrated from this dataset.
2. **Infrastructure memory vs fleet memory**: build the same prior only from what passing vehicles could see
   (using the occlusion model here) and compare.
3. Longer, multi-condition data (V2X-Seq, inD, SinD) for weather/time conditioning and rare-event statistics.
