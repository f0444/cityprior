# CityPrior — location memory from infrastructure cameras as a prior for autonomous driving

> *The vehicle understands its surroundings. The city understands the place.*

A robotaxi sees what is around it **now**. A fixed infrastructure camera watches the **same place** for hours,
days and months. This repository tests, on real data, whether that long-term *location memory* contains
information a vehicle arriving for the first time does not have, and what it would cost to send it.

Two questions, in order:

1. **Does memory of a place predict what happens there later?** — on real infrastructure-camera data ([Part 1](#part-1--real-data-does-memory-predict)).
2. **Does it help a robotaxi drive?** — closed-loop simulation calibrated on that data ([Part 2](#part-2--closed-loop-simulation-does-memory-help-a-robotaxi-drive)).

**Headline:** on a real Washington DC block, 10 hours of location memory lets an occlusion-aware robotaxi hit
**26% fewer pedestrians at the same trip time** (95% CI 20–33%), as much as perfect knowledge of the place.
Live infrastructure cameras are a far bigger lever (≈ 99% fewer), but only while they work: a camera that fails
*silently* is the most dangerous state in the study, and a stale memory is worse than none unless it is only
allowed to *add* caution.

# Part 1 — real data: does memory predict?

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

# Part 2 — closed-loop simulation: does memory help a robotaxi drive?

**Scene** (calibrated on Part 1): a 90 m block with a dense parked row; the robotaxi drives in the adjacent lane
with 1.0 m clearance. Pedestrians step out between parked cars **where and how often they did in TGSIM**
(99 real mid-block emergences on the top street, ≈ 50/h, peak 3.3× the mean) at the **real walking-speed
distribution** (median 1.5 m/s). 90% are attentive (wait if the car is < 3 s away), 10% step out without looking;
nobody walks into the side of a passing car. Robotaxi: roof sensor with 2-D line of sight through the parked row,
0.25 s perception, 7 m/s² emergency braking, 25 mph limit. Optional infrastructure camera: 0.3 s latency.

**Planners** share one occlusion risk model and one exchange rate between time and risk (the caution level); they
differ only in the pedestrian-intensity map they believe:

| Planner | Believes |
|---|---|
| worst case | a pedestrian may step out anywhere, any time (9 km/h past parked cars) |
| **no memory** | the street's true *average* rate everywhere: knows *how many*, not *where* |
| **location memory** | a map estimated from 10 h of simulated camera observation (≈ 450 events, with sampling noise) |
| memory, may only add caution | memory, but never faster than the no-memory planner |
| perfect memory | the true map (upper bound) |
| + live camera | pedestrians approaching the curb are reported; covered sections are believed clear |

**Protocol**: rare-event design, one pedestrian per episode, weighted by the real emergence rate; 30,000
inattentive + 3,000 attentive episodes per configuration, identical across planners (common random numbers);
paired bootstrap CIs. `tests/test_sim.py` checks, among others, that the worst-case planner never hits a
pedestrian walking at the planner's design speed, i.e. planner model and simulator agree.

## Results

| Question | Result |
|---|---|
| Does memory move the safety ↔ time frontier? | **−26% collisions at equal trip time** [20, 33]; or **1.1 s faster** (−6%) at equal risk; near misses also lower. 10 h memory ≈ perfect memory (−27%) |
| How much history is needed? | saturates at **≈ 4 h** (≈ 200 observed emergences on this street); 15 min is sometimes worse than no memory |
| Where does memory pay off? | grows with how concentrated activity is: 0% (uniform) → 11% → **27% (this street)** → 47% → 56% (peak 10× mean) |
| Live cameras on the whole block | 0.005 vs 2.1 collisions / 10k traversals **and** 3 s faster; memory then adds nothing (H3) |
| Partial camera coverage | memory still cuts collisions 10–36%, depending on how uneven the *uncovered* part is |
| Stale memory (hotspot moved 21 m to the quietest spot) | symmetric memory **worse than none** (2.52 vs 2.00 / 10k); a caution floor barely helps (2.44); **"may only add caution" → 1.62** (+1.4 s) |
| Camera fails silently vs with heartbeat | **3.83 / 10k** (1.8× worse than no infrastructure) vs **1.54** falling back to memory |

![Pareto](results/figures/sim1_pareto.png)
![Coverage, history, heterogeneity](results/figures/sim2_coverage_history.png)
![Robustness](results/figures/sim3_robustness.png)

**Design rules that fall out of the experiments**

1. **Asymmetric trust.** A location prior may *add* caution freely; *removing* caution needs fresh evidence
   (change detection, live confirmation). The cost is small: −21% instead of −26% collisions at equal time when
   the memory is right; the gain is large when it is wrong.
2. **Trust must expire.** Live "this area is clear" is the most valuable and the most dangerous message: every
   message needs an age, and a missing heartbeat must revert the car to its prior.
3. **Where to invest is predictable.** The benefit of memory tracks one number per place,
   1 − (E√λ)²/Eλ (how uneven pedestrian activity is), measurable from a few hours of camera data.

**Simulation limitations**: 1-D longitudinal planner and 2-D line of sight; one pedestrian per episode; synthetic
parked row (TGSIM polygons cover the parking lane only partly); pedestrians from the near side only; the
inattentive share is a free parameter, so **absolute collision rates are not calibrated to crash statistics —
only comparisons between planners are meaningful**. The camera never raises false alarms (sidewalk walkers who
do not cross are not simulated), which flatters live infrastructure; and with the camera up the simple planner
drives at the limit and brakes hard twice as often (101 vs 50 per 1,000 traversals) because attentive
pedestrians accept 3 s gaps: comfort-aware planning with live data is future work.

# Details

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

```bash
.venv/bin/python -m cityprior.sim.experiments   # Part 2, ~8 min; writes results/sim_metrics.json, figures/sim*.png
```

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
  pipeline.py    end-to-end run (Part 1)
  sim/
    world.py        street geometry, parameters, TGSIM-calibrated emergence profile
    planner.py      occlusion risk model, speed profiles, memory estimation, camera coverage
    engine.py       vectorised closed-loop simulator, rare-event summary
    experiments.py  E1–E6 (Part 2)
    figures.py
tests/           geometry, TTC, braking detector, memory, simulator consistency
```

## Next steps

1. **Infrastructure memory vs fleet memory**: build the prior only from what passing vehicles could see (the
   occlusion model of Part 1) and compare in the simulator: the case for fixed sensors over fleet data.
2. **Change detection** for memory freshness (the rule-1 counterpart): detect that the hotspot moved from live
   camera or fleet observations and measure time-to-recover.
3. **False alarms and spoofing** from the camera (sidewalk walkers, injected / deleted objects).
4. Longer, multi-condition real data (V2X-Seq, inD, SinD) for weather / time-of-day conditioning and rare events.
