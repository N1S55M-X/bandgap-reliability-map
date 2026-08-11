# BandGap Reliability Map

Reliability-aware prediction of positive experimental inorganic band gaps from chemical composition.

This project does more than report a high model score. It tests **where predictions are reliable, where performance weakens, how the model fails, and whether risky predictions can be detected before they are used**.

## Project question

Can the positive experimental band gap of an inorganic material be predicted from its chemical formula—and can the system identify conditions in which its own prediction should not be trusted?

```text
Chemical formula
      ↓
Composition descriptors
      ↓
Band-gap prediction
      ↓
Reliability and applicability checks
      ↓
Predict, warn, or abstain
```

## Why this project matters

A single accuracy score describes average performance. It does not reveal whether a model fails on unfamiliar chemical families, unseen elements, extreme target values, or isolated difficult cases.

This project therefore builds a **structured failure map** using:

- Random cross-validation
- Chemical-system group validation
- Composition-cluster group validation
- Leave-one-element-out tests
- Bootstrap confidence intervals
- Conformal prediction intervals
- Applicability-domain analysis
- Model-uncertainty analysis
- Risk–coverage and abstention testing
- Target-window screening metrics
- Performance and bias analysis across band-gap ranges
- Learning curves
- Feature importance

The goal is not merely to ask:

> How accurate is the model?

It also asks:

> Under which chemical conditions is it accurate, where does it fail, how does it fail, and can that danger be recognized before acting on the prediction?

## Dataset

The analysis uses the **Matbench experimental band-gap dataset**.

| Stage | Materials |
|---|---:|
| Original dataset | 4,604 |
| Zero-gap rows removed | 2,450 |
| Positive-gap rows retained | 2,154 |

Only positive-gap records were used in the present regression task.

Consequently, the model estimates:

> The likely band-gap magnitude, conditional on the material belonging to the positive-gap subset.

It does **not** determine whether a material is a metal or whether its band gap is zero.

A universal screening pipeline would require a separate metal/non-metal classification stage or a carefully validated treatment of the zero-gap records.

## Representation

Chemical formulas cannot be passed directly to conventional regression models. Each composition is converted into **219 numerical descriptors** using `matminer`.

These descriptors summarize information such as:

- Element fractions
- Electronegativity
- Covalent radius
- Atomic weight
- Valence-electron counts
- Periodic-table position
- Elemental-property averages
- Elemental-property ranges
- Variation between elemental properties

For example:

```text
BaTiO3
   ↓
Fractions of Ba, Ti and O
   ↓
Statistics of their elemental properties
   ↓
219-number composition representation
```

The model learns statistical relationships between these composition-derived patterns and measured positive band gaps.

## Models

### Extra Trees

Extra Trees was the main model.

It builds many randomized decision trees, lets each tree estimate the band gap, and then averages their predictions.

```text
One composition
     ↓
Tree 1: 2.1 eV
Tree 2: 1.8 eV
Tree 3: 2.0 eV
Tree 4: 2.2 eV
     ↓
Average prediction
```

Randomizing features and split values creates diverse trees. Averaging them reduces the instability of any single tree.

Variation among the individual tree predictions also supplies a useful uncertainty signal.

### HistGradientBoosting

HistGradientBoosting builds trees sequentially.

Each new tree focuses on errors left by the earlier trees.

```text
First tree makes predictions
          ↓
Next tree studies the errors
          ↓
Another tree corrects remaining errors
          ↓
Final combined prediction
```

It achieved almost the same overall performance as Extra Trees and was slightly better calibrated in some measurements.

### Median baseline

The baseline predicts the same median training value for every material.

It verifies that the trained models learn useful composition-dependent information rather than merely reproducing a central value.

### Ridge regression

Ridge regression was numerically unstable in the current descriptor space and produced physically absurd extrapolations.

It is retained only as a failure case showing why correlation alone cannot validate a regression model. It must not be presented as a usable predictor.

## Main predictive results

### Extra Trees under random cross-validation

| Metric | Result | Interpretation |
|---|---:|---|
| MAE | 0.391 eV | Average absolute error |
| Median absolute error | 0.230 eV | Half of predictions were within this error |
| RMSE | 0.618 eV | Reveals the influence of larger errors |
| 90th-percentile error | 0.950 eV | About 90% of errors were below this value |
| Maximum error | 5.515 eV | At least one prediction failed severely |
| R² | 0.822 | Strong overall predictive structure |
| Spearman correlation | 0.895 | Strong ranking of lower- and higher-gap materials |
| CCC | 0.899 | Strong agreement with measurements |
| Mean bias | −0.002 eV | Almost no global directional bias |
| Calibration slope | 1.038 | Predictions were well scaled overall |

The median baseline produced an MAE of `1.066 eV`.

Extra Trees reduced the average absolute error by approximately **63%** relative to this trivial predictor.

No single metric gives the complete result:

```text
MAE                  → typical numerical accuracy
RMSE and maximum     → dangerous large errors
R²                   → captured variation
Spearman             → ranking quality
CCC                  → agreement with measurements
Bias and calibration → systematic distortion
```

## Why multiple metrics are necessary

Ridge regression produced a respectable Spearman correlation but catastrophic numerical errors.

This means it often ranked materials in approximately the correct order while predicting completely unrealistic band-gap values.

Therefore:

> Correct ranking does not guarantee correct numerical prediction.

A combination of MAE, RMSE, maximum error, R², ranking, calibration and subgroup analysis is needed to expose different kinds of failure.

## Generalization beyond easy random splits

| Validation condition | MAE | R² | What it tests |
|---|---:|---:|---|
| Random split | 0.391 eV | 0.822 | Ordinary unseen samples from similar data |
| Unseen chemical systems | 0.414 eV | 0.823 | Withheld exact combinations of elements |
| Unseen composition clusters | 0.555 eV | 0.728 | Withheld, chemically distant regions |

Performance changed only modestly when exact chemical systems were withheld.

However, cluster-separated error increased by approximately **41.9%** relative to random validation.

This is the central generalization result:

> The model transfers reasonably within related chemistry but becomes substantially less reliable in distant composition space.

### Random split

Materials are randomly distributed between training and testing folds.

This tests whether the model predicts ordinary unseen samples drawn from approximately the same dataset.

### Chemical-system split

Materials containing the same exact set of elements are kept together.

For example, the complete `Ba-Ti-O` chemical system is placed in either training or testing, but not divided between both.

This tests whether the model can handle new combinations of elements.

However, the individual elements may still appear in other training compositions.

### Composition-cluster split

Materials are grouped according to similarity in their descriptor representation. Entire groups are withheld from training.

This produces a harder test:

> Can the model predict materials from an unfamiliar region of composition space?

The large performance decrease shows that random cross-validation alone would have hidden an important reliability boundary.

## Bootstrap confidence intervals

| Validation condition | MAE | 95% confidence interval |
|---|---:|---:|
| Random | 0.391 eV | 0.372–0.411 eV |
| Chemical system | 0.414 eV | 0.395–0.432 eV |
| Composition cluster | 0.555 eV | 0.534–0.578 eV |

Bootstrap analysis repeatedly resamples the predictions and recalculates the error.

The cluster-split interval is clearly separated from the random-split interval.

Therefore:

> The deterioration in distant composition space is unlikely to be explained by resampling noise alone.

## Completely unseen elements

Leave-one-element-out testing removes every composition containing a chosen element from training and then tests only on materials containing that element.

For oxygen:

```text
Remove all oxygen-containing materials from training
                       ↓
Train without seeing oxygen chemistry
                       ↓
Test only on oxygen-containing materials
```

Results:

| Unseen element | MAE | R² |
|---|---:|---:|
| Te | 0.374 eV | 0.791 |
| In | 0.390 eV | 0.743 |
| Sb | 0.402 eV | 0.702 |
| Se | 0.467 eV | 0.699 |
| Ga | 0.490 eV | 0.572 |
| P | 0.497 eV | 0.703 |
| S | 0.502 eV | 0.433 |
| O | 1.083 eV | 0.110 |

Transfer depended strongly on the withheld element.

Oxygen was the clearest failure case: a model trained without oxides could not reliably recover oxide behaviour from other chemistry.

This gives an important applicability rule:

> Predictions should not automatically be trusted when a composition contains an element completely absent from the training data.

These elemental subsets differ in size and target distribution, so their scores should not be interpreted as a simple ranking of elemental difficulty.

## Prediction intervals

A point prediction gives one number:

```text
Predicted band gap: 2.0 eV
```

Conformal prediction adds a plausible range:

```text
Predicted band gap: 2.0 eV
90% prediction interval: 1.1–3.0 eV
```

Results:

| Requested coverage | Observed coverage | Mean interval width |
|---:|---:|---:|
| 80% | 86.8% | 1.55 eV |
| 90% | 91.2% | 2.06 eV |
| 95% | 96.1% | 2.67 eV |

The intervals achieved approximately their promised coverage, although they were broad.

```text
Coverage → Is the interval statistically honest?
Width    → Is the interval practically precise?
```

A very wide interval may achieve excellent coverage without being useful for precise screening.

Therefore, both coverage and width must be reported.

## Applicability domain

Chemical distance estimates how far a new composition lies from the training domain.

```text
Close to training data → familiar chemistry
Far from training data → unfamiliar chemistry
```

| Region | MAE |
|---|---:|
| Nearest 75% | 0.355 eV |
| Most distant 25% | 0.759 eV |
| Explicit OOD-warning group | 0.854 eV |
| Non-OOD group | 0.426 eV |

The most distant compositions had more than twice the error of the closer group.

This means chemical distance can be used to generate a warning:

```text
Warning: This composition lies outside the reliable training domain.
```

Chemical distance is a warning signal, not proof that a prediction is wrong.

## Failure detection

Two warning scores were evaluated for detecting large errors:

| Warning signal | Correlation with absolute error | High-error AUROC |
|---|---:|---:|
| Chemical distance | 0.420 | 0.696 |
| Model uncertainty | 0.584 | 0.797 |

Model uncertainty was the stronger failure detector.

Chemical distance mainly asks:

```text
How unfamiliar is this composition?
```

Model uncertainty can additionally reflect:

- Disagreement among trees
- Sparse training regions
- Ambiguous descriptor patterns
- Difficult target regions
- Instability in the learned relationship

An AUROC of `0.797` indicates useful but imperfect separation between high-error and safer predictions.

## Selective prediction: allowing the model to abstain

The system can reject the most uncertain cases instead of forcing a prediction for every material.

| Predictions retained | MAE | R² |
|---:|---:|---:|
| 100% | 0.456 eV | 0.750 |
| 90% | 0.348 eV | 0.818 |
| 80% | 0.306 eV | 0.855 |
| 70% | 0.284 eV | 0.867 |
| 50% | 0.219 eV | 0.915 |
| 30% | 0.173 eV | 0.938 |

Keeping only the safest half reduced MAE by approximately **52%**, from `0.456` to `0.219 eV`.

This demonstrates that uncertainty is operationally useful:

```text
Low uncertainty  → provide a screening prediction
High uncertainty → abstain and request verification
```

The trade-off is:

```text
More predictions → lower reliability
Fewer predictions → higher reliability
```

The values in this risk–coverage analysis come from its dedicated evaluation split and therefore should not be directly substituted for the random cross-validation MAE.

## Screening materials in the 1–2 eV window

The regression predictions were also evaluated as a candidate-screening system.

The objective was:

```text
Find materials with positive band gaps between 1 and 2 eV
```

Results:

| Metric | Result |
|---|---:|
| Precision | 59.2% |
| Recall | 67.7% |
| Specificity | 81.1% |
| F1 score | 0.632 |
| MCC | 0.470 |
| AUROC | 0.823 |
| Average precision | 0.622 |
| Enrichment factor | 2.06 |

The selected set contained about twice the proportion of target-window materials expected from random selection.

This supports **candidate prioritization**, not final material selection.

Experimental or higher-fidelity computational confirmation remains necessary.

## Known systematic failure pattern

| True band-gap region | MAE | Mean bias | Behaviour |
|---|---:|---:|---|
| 0–1 eV | 0.474 eV | +0.456 eV | Overprediction |
| 1–2 eV | 0.324 eV | +0.141 eV | Mild overprediction |
| 2–3 eV | 0.346 eV | +0.011 eV | Strongest calibration |
| 3–4 eV | 0.457 eV | −0.229 eV | Underprediction |
| Above 4 eV | 1.213 eV | −1.089 eV | Severe underprediction |

The model pulls extreme values toward the more common middle of the dataset.

```text
Very low gaps  → pulled upward
Very high gaps → pulled downward
```

Positive bias at low gaps and negative bias at high gaps cancel.

This explains why the global mean bias is almost zero even though important local biases exist.

Therefore:

> Global bias near zero does not mean the model is unbiased in every band-gap region.

## Learning curve

| Training examples | MAE |
|---:|---:|
| 258 | 0.606 eV |
| 516 | 0.525 eV |
| 775 | 0.513 eV |
| 1,033 | 0.484 eV |
| 1,292 | 0.458 eV |

Error generally decreased as more training data were added.

The model has not fully saturated and may improve with additional high-quality experimental data, particularly for:

- Rare chemical families
- Underrepresented elements
- Distant compositions
- Very high-band-gap materials

## What the model learned

Influential descriptors included:

- Electronegativity statistics
- Covalent-radius statistics
- Valence-electron counts
- Periodic-table row
- Oxygen and fluorine fractions
- Elemental melting-point statistics
- Average d-valence electrons

These dependencies are chemically plausible because bonding, orbital occupation, atomic size and electronegativity influence electronic structure.

However, feature importance means:

```text
The model used this descriptor for prediction.
```

It does not prove:

```text
This descriptor independently causes the band gap.
```

Correlated descriptors can carry overlapping information.

## Reliability logic for a new prediction

A prediction should be interpreted using several checks together:

```text
Predicted positive band gap
          +
Are all elements represented in training?
          +
Is the composition near the training domain?
          +
Do the trees agree?
          +
Is the prediction interval acceptably narrow?
          +
Is the target in a known reliable band-gap region?
          ↓
Use, warn, or abstain
```

Example of a relatively safe screening result:

```text
Prediction: 2.1 eV
Elements represented: yes
Chemical distance: low
Tree disagreement: low
Prediction interval: acceptably narrow
Target region: comparatively reliable
Decision: retain for screening
```

Example of a risky result:

```text
Prediction: 5.2 eV
Chemical distance: high
Tree disagreement: high
Prediction interval: wide
Target region: known underprediction risk
Decision: abstain and verify independently
```

## Complete workflow

```text
Real experimental compositions and positive band gaps
                         ↓
Convert formulas into 219 composition descriptors
                         ↓
Train Extra Trees and comparison models
                         ↓
Evaluate ordinary predictive accuracy
                         ↓
Test unseen chemical systems
                         ↓
Test distant composition clusters
                         ↓
Remove individual elements and test extrapolation
                         ↓
Estimate uncertainty and prediction intervals
                         ↓
Measure distance from the training domain
                         ↓
Map performance across band-gap regions
                         ↓
Test whether uncertainty detects large errors
                         ↓
Allow the model to reject unsafe predictions
                         ↓
Use reliable predictions to prioritize experiments
```

## Installation

```bash
pip install numpy pandas scipy scikit-learn matplotlib seaborn matminer matbench pymatgen
```

Use the exact dependency versions recorded with the analysis when reproducing the final published results.

## Reproducing the analysis

1. Clone this repository.
2. Install the required packages.
3. Run the project notebook or analysis script from beginning to end.
4. Confirm that the Matbench data load successfully.
5. Inspect the generated tables and figures for every validation stage.

Before release, add:

- The exact notebook or script filename
- A pinned `requirements.txt` or environment file
- The random seeds used by the analysis
- Generated figures
- An open-source license

## Scope and limitations

- The model predicts only the magnitude of positive band gaps in the retained dataset.
- It does not classify metals versus non-metals.
- Composition descriptors omit crystal structure.
- The model does not represent defects or polymorphs.
- Temperature, pressure and measurement conditions are not included.
- Random cross-validation is not sufficient evidence of chemical extrapolation.
- Performance decreases for distant composition clusters.
- Transfer to completely unseen elements is element-dependent.
- Transfer failed strongly when oxygen was completely absent from training.
- Prediction intervals are reasonably calibrated but can be too broad for precise screening.
- High-gap materials are systematically underestimated.
- Model outputs are priorities for further investigation, not experimental facts.
- Feature importance is not proof of causality.

## Correct scientific conclusion

> Extra Trees combined with composition-derived descriptors provides strong prediction and ranking of positive experimental inorganic band gaps within represented chemical space. Random cross-validation achieved an MAE of 0.391 eV and R² of 0.822. Performance remained similar for withheld exact chemical systems but deteriorated for distant composition clusters and completely unseen elements, particularly oxygen. Conformal intervals achieved near-nominal coverage, while model-derived uncertainty identified many high-error predictions and enabled effective abstention. The approach is useful for candidate prioritization within its applicability domain, but it cannot identify metals, guarantee experimental behaviour, or reliably extrapolate to high-gap and substantially unfamiliar materials.

## Simplest interpretation

```text
Ordinary predictive accuracy: Strong
Candidate ranking: Strong
Related-chemistry transfer: Strong
Distant-chemistry transfer: Moderate
Unseen-element transfer: Element-dependent
Unseen oxygen transfer: Weak
Uncertainty calibration: Good but broad
Failure detection: Useful
Safe abstention: Highly effective
1–2 eV screening: Useful for prioritization
Very high-gap prediction: Weak
Universal band-gap prediction: Not supported
```

## Central principle

> A scientifically trustworthy model should describe the boundary of its knowledge, not merely demonstrate success inside that boundary.

This repository treats uncertainty as a structured failure profile rather than compressing it into a single confidence number.

It does not merely report:

```text
How uncertain is the model?
```

It identifies:

```text
Which chemical regions are uncertain?
Which elements create extrapolation risk?
Which band-gap ranges are biased?
Which individual predictions are unsafe?
Can the model recognize risk before the prediction is used?
```

## Suggested future work

- Build a first-stage classifier for zero-gap versus positive-gap materials.
- Add structure-aware descriptors or crystal graph models.
- Evaluate external datasets collected from independent sources.
- Calibrate uncertainty separately across chemical families.
- Calibrate uncertainty separately across target ranges.
- Increase representation of rare elements.
- Increase representation of very high-gap materials.
- Compare uncertainty methods under identical held-out splits.
- Package inference as a tool that returns a prediction, interval, domain warning and abstention decision together.

## Responsible use

Use predictions to rank candidates for experimental or higher-fidelity computational follow-up.

Do not treat a predicted band gap as a confirmed material property.
