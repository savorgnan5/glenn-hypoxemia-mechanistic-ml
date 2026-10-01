# Glenn Hypoxemia Mechanistic ML — Reproducibility Materials

Reproducibility materials for the manuscript **“Machine Learning to Distinguish Physiologic Causes of Hypoxemia After Bidirectional Glenn Using Bedside Markers.”**

This repository contains the frozen reviewer-revision analyses for the Glenn hypoxemia mechanistic recoverability study. **No patient-level clinical data are included.**

## Study scope

A zero-dimensional closed-loop Glenn circulation model was used to generate five encoded physiologic mechanisms:

1. Elevated pulmonary vascular resistance (PVR)
2. Veno-venous collateral shunting
3. Pulmonary gas-exchange impairment
4. Increased metabolic demand
5. Ventricular dysfunction

The classifier is trained and tested within the same simulated framework. Therefore, reported performance quantifies **internal mechanistic recoverability under the model assumptions**, not clinical diagnostic accuracy.

## Final primary analysis

- 150,000 generated states (30,000 per encoded mechanism)
- Broad SVC-saturation plausibility filter: 20–90%
- Fixed random seed: **20260812**
- Primary classifier: multinomial logistic regression
- Comparator: 500-tree random forest
- Final balanced cohort: **26,400** virtual patients
- Held-out test cohort: **6,600**
- Primary accuracy: **56.1%**
- Macro-AUROC: **0.845**

## Reviewer-requested robustness analyses

The repository includes additional analyses for:

- progressive hidden atrial-pressure elevation within ventricular dysfunction;
- a mixed elevated-PVR + pulmonary gas-exchange impairment stress test;
- a transparent physiologic decision-rule comparator evaluated on the same held-out test cohort;
- broader measurement-noise and ventricular-phenotype sensitivity analyses;
- sensitivity cohort-flow characterization;
- exploratory internal calibration.

These are simulation-based sensitivity analyses and **not clinical validation**.

## Key files

- `final_glenn_analysis.py` — frozen primary simulation and machine-learning analysis
- `reviewer2_additional_analysis.py` — Reviewer 2 robustness analyses
- `results.json` — primary and sensitivity numerical outputs
- `reviewer2_results.json` — Reviewer 2 numerical outputs
- `synthetic_glenn_balanced_final.csv` — final balanced synthetic cohort
- `synthetic_glenn_strict_unbalanced_final.csv` — strict unbalanced synthetic cohort
- `Supplementary_Table_S1_Sensitivity_Cohort_Flow.csv` — cohort flow across sensitivity scenarios
- `Supplementary_Table_S2_Reviewer2_Robustness.csv` — summary of Reviewer 2 robustness analyses
- `Supplementary_Atrial_Pressure_Sensitivity.csv` — atrial-pressure sensitivity output
- `Supplementary_Mixed_PVR_Gas_Exchange.csv` — mixed-mechanism output
- `Supplementary_Physiologic_Rule_Comparator.csv` — rule comparator output
- `Supplementary_Calibration_Bins.csv` — exploratory calibration bins
- `figure1.png`, `figure2.png` — final manuscript figures
- `requirements.txt` — Python package requirements

## Reproducibility

Run `final_glenn_analysis.py` first, followed by `reviewer2_additional_analysis.py`. The fixed seed used in the final analysis is 20260812.

## Clinical interpretation

The numerical classification results should not be interpreted as patient-level diagnostic performance. The study is intended as a mechanistic hypothesis-generation and recoverability experiment to identify which simulated physiologic states are theoretically distinguishable from bedside-observable patterns and which remain overlapping.

## Contact

Fabio Savorgnan, MD  
Baylor College of Medicine / Texas Children’s Hospital
