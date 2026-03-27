# Methodology Review Response and Figure Inventory

Date: 2026-03-27

This document answers all outstanding methodology-review points against the current MetaMP implementation. It is based on the live code and the current production publication snapshot, not on aspirational wording.

## Executive Summary

The remaining review comments are mostly valid. The largest real vulnerability is still the machine-learning section, but the correct fix is to describe the implemented workflow more precisely rather than to add stronger-looking parameters that are not actually used.

The most important implementation-backed conclusions are:

- The supervised and semi-supervised ML workflows need to be described separately.
- The code uses the same six classifier families in both modes: Logistic Regression, Decision Tree, Random Forest, KNeighbors Classifier, Gradient Boosting Classifier, and SVM.
- Production model bundles exist only for `no_dr`, `pca`, and `umap`. `t-SNE` is exploratory-only.
- Semi-supervised learning uses sklearn `SelfTrainingClassifier` defaults from scikit-learn `1.3.2`; it does not currently define a custom threshold, custom stopping rule, or fold-wise pseudo-label regeneration.
- Supervised models use 5-fold stratified cross-validation for the reported CV metrics. Semi-supervised models do not currently run the same fold-wise CV procedure; they are evaluated through a labeled train/test split plus held-out expert benchmarking.
- The review suggestion to claim `threshold=0.9`, `max_iter=10`, `stop if fewer than 5 new samples are added`, `daily refresh`, or `weekly topology jobs` would be inaccurate for the current codebase.

## Primary Source Files Audited

- [paper.tex](/Users/awotoroebenezer/Desktop/MetaMP-Server/paper.tex)
- [src/Jobs/MLJobs.py](/Users/awotoroebenezer/Desktop/MetaMP-Server/src/Jobs/MLJobs.py)
- [src/Jobs/Utils.py](/Users/awotoroebenezer/Desktop/MetaMP-Server/src/Jobs/Utils.py)
- [src/Dashboard/scientific_assessment.py](/Users/awotoroebenezer/Desktop/MetaMP-Server/src/Dashboard/scientific_assessment.py)
- [src/core/celery_factory.py](/Users/awotoroebenezer/Desktop/MetaMP-Server/src/core/celery_factory.py)
- [src/Jobs/tasks/task1.py](/Users/awotoroebenezer/Desktop/MetaMP-Server/src/Jobs/tasks/task1.py)
- [requirements.txt](/Users/awotoroebenezer/Desktop/MetaMP-Server/requirements.txt)
- [datasets/expert_annotation_predicted.csv](/Users/awotoroebenezer/Desktop/MetaMP-Server/datasets/expert_annotation_predicted.csv)
- [training_variables.json](/Users/awotoroebenezer/Desktop/MetaMP-Server/.metamp-publication/runs/20260327T163411Z/copied/production_ml/specs/training_variables.json)
- [model_bundle_registry.csv](/Users/awotoroebenezer/Desktop/MetaMP-Server/.metamp-publication/runs/20260327T163411Z/copied/production_ml/tables/model_bundle_registry.csv)
- [publication figure manifest](/Users/awotoroebenezer/Desktop/MetaMP-Server/.metamp-publication/runs/20260327T163411Z/copied/publication_figures/model_registry/figure_manifest.json)

## Current Production Snapshot Facts

- Training rows: `3966`
- Training unique PDB codes: `3963`
- Reserved benchmark exclusions: `126`
- Expert benchmark rows: `121`
- Discrepancy exclusion codes: `111`
- Legacy benchmark exclusion codes: `53`
- CV splitter recorded in the snapshot: `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`
- Selected upload artifact in the current registry: `semi_supervised_no_dr_decision_tree`

These values come from [training_variables.json](/Users/awotoroebenezer/Desktop/MetaMP-Server/.metamp-publication/runs/20260327T163411Z/copied/production_ml/specs/training_variables.json) and [model_bundle_registry.csv](/Users/awotoroebenezer/Desktop/MetaMP-Server/.metamp-publication/runs/20260327T163411Z/copied/production_ml/tables/model_bundle_registry.csv).

## Responses to Remaining Major Concerns

### 1. The ML section is improved, but still the most vulnerable part

Verdict: Valid, and this is still the most important remaining methodological risk.

What is valid in the review:

- The current wording is still too abstract for strict reproducibility if it simply says `threshold-based self-training` and `bounded iteration schedule`.
- A reviewer is justified in asking:
  - what threshold was used
  - what maximum number of iterations was allowed
  - what stopping criterion was used
  - whether all models used identical self-training settings
  - whether pseudo-labels were regenerated independently within each CV fold

What the implementation actually does:

- Semi-supervised models are wrapped with sklearn's `SelfTrainingClassifier(clf)` in [src/Jobs/Utils.py](/Users/awotoroebenezer/Desktop/MetaMP-Server/src/Jobs/Utils.py).
- The project pins `scikit-learn==1.3.2` in [requirements.txt](/Users/awotoroebenezer/Desktop/MetaMP-Server/requirements.txt).
- The installed sklearn source in [._self_training.py](/Users/awotoroebenezer/Desktop/MetaMP-Server/.mpvis/lib/python3.10/site-packages/sklearn/semi_supervised/_self_training.py) shows the defaults:
  - `criterion='threshold'`
  - `threshold=0.75`
  - `k_best=10`
  - `max_iter=10`
  - termination when `max_iter` is reached, no pseudo-labels are added, or all unlabeled records become labeled
- All six semi-supervised base classifiers use the same wrapper style, so identical wrapper settings across classifier families is true.
- Pseudo-labelled samples are not used in held-out expert-benchmark evaluation or discrepancy exports.

What remains important to correct conceptually:

- The current implementation does not support the claim that pseudo-labels are regenerated independently within each cross-validation fold.
- In fact, the current code separates supervised CV from semi-supervised hold-out evaluation:
  - supervised mode uses 5-fold `StratifiedKFold` in [src/Jobs/MLJobs.py](/Users/awotoroebenezer/Desktop/MetaMP-Server/src/Jobs/MLJobs.py)
  - semi-supervised mode uses a labeled train/test split inside [ClassifierComparisonSemiSupervised](/Users/awotoroebenezer/Desktop/MetaMP-Server/src/Jobs/Utils.py), plus the held-out expert benchmark
- Therefore, a sentence implying fold-wise semi-supervised CV would currently be inaccurate.

What should be said cleanly in a reviewer response:

- The concern is valid.
- The implementation is reproducible, but the manuscript should distinguish:
  - supervised CV evaluation
  - semi-supervised train/test self-training evaluation
  - external held-out expert benchmarking used for both
- The manuscript should state that semi-supervised self-training currently relies on sklearn `1.3.2` `SelfTrainingClassifier` defaults unless and until explicit custom parameters are introduced in code.

### 1A. Correct supervised workflow description

This should be stated separately from semi-supervised learning.

Current supervised implementation:

- Input matrix: labeled training matrix only
- Features:
  - numeric: `thickness`, `subunit_segments`, `tilt`, `gibbs`
  - categorical: `topology_subunit`, `membrane_topology_in`, `membrane_topology_out`
- Missing numeric values are median-imputed inside the classifier wrapper
- The supervised wrapper in [ClassifierComparison](/Users/awotoroebenezer/Desktop/MetaMP-Server/src/Jobs/Utils.py) does not apply `StandardScaler`
- Internal split:
  - `train_test_split(..., test_size=0.2, stratify=y, random_state=42)`
- CV metrics:
  - 5-fold `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` in [src/Jobs/MLJobs.py](/Users/awotoroebenezer/Desktop/MetaMP-Server/src/Jobs/MLJobs.py)
- Production reductions:
  - `no_dr`
  - `pca`
  - `umap`
- External benchmark:
  - all supervised bundles are evaluated on the 121-row expert holdout after training

### 1B. Correct semi-supervised workflow description

This should not be described as identical to the supervised CV path.

Current semi-supervised implementation:

- Labeled matrix and unlabeled matrix are built separately in [src/Jobs/MLJobs.py](/Users/awotoroebenezer/Desktop/MetaMP-Server/src/Jobs/MLJobs.py)
- The semi-supervised wrapper in [ClassifierComparisonSemiSupervised](/Users/awotoroebenezer/Desktop/MetaMP-Server/src/Jobs/Utils.py):
  - stratifies the labeled data into train/test using `test_size=0.2` and `random_state=42`
  - median-imputes the labeled and unlabeled matrices
  - applies `StandardScaler`
  - concatenates labeled and unlabeled records
  - marks unlabeled rows with `-1`
  - shuffles the combined matrix with `random_state=42`
  - wraps the base estimator with sklearn `SelfTrainingClassifier`
- The current code does not define custom self-training parameters. With sklearn `1.3.2`, this means:
  - `criterion='threshold'`
  - `threshold=0.75`
  - `max_iter=10`
  - early termination when no additional pseudo-labels are added or all unlabeled samples are labeled
- Evaluation:
  - internal labeled test split metrics
  - external held-out expert benchmark across 121 expert-reviewed rows
- Important nuance:
  - the semi-supervised metrics later stored in registry columns named `cv_mean_*` are not generated through the same fold-wise cross-validation procedure as the supervised metrics
  - this naming is therefore convenient operationally but methodologically imprecise

### 1C. Which reviewer suggestions are correct and which are not

Correct reviewer pressure:

- ask for exact self-training settings
- ask for separation of semi-supervised and supervised evaluation logic
- ask for exact leakage-prevention wording

Incorrect if copied literally without code changes:

- `threshold = 0.9`
- `maximum 10 iterations` as a project-specific choice rather than a sklearn default
- `stop if fewer than 5 new samples are added`
- `pseudo-labels regenerated independently within each fold`

Those would be new design claims, not honest descriptions of the current implementation.

### 2. Presentation layer remains slightly too long and UI-descriptive

Verdict: Valid.

Why the concern is fair:

- The current nine-view enumeration in [paper.tex](/Users/awotoroebenezer/Desktop/MetaMP-Server/paper.tex) still reads partly like interface documentation.
- The methodologically distinctive views are:
  - `Data Discrepancy`
  - `Outlier Detection`
  - `Single-Entry Structural`
- The other views can be summarized more compactly without loss of scientific meaning.

Best response:

- Acknowledge that this is an editorial rather than conceptual issue.
- State that the section can be condensed, with detailed view descriptions moved to supplementary material if required by venue.

### 3. Some implementation detail is still unevenly distributed

Verdict: Valid.

Why this is fair:

- ETL verification is concrete
- benchmarking caveats are concrete
- user-study statistics are concrete
- ML self-training remains under-specified
- scientific flags remain heuristic
- updating is functionally described, but only partly operationalized in the text

Important correction to the suggested fix:

- The proposed wording that background jobs run `daily` for source updates and `weekly` for topology workflows is not supported by the current code.
- The built-in Celery beat schedule in [src/core/celery_factory.py](/Users/awotoroebenezer/Desktop/MetaMP-Server/src/core/celery_factory.py) schedules `monthly-production-maintenance` every 30 days.
- Task definitions exist for refresh and sync jobs in [src/Jobs/tasks/task1.py](/Users/awotoroebenezer/Desktop/MetaMP-Server/src/Jobs/tasks/task1.py), but the repository does not show an explicit daily/weekly scheduler hierarchy.

So the concern is valid, but the proposed wording should be corrected to something like:

- updates are handled through background Celery tasks
- maintenance scheduling is coordinated through Celery beat
- the currently configured built-in periodic schedule is monthly production maintenance

### 4. Scientific assessment flags are useful, but still somewhat subjective in method description

Verdict: Valid.

Why this is fair:

- The implementation in [src/Dashboard/scientific_assessment.py](/Users/awotoroebenezer/Desktop/MetaMP-Server/src/Dashboard/scientific_assessment.py) is explicitly rule-based and heuristic:
  - entry-specific overrides
  - keyword-triggered rules
  - chain-count structure context
  - replacement metadata
- That makes the system transparent and auditable, but it remains interpretive rather than statistically validated.

What is safe to say:

- the rules are transparent
- the rules are public in the repository
- the system is intended as cautionary support, not definitive truth

What is not safe to say without extra documentation:

- that the rules were fully specified a priori
- that they were lightly refined on a calibration subset
- that they were formally validated on an independent subset

Current honest position:

- false positives are handled by transparency rather than by formal optimization
- records can still be included with caution
- flags and exclusion reasons are exposed, not hidden

## Responses to Medium-Level Concerns

### 1. “Largely non-redundant” may be slightly overstated

Verdict: Valid but minor.

Why:

- The sources are complementary, but there is clearly partial overlap in membrane-annotation scope.
- A softer phrase such as `complementary with partial redundancy` would be harder to challenge.

### 2. The architecture stack may still be too detailed for some journals

Verdict: Valid but venue-dependent.

Interpretation:

- For a systems/platform paper, the current stack description is acceptable.
- For a biology-oriented journal, some of the implementation stack may be better placed in supplementary material.

This is an editorial choice, not a scientific-method failure.

### 3. The discrepancy definition is broad

Verdict: Valid.

Why:

- broad-group disagreement
- transmembrane-count disagreement
- transmembrane-boundary-support disagreement

These are related but distinct discrepancy types. Separating them conceptually would improve precision.

### 4. User-study tasks are still simple

Verdict: Valid.

What the current manuscript can safely claim:

- the user study supports workflow usability
- the user study supports learnability of selected analytical tasks

What it should not overclaim:

- expert scientific effectiveness
- deep biological validation

This is now mostly a claim-discipline issue in Results and Discussion rather than a Methods failure.

## Publication-Readiness Assessment by Section

### Data Sources and System Design

Assessment: Strong.

Reason:

- source scope is justified
- boundaries are explicit
- platform framing is present

### Data Layer

Assessment: Very strong.

Reason:

- ETL logic is concrete
- provenance preservation is a clear strength
- incremental update behavior is well aligned with the platform’s purpose

### Application Layer

Assessment: Strong.

Reason:

- functionally clear
- discrepancy handling is well described
- operational scheduling detail is still lighter than other sections

### Presentation Layer

Assessment: Adequate to strong.

Reason:

- scientifically relevant for a platform paper
- currently somewhat too detailed in interface-walkthrough style

### Expert Reference Set and Concordance

Assessment: Very strong.

Reason:

- benchmark limitation is now responsibly framed
- benchmark-aware concordance is biologically thoughtful

### Transmembrane-Segment Benchmarking

Assessment: Strong.

Reason:

- reference standard and granularity caveat are both well handled
- predictor integration is transparent

### Machine-Learning Module

Assessment: Moderate to strong.

Reason:

- major conceptual problems have been reduced
- this remains the most exposed section because the supervised and semi-supervised evaluation flows are not identical and should be described separately

### Scientific Assessment Flags

Assessment: Strong conceptually, moderate methodologically.

Reason:

- useful and transparent
- still heuristic rather than formally validated

### Task-Oriented User Evaluation

Assessment: Strong for a platform paper.

Reason:

- the rationale for inclusion is now defensible
- the scope is appropriately limited

## Correct Figure Inventory for the Methods and Related ML Sections

This section distinguishes figures that are actually present in the current live snapshot from legacy manuscript figure references that are not currently available in this workspace.

### A. Current live publication figures that are present and defensible

#### Data-source / overview figure

- Database-year proportional representation:
  - [database_year_proportional_representation.pdf](/Users/awotoroebenezer/Desktop/MetaMP-Server/.metamp-publication/runs/20260327T163411Z/copied/publication_figures/database_year_proportional_representation.pdf)
  - [database_year_proportional_representation.png](/Users/awotoroebenezer/Desktop/MetaMP-Server/.metamp-publication/runs/20260327T163411Z/copied/publication_figures/database_year_proportional_representation.png)

#### Exploratory dimensionality-reduction figures

- PCA:
  - [pca.pdf](/Users/awotoroebenezer/Desktop/MetaMP-Server/.metamp-publication/runs/20260327T163411Z/copied/publication_figures/exploratory_dr/pca.pdf)
  - [pca.png](/Users/awotoroebenezer/Desktop/MetaMP-Server/.metamp-publication/runs/20260327T163411Z/copied/publication_figures/exploratory_dr/pca.png)
- t-SNE:
  - [tsne.pdf](/Users/awotoroebenezer/Desktop/MetaMP-Server/.metamp-publication/runs/20260327T163411Z/copied/publication_figures/exploratory_dr/tsne.pdf)
  - [tsne.png](/Users/awotoroebenezer/Desktop/MetaMP-Server/.metamp-publication/runs/20260327T163411Z/copied/publication_figures/exploratory_dr/tsne.png)
- UMAP:
  - [umap.pdf](/Users/awotoroebenezer/Desktop/MetaMP-Server/.metamp-publication/runs/20260327T163411Z/copied/publication_figures/exploratory_dr/umap.pdf)
  - [umap.png](/Users/awotoroebenezer/Desktop/MetaMP-Server/.metamp-publication/runs/20260327T163411Z/copied/publication_figures/exploratory_dr/umap.png)

Important note:

- These are exploratory DR figures for the training records.
- They are not the same thing as production model-bundle reductions.

#### Production ML comparison figures

- Semi-supervised vs supervised performance:
  - [semi_vs_supervised_performance.pdf](/Users/awotoroebenezer/Desktop/MetaMP-Server/.metamp-publication/runs/20260327T163411Z/copied/production_ml/figures/semi_vs_supervised_performance.pdf)
  - [semi_vs_supervised_performance.png](/Users/awotoroebenezer/Desktop/MetaMP-Server/.metamp-publication/runs/20260327T163411Z/copied/production_ml/figures/semi_vs_supervised_performance.png)
- Expert benchmark leaderboard:
  - [expert_benchmark_leaderboard.pdf](/Users/awotoroebenezer/Desktop/MetaMP-Server/.metamp-publication/runs/20260327T163411Z/copied/production_ml/figures/expert_benchmark_leaderboard.pdf)
  - [expert_benchmark_leaderboard.png](/Users/awotoroebenezer/Desktop/MetaMP-Server/.metamp-publication/runs/20260327T163411Z/copied/production_ml/figures/expert_benchmark_leaderboard.png)

#### SHAP figures

- Global SHAP bar plot:
  - [semi_supervised_no_dr_decision_tree_shap_bar.pdf](/Users/awotoroebenezer/Desktop/MetaMP-Server/.metamp-publication/runs/20260327T163411Z/copied/production_ml/figures/semi_supervised_no_dr_decision_tree_shap_bar.pdf)
  - [semi_supervised_no_dr_decision_tree_shap_bar.png](/Users/awotoroebenezer/Desktop/MetaMP-Server/.metamp-publication/runs/20260327T163411Z/copied/production_ml/figures/semi_supervised_no_dr_decision_tree_shap_bar.png)
- Class-specific beeswarms:
  - [monotopic beeswarm pdf](/Users/awotoroebenezer/Desktop/MetaMP-Server/.metamp-publication/runs/20260327T163411Z/copied/production_ml/figures/semi_supervised_no_dr_decision_tree_shap_beeswarm_monotopic_membrane_proteins.pdf)
  - [alpha-helical beeswarm pdf](/Users/awotoroebenezer/Desktop/MetaMP-Server/.metamp-publication/runs/20260327T163411Z/copied/production_ml/figures/semi_supervised_no_dr_decision_tree_shap_beeswarm_transmembrane_proteins_alpha_helical.pdf)
  - [beta-barrel beeswarm pdf](/Users/awotoroebenezer/Desktop/MetaMP-Server/.metamp-publication/runs/20260327T163411Z/copied/production_ml/figures/semi_supervised_no_dr_decision_tree_shap_beeswarm_transmembrane_proteins_beta_barrel.pdf)

#### Model-registry publication figures

These are present and documented in [figure_manifest.json](/Users/awotoroebenezer/Desktop/MetaMP-Server/.metamp-publication/runs/20260327T163411Z/copied/publication_figures/model_registry/figure_manifest.json):

- [fig1_cv_vs_expert_scatter.pdf](/Users/awotoroebenezer/Desktop/MetaMP-Server/.metamp-publication/runs/20260327T163411Z/copied/publication_figures/model_registry/fig1_cv_vs_expert_scatter.pdf)
- [fig2_top_n_ranked_bar.pdf](/Users/awotoroebenezer/Desktop/MetaMP-Server/.metamp-publication/runs/20260327T163411Z/copied/publication_figures/model_registry/fig2_top_n_ranked_bar.pdf)
- [fig3_grouped_bar_classifier_mode.pdf](/Users/awotoroebenezer/Desktop/MetaMP-Server/.metamp-publication/runs/20260327T163411Z/copied/publication_figures/model_registry/fig3_grouped_bar_classifier_mode.pdf)
- [fig4_heatmap_classifier_dr.pdf](/Users/awotoroebenezer/Desktop/MetaMP-Server/.metamp-publication/runs/20260327T163411Z/copied/publication_figures/model_registry/fig4_heatmap_classifier_dr.pdf)
- [fig5_cv_expert_gap_boxplot.pdf](/Users/awotoroebenezer/Desktop/MetaMP-Server/.metamp-publication/runs/20260327T163411Z/copied/publication_figures/model_registry/fig5_cv_expert_gap_boxplot.pdf)
- [fig6_bubble_cv_expert.pdf](/Users/awotoroebenezer/Desktop/MetaMP-Server/.metamp-publication/runs/20260327T163411Z/copied/publication_figures/model_registry/fig6_bubble_cv_expert.pdf)
- [fig7_parallel_coordinates.pdf](/Users/awotoroebenezer/Desktop/MetaMP-Server/.metamp-publication/runs/20260327T163411Z/copied/publication_figures/model_registry/fig7_parallel_coordinates.pdf)
- [fig8_dr_subplot_comparison.pdf](/Users/awotoroebenezer/Desktop/MetaMP-Server/.metamp-publication/runs/20260327T163411Z/copied/publication_figures/model_registry/fig8_dr_subplot_comparison.pdf)

Important note:

- These figures correctly reflect only `no_dr`, `pca`, and `umap` as production DR methods.
- `t-SNE` is not a production-bundle column in the current registry.

### B. Legacy manuscript figure references that are not currently present in this workspace

The current `paper.tex` still references several legacy `images/...` assets that were not found in the present workspace audit:

- `images/ETL.png`
- `images/chart_exports/database_year_proportional_representation.pdf`
- `images/welcomePage.pdf`
- `images/chart_exports/Inconsistencies.pdf`
- `images/singleViewCropOptim.pdf`
- `images/singleViewCrop1.pdf`
- `images/singleViewCrop2.pdf`

Status:

- `database_year_proportional_representation.pdf` exists in the live publication snapshot, but not under the legacy `images/chart_exports/` path used in the manuscript
- `semi_supervised_no_dr_decision_tree_shap_bar.pdf` also exists in the live production snapshot and is consistent with the current methods/results discussion
- the other listed legacy `images/...` assets were not found in this workspace during the present audit

Implication:

- if the manuscript is built from this checkout, those legacy image paths should not be assumed to be valid
- for a final submission package, the figure references should either:
  - be updated to the current publication snapshot assets, or
  - the missing legacy files should be restored into the manuscript asset directory

## Clean Answers to the Reviewer’s Implied Questions

### What threshold was used for self-training?

Current answer:

- The code does not set a project-specific threshold explicitly.
- With sklearn `1.3.2`, `SelfTrainingClassifier` defaults to `criterion='threshold'` and `threshold=0.75`.

### What was the maximum number of self-training iterations?

Current answer:

- The code does not override the sklearn default.
- With sklearn `1.3.2`, `max_iter=10`.

### What was the stopping criterion?

Current answer:

- The current implementation uses the stopping behavior of sklearn `SelfTrainingClassifier`.
- Training terminates when:
  - `max_iter` is reached, or
  - no new pseudo-labels are added, or
  - all unlabeled samples are labeled

### Were all models wrapped with identical self-training settings?

Current answer:

- Yes.
- All semi-supervised classifier families are wrapped using the same `SelfTrainingClassifier(clf)` call path.

### Were pseudo-labels regenerated independently within each fold?

Current answer:

- No, not in the sense implied by fold-wise semi-supervised cross-validation.
- The current semi-supervised implementation uses a labeled train/test split plus held-out expert benchmarking.
- The supervised CV path is separate.

### Are supervised and semi-supervised workflows evaluated identically?

Current answer:

- No.
- They share classifier families, feature definitions, benchmark exclusions, and external expert benchmarking.
- But they do not share the same internal evaluation procedure:
  - supervised mode reports 5-fold stratified CV on labeled data
  - semi-supervised mode reports labeled hold-out split performance plus held-out expert benchmarking

### Are the scientific assessment flags rule-based and inspectable?

Current answer:

- Yes.
- The rule logic is public and inspectable in [src/Dashboard/scientific_assessment.py](/Users/awotoroebenezer/Desktop/MetaMP-Server/src/Dashboard/scientific_assessment.py).
- They are heuristics, not a formally validated probabilistic model.

### Is there an explicit daily or weekly scheduled refresh hierarchy?

Current answer:

- Not in the currently audited code.
- The built-in periodic schedule clearly configured in code is monthly production maintenance.

## Final Bottom Line

The remaining critiques are mostly valid and manageable. The most important correction is to describe the supervised and semi-supervised workflows separately and honestly:

- supervised: 5-fold stratified CV plus held-out expert benchmark
- semi-supervised: sklearn self-training defaults on labeled plus unlabeled data, labeled hold-out evaluation, plus held-out expert benchmark

The figure situation also needs to be handled transparently:

- the current live publication and production-ML figures are present and usable
- several legacy manuscript `images/...` references are not present in this workspace and should not be treated as guaranteed submission assets without restoration or path correction

If this document is used as the basis for a reviewer response, the manuscript can be defended as a serious and mature platform methodology, with the ML section still requiring the most careful wording discipline.
