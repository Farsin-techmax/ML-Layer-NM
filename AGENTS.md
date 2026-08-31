# project instructions
What: This is an analytics project where a set of customers/vehicle keys are filtered out a from a larger base of all the vehicles for a specific milestone [20,30,40,50,60,70,80,90,100] and find the probability of them coming over in the expected quarter

training: for each milestone, find the vehicles (by VIN/Vehicle keys) that had previously completed that milestone from the source table service_history_Q2_2026 (servic history upto Q2 2026, will be updated on each quarter), calculate and produce a feature matrix. the model will be trained on these

> **Labelling — POC convention, decided 2026-08-12.** *Ever completed*, no quarter-strictness:
> `1` = the vehicle has done this milestone at any point up to the latest data, `0` = it was expected
> for the milestone and is still pending past its expected date. The quarter selects **which pending
> cohort is sampled as the 0s**, nothing more. This is what both pipelines already do —
> `pmstrainfeatureeng_refactored.py:136` and legacy `pmstrainfeatureEng_6.py:229-234`
> (`turnup = df.query("\`Last Service - PMS\` == '{svc}'")`, `TargetFlag = 1`) — so it is the
> convention, not a defect. An earlier note here called it one; that is superseded.
>
> Two things follow from it and are accepted for the POC: `Service_Num` still leaks the label
> (stamped only on turn-up, so it is dropped explicitly — see CLAUDE.md gotchas), and a `0` can
> become a `1` in a later extract when the customer finally turns up, so labels are a function of
> the extract date. Do not switch to a quarter-conditioned rule without saying so — it invalidates
> every training matrix and published metric.
>
> **A third consequence, measured 2026-08-16 and now the pipeline's biggest open problem.** Because
> positives are drawn from all time and negatives from a single due-date window, the two cohorts
> differ systematically in how much service history they have — so history-derived features carry
> the *opposite* sign in training to the one they have at test time. On 20k, `has_10` scores raw AUC
> 0.3602 in training and 0.6885 on the Q1 test set; `PMS_Count_Prior` 0.3792 vs 0.7224. Shuffling
> the whole schedule-keeping family **raises** 20k test AUC by 0.19. This is the labelling
> convention working as specified, not a bug — but it caps what the 20k model can reach
> (AUC 0.62 vs 0.95 on 60k, where the milestones nearest the target stay consistent). See the
> CLAUDE.md gotcha for the full table.
>
> **The obvious fix was tried on 2026-08-16 and did not work.** `--label-mode window` makes the
> positives share the negatives' due-date window, which is what the test cohorts already do. It
> improved calibration but **lost AUC on 6 of 8 milestone/quarter combinations** (40k worst, −0.13
> Q1 / −0.14 Q2); 20k stayed broken either way, with Q2 AUC **below 0.5**. Two prerequisites came
> out of that attempt and must be fixed before the labelling rule is revisited: `Next{m}K_Due` is
> back-filled from the actual turn-up date for `ever`-mode positives, and the expected-date formula
> runs ~12.7 months early at 60k. Both are documented in CLAUDE.md. **So the 'ever' convention
> stands as the default** — not because it is right, but because the alternative is not yet better.

testing: same labelling convention as training — ever completed = 1, expected-but-pending = 0. Build
the test set without leakage (if we are testing for Q2 2025, then training might be upto Q4 2024 (not
a rule, could be closer or farther)). the features, similar to the features we made in our training
pipeline would produced for this as well, and then we will predict for this expected cohort, validate
with the actual scenario and then complete the evaluation metrics.

> One deliberate train/test difference: a vehicle that completed the milestone **before the feature
> cutoff** is a `1` in training but is dropped from the test cohort entirely
> (`prepare_test_set.py:76-82`). Its outcome is already known at scoring time, so scoring it would
> inflate the metrics.

tech stack: Numpy, Pandas, Pytorch, Matplotlib, SHAP, Plotly, other important libraries. this is the data processing and scoring pipeline part of the project.

project structure and arch: 
/data : all the master sheets and source tables
/models: model weights + scaler/imputer/selected_features per milestone. legacy runs in /models/{m}k,
         current retrain.py runs in /models/models_alan/{m}k. training matrices are NOT here — they
         live in /tests_alan/traindataq1_q2 and /refactored_test_dir
/test_sets: test cohort keys for each milestone that will be used for scoring the model (only
         20/30/40/50k exist today, and their column set is wrong — see CLAUDE.md gotchas)
/validatecode: earlier used for storing intermitten artefacts like clusters and final merged datasets
         for each milestone for training. NOTE it is still a live *input*: predservicemil_4.py:2307
         reads Model_clusters_{m}.csv / Variant_clusters_{m}.csv / Nationality_clusters_{m}.csv here
/refactored_test_dir: same idea as validatecode, for the refactored pipeline
(no /plots dir — plots are written next to the weights, e.g. models/models_alan/{m}k/shap_summary.png)

main files: pmstrainfeatureeng_refactored.py - the file with refactoring and fixes applied to the legacy script pmstrainfeatureEng_6.py (capital E)

prepare_test_set.py: finding and making the test set, predservicemil_4.py is the legacy

train_milestone.py and retrain.py: model training scripts, training_run.py is the legacy

score_milestone.py: is the scoring of the expected set we made in prepare_test_set.py for each milestone for each quarter. prediction_run.py is the legacy

features.py: the shared module for each of the features shared between the training and testing pipeline

eval_test_metrics.py: evaluation scripts, eval_alan_tests.py was for testing the test set which i obtained externally and stored in /tests_alan

milestone_features.py: to add a few milestone based features to the training matrix

# coding style:
code in simple mode, act like a intermediate level/staff level software engineer, with each change or modifications, make the user also aware about that and teach them the steps, and often ask clarifying questions to make sure they caught the context

# PR and commit
before committing anything, make the changes visible in plain langauge and only commit upon approval
