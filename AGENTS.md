# project instructions
What: This is an analytics project where a set of customers/vehicle keys are filtered out a from a larger base of all the vehicles for a specific milestone [20,30,40,50,60,70,80,90,100] and find the probability of them coming over in the expected quarter

training: for each milestone, find the vehicles (by VIN/Vehicle keys) that had previously completed that milestone from the source table service_history_Q2_2026 (servic history upto Q2 2026, will be updated on each quarter), calculate and produce a feature matrix. for labelling, check if they appeared in the quarter they were expected, if no then 0, if they ever have done that experience upto date, then 1. the model will be trained on these

> **Labelling — intended vs actual.** The rule above (turned up *in the expected quarter*) is the
> goal. The code does NOT do this yet: `pmstrainfeatureeng_refactored.py:136` labels 1 for "last PMS
> == this milestone" with no date condition — i.e. completed it *ever* — and uses the quarter only to
> choose which still-pending vehicles become the 0s. Known defect, not yet fixed. It is also why
> `Service_Num` leaks the label (it is stamped only after turn-up). Do not change the rule as a side
> effect of another task — it invalidates every training matrix and published metric. See CLAUDE.md.
testing: upon the training, we will make a testing set, without any leakage (if we are testing for Q2 2025, then training might be upto Q4 2024 (not a rule, could be closer or farther)). the features, similar to the features we made in our training pipeline would produced for this as well, and then we will predict for this expected cohort, validate with the actual scenario (if the quarter had already completed) and then complete the evaluation metrics

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
