#!/bin/bash

cd ${WORK_DIR}isolated_text_effects/

# MODELS=(lr svm gbm mlp)
# MODELS=(lr gbm mlp)
MODELS=(gbm)
# MODELS=(gbm)
# MODELS=(lr)
# MODELS=(gbm mlp)
BOUND_TYPE=(bound)
# STRAT=3
STRAT=1

for MODEL in "${MODELS[@]}"; do
    # sbatch run_tirzepatide_all_reps.sh $MODEL cv dr
    # sbatch run_tirzepatide_all_reps.sh $MODEL cv dr "--truncate 0.01"
    # sbatch run_tirzepatide_all_reps.sh $MODEL cv dr "--truncate 0.05"
    # sbatch run_tirzepatide_all_reps.sh $MODEL cv dr "--truncate 0.1"
    for BOUND in "${BOUND_TYPE[@]}"; do
        # for STRAT in {1..3}; do
            # sbatch run_tirzepatide_robustness.sh $MODEL cv dr $BOUND "" no_ood_
            # sbatch run_tirzepatide_robustness.sh $MODEL cv dr $BOUND "--truncate 0.01" no_ood_trunc0.01_
            # sbatch run_tirzepatide_robustness.sh $MODEL cv dr $BOUND "--truncate 0.05" no_ood_trunc0.05_
            # sbatch run_tirzepatide_robustness.sh $MODEL cv dr $BOUND "--truncate 0.1" no_ood_trunc0.1_
            # sbatch -J "s${STRAT}_t${MODEL}_${BOUND}" run_tirzepatide_robustness_RV0.sh $MODEL cv dr $BOUND "" no_ood_ $STRAT
            # sbatch -J "s${STRAT}_t0.01_${MODEL}_${BOUND}" run_tirzepatide_robustness_RV0.sh $MODEL cv dr $BOUND "--truncate 0.01" no_ood_trunc0.01_ $STRAT
            # sbatch -J "s${STRAT}_t0.05_${MODEL}_${BOUND}" run_tirzepatide_robustness_RV0.sh $MODEL cv dr $BOUND "--truncate 0.05" no_ood_trunc0.05_ $STRAT
            # sbatch -J "s${STRAT}_t0.1_${MODEL}_${BOUND}" run_tirzepatide_robustness_RV0.sh $MODEL cv dr $BOUND "--truncate 0.1" no_ood_trunc0.1_ $STRAT
            # if [ $MODEL == "lr" ]; then
            # sbatch -J "s${STRAT}_t${MODEL}_${BOUND}" run_tirzepatide_save_models.sh $MODEL cv dr $BOUND "" no_ood_ $STRAT ""
            # sbatch -J "s${STRAT}_t0.01${MODEL}_${BOUND}" run_tirzepatide_save_models.sh $MODEL cv dr $BOUND "--truncate 0.01" no_ood_trunc0.01_ $STRAT ""
            # sbatch -J "s${STRAT}_t0.05${MODEL}_${BOUND}" run_tirzepatide_save_models.sh $MODEL cv dr $BOUND "--truncate 0.05" no_ood_trunc0.01_ $STRAT ""
            # sbatch -J "s${STRAT}_t0.1${MODEL}_${BOUND}" run_tirzepatide_save_models.sh $MODEL cv dr $BOUND "--truncate 0.1" no_ood_trunc0.1_ $STRAT ""
            # else
            sbatch -J "t${STRAT}-${MODEL}_${BOUND}" run_tirzepatide_save_models.sh $MODEL cv dr $BOUND "" no_ood_ $STRAT "--compute-ovb-grid"
            # sbatch -J "t${STRAT}-0.1${MODEL}_${BOUND}" run_tirzepatide_save_models.sh $MODEL cv dr $BOUND "--truncate 0.1" no_ood_trunc0.1_ $STRAT "--compute-ovb-grid"
            # sbatch -J "t${STRAT}-0.01${MODEL}_${BOUND}" run_tirzepatide_save_models.sh $MODEL cv dr $BOUND "--truncate 0.01" no_ood_trunc0.01_ $STRAT ""
            # sbatch -J "t${STRAT}-0.05${MODEL}_${BOUND}" run_tirzepatide_save_models.sh $MODEL cv dr $BOUND "--truncate 0.05" no_ood_trunc0.05_ $STRAT ""
            # fi
        # done
    done
    # sbatch run_tirzepatide.sh $MODEL cv plugin
    # sbatch run_tirzepatide.sh $MODEL cv dr
    # sbatch run_tirzepatide.sh $MODEL cv plugin "--truncate 0.01"
    # sbatch run_tirzepatide.sh $MODEL cv dr "--truncate 0.01"
    # sbatch run_tirzepatide.sh $MODEL cv plugin "--truncate 0.05"
    # sbatch run_tirzepatide.sh $MODEL cv dr "--truncate 0.05"
    # sbatch run_tirzepatide.sh $MODEL cv plugin "--truncate 0.1"
    # sbatch run_tirzepatide.sh $MODEL cv dr "--truncate 0.1"
done