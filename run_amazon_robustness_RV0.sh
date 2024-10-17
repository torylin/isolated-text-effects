#!/bin/bash
#
#SBATCH -p low
#SBATCH --mem=12G
#SBATCH -o ${WORK_DIR}slurm/logs/log%j.out        # STDOUT. %j specifies JOB_ID.
#SBATCH -e ${WORK_DIR}slurm/logs/log%j.err        # STDERR. See the first link for more options.
#SBATCH --mail-type=ALL

source ${WORK_DIR}miniconda3/etc/profile.d/conda.sh
conda activate dclf

cd ${WORK_DIR}isolated_text_effects/

CLASSIFIER=$1
G=$1
BOOTSTRAP_ITERS=1000
CV_FOLDS=5
SPLIT_TYPE=$2
NORM_LIST=()
# NORM_LIST=(norm_a norm_total)
NU_TYPE=$3
RV_TYPE=$4
SE_TYPE=$5
TRUNCATE=$6
if [ $SPLIT_TYPE == "cv" ]; then
    OUTPUT_CSV=${WORK_DIR}isolated_text_effects/results/amazon_synthetic/ovb/${SE_TYPE}/norm_none/pr_label_synthetic_direct_iso_effects_RV0${RV_TYPE}.csv
elif [ $SPLIT_TYPE == "bootstrap" ]; then
    OUTPUT_CSV=${WORK_DIR}isolated_text_effects/results/amazon_synthetic/ovb/bootstrap/norm_none/pr_label_synthetic_direct_iso_effects_RV0${RV_TYPE}.csv
fi
# OOD='no_ood_'
OOD=$7
# OOD=''
RV_INTERVAL=0.01
MODEL_DIR=${WORK_DIR}isolated_text_effects/models/amazon/
# SAVE_MODELS="--save-models"
SAVE_MODELS=""
SAVE_RESULTS="--save-results"


# for STRAT in {1..3}; do
STRAT=$8
NUM_FEATURES=2
python estimate.py \
    --data-dir ${WORK_DIR}data/causal_text/amazon_synthetic/ \
    --bootstrap-iters $BOOTSTRAP_ITERS \
    --cv-folds $CV_FOLDS \
    --split-type $SPLIT_TYPE \
    --output-csv $OUTPUT_CSV \
    --estimation-strat $STRAT \
    --clf $CLASSIFIER \
    --g $G \
    --num-features $NUM_FEATURES \
    --nu-type $NU_TYPE \
    --RV-type $RV_TYPE \
    --RV-interval $RV_INTERVAL \
    --no-ood \
    --RV0 \
    --se-type $SE_TYPE \
    --scale \
    --model-dir $MODEL_DIR \
    $TRUNCATE \
    $SAVE_MODELS \
    $SAVE_RESULTS \

# for NORM_TYPE in "${NORM_LIST[@]}"; do
#     OUTPUT_CSV_NORM=${WORK_DIR}isolated_text_effects/results/amazon_synthetic/ovb/${SE_TYPE}/${NORM_TYPE}/pr_label_synthetic_direct_iso_effects_RV0${RV_TYPE}.csv
#     python estimate.py \
#         --data-dir ${WORK_DIR}data/causal_text/amazon_synthetic/ \
#         --bootstrap-iters $BOOTSTRAP_ITERS \
#         --cv-folds $CV_FOLDS \
#         --split-type $SPLIT_TYPE \
#         --output-csv $OUTPUT_CSV_NORM \
#         --estimation-strat $STRAT \
#         --normalize-weights $NORM_TYPE \
#         --clf $CLASSIFIER \
#         --g $G \
#         --num-features $NUM_FEATURES \
#         --nu-type $NU_TYPE \
#         --RV-type $RV_TYPE \
#         --RV-interval $RV_INTERVAL \
#         --no-ood \
#         --RV0 \
#         --se-type $SE_TYPE \
#         --scale \
#         --model-dir $MODEL_DIR \
#         $TRUNCATE \
#         --save-models
#         # --save-results \
# done

for NUM_FEATURES in {3..10}; do
    python estimate.py \
        --data-dir ${WORK_DIR}data/causal_text/amazon_synthetic/ \
        --bootstrap-iters $BOOTSTRAP_ITERS \
        --cv-folds $CV_FOLDS \
        --split-type $SPLIT_TYPE \
        --output-csv $OUTPUT_CSV \
        --estimation-strat $STRAT \
        --clf $CLASSIFIER \
        --g $G \
        --num-features $NUM_FEATURES \
        --nu-type $NU_TYPE \
        --RV-type $RV_TYPE \
        --RV-interval $RV_INTERVAL \
        --append \
        --no-ood \
        --RV0 \
        --se-type $SE_TYPE \
        --scale \
        --model-dir $MODEL_DIR \
        $TRUNCATE \
        $SAVE_MODELS \
        $SAVE_RESULTS \

    # for NORM_TYPE in "${NORM_LIST[@]}"; do
    #     OUTPUT_CSV_NORM=${WORK_DIR}isolated_text_effects/results/amazon_synthetic/ovb/${SE_TYPE}/${NORM_TYPE}/pr_label_synthetic_direct_iso_effects_RV0${RV_TYPE}.csv
    #     python estimate.py \
    #         --data-dir ${WORK_DIR}data/causal_text/amazon_synthetic/ \
    #         --bootstrap-iters $BOOTSTRAP_ITERS \
    #         --cv-folds $CV_FOLDS \
    #         --split-type $SPLIT_TYPE \
    #         --save-results \
    #         --output-csv $OUTPUT_CSV_NORM \
    #         --estimation-strat $STRAT \
    #         --normalize-weights $NORM_TYPE \
    #         --clf $CLASSIFIER \
    #         --g $G \
    #         --num-features $NUM_FEATURES \
    #         --nu-type $NU_TYPE \
    #         --RV-type $RV_TYPE \
    #         --RV-interval $RV_INTERVAL \
    #         --append \
    #         --no-ood \
    #         --RV0 \
    #         --se-type $SE_TYPE \
    #         --scale \
    #         --model-dir $MODEL_DIR \
    #         $TRUNCATE \
    #         --save-models
    #         # --save-results \
    # done
done
# done