#!/bin/bash
#
#SBATCH -p low
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
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
# NORM_LIST=(norm_a norm_total)
NORM_LIST=()
NU_TYPE=$3
RV_TYPE=$4
SE_TYPE=se2
TRUNCATE=$5
# OUTPUT_CSV=${WORK_DIR}isolated_text_effects/results/tirzepatide/discrete_covs/norm_none/discrete_vars_only.csv
OUTPUT_CSV=${WORK_DIR}isolated_text_effects/results/tirzepatide/text_covs/norm_none/all_reps_RV0${RV_TYPE}.csv
CY=0
CD=0
COVARIATES=from_header
OUTCOME=target_achieved
DATA_DIR=${WORK_DIR}data/causal_text/tirzepatide/
TRUE_EFFECTS_PATH=true_effect.csv
CSV_PATH=filtered_clean_discrete_vars_only.csv
OOD=$6
RV_INTERVAL=0.01
STRAT=$7
OVB_GRID=$8
MODEL_DIR=${WORK_DIR}isolated_text_effects/models/tirzepatide/text_covs/norm_none/
SAVE_MODELS="--save-models"
# SAVE_RESULTS="--save-results"
SAVE_RESULTS=""
DROP_FEAT=$9
SAVE_PREDS="--save-preds"

# for STRAT in {1..3}; do
# python estimate.py \
#     --data-dir $DATA_DIR \
#     --true-effects-path $TRUE_EFFECTS_PATH \
#     --csv-path $CSV_PATH \
#     --covariates $COVARIATES \
#     --outcome $OUTCOME \
#     --bootstrap-iters $BOOTSTRAP_ITERS \
#     --cv-folds $CV_FOLDS \
#     --split-type $SPLIT_TYPE \
#     --output-csv $OUTPUT_CSV \
#     --estimation-strat $STRAT \
#     --clf $CLASSIFIER \
#     --g $G \
#     --Cy $CY \
#     --Cd $CD \
#     --nu-type $NU_TYPE \
#     --RV-type $RV_TYPE \
#     --RV-interval $RV_INTERVAL \
#     --RV0 \
#     --no-ood \
#     --se-type $SE_TYPE \
#     --scale \
#     --model-dir $MODEL_DIR \
#     $DROP_FEAT \
#     $OVB_GRID \
#     $TRUNCATE \
#     $SAVE_MODELS \
#     $SAVE_RESULTS \
#     $SAVE_PREDS

OUTPUT_CSV=${WORK_DIR}isolated_text_effects/results/tirzepatide/text_covs/norm_none/all_reps_RV0${RV_TYPE}.csv
COVARIATES=text
CSV_PATH=filtered_clean_masked.csv
# TEXT_COLS=(comment_masked post_masked comment post)
TEXT_COLS=(comment)
# LM_LIBRARIES=(lexicon lexicon sentecon_liwc sentecon_empath sentence-transformers)
LM_LIBRARIES=(sentecon_empath)
# LM_NAMES=(liwc empath all-mpnet-base-v2 all-mpnet-base-v2 all-mpnet-base-v2)
LM_NAMES=(all-mpnet-base-v2)
for TEXT_COL in "${TEXT_COLS[@]}"; do
    for i in "${!LM_LIBRARIES[@]}"; do
        LM_LIBRARY=${LM_LIBRARIES[$i]}
        LM_NAME=${LM_NAMES[$i]}
        # for STRAT in {1..3}; do
        python estimate.py \
            --data-dir $DATA_DIR \
            --true-effects-path $TRUE_EFFECTS_PATH \
            --csv-path $CSV_PATH \
            --covariates $COVARIATES \
            --outcome $OUTCOME \
            --bootstrap-iters $BOOTSTRAP_ITERS \
            --cv-folds $CV_FOLDS \
            --split-type $SPLIT_TYPE \
            --output-csv $OUTPUT_CSV \
            --estimation-strat $STRAT \
            --clf $CLASSIFIER \
            --g $G \
            --Cy $CY \
            --Cd $CD \
            --nu-type $NU_TYPE \
            --no-ood \
            --RV-type $RV_TYPE \
            --RV-interval $RV_INTERVAL \
            --RV0 \
            --se-type $SE_TYPE \
            --text-col $TEXT_COL \
            --lm-library $LM_LIBRARY \
            --lm-name $LM_NAME \
            --append \
            --scale \
            --model-dir $MODEL_DIR \
            $DROP_FEAT \
            $OVB_GRID \
            $TRUNCATE \
            $SAVE_MODELS \
            $SAVE_RESULTS \
            $SAVE_PREDS
    done
done