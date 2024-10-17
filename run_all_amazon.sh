#!/bin/bash

cd ${WORK_DIR}isolated_text_effects/

# MODELS=(lr gbm mlp)

# for MODEL in "${MODELS[@]}"; do
    # if [ $MODEL != "lr" ]; then
    #     sbatch run_amazon.sh $MODEL cv plugin se2
    #     sbatch run_amazon.sh $MODEL cv dr se2
    # fi
    # sbatch run_amazon.sh $MODEL cv plugin se2 "--truncate 0.01"
    # sbatch run_amazon.sh $MODEL cv dr se2 "--truncate 0.01"
    # sbatch run_amazon.sh $MODEL cv plugin se2 "--truncate 0.05"
    # sbatch run_amazon.sh $MODEL cv dr se2 "--truncate 0.05"
#     sbatch run_amazon.sh $MODEL cv plugin se2 "--truncate 0.1"
#     sbatch run_amazon.sh $MODEL cv dr se2 "--truncate 0.1"
# done

MODELS=(lr gbm mlp)
BOUNDS=(bound)
# MODELS=(lr)
# BOUNDS=(bound)
# STRAT=1

for MODEL in "${MODELS[@]}"; do
    for BOUND in "${BOUNDS[@]}"; do
        for STRAT in {1..3}; do
            sbatch -J "a${STRAT}-${MODEL}_${BOUND}" run_amazon_robustness_RV0.sh $MODEL cv dr $BOUND se2 "" no_ood_ $STRAT
            sbatch -J "a${STRAT}-0.01_${MODEL}_${BOUND}" run_amazon_robustness_RV0.sh $MODEL cv dr $BOUND se2 "--truncate 0.01" no_ood_trunc0.01_ $STRAT
            sbatch -J "a${STRAT}-0.05_${MODEL}_${BOUND}" run_amazon_robustness_RV0.sh $MODEL cv dr $BOUND se2 "--truncate 0.05" no_ood_trunc0.05_ $STRAT
            sbatch -J "a${STRAT}-0.1_${MODEL}_${BOUND}" run_amazon_robustness_RV0.sh $MODEL cv dr $BOUND se2 "--truncate 0.1" no_ood_trunc0.1_ $STRAT
            
            # if [ $MODEL != "lr" ]; then
            # sbatch run_amazon_robustness.sh $MODEL cv plugin $BOUND se2 "" no_ood_
            # sbatch run_amazon_robustness.sh $MODEL cv dr $BOUND se2 "" no_ood_
            # fi
            # sbatch run_amazon_robustness.sh $MODEL cv plugin $BOUND se2 "--truncate 0.01" no_ood_trunc0.01_
            # sbatch run_amazon_robustness.sh $MODEL cv dr $BOUND se2 "--truncate 0.01" no_ood_trunc0.01_
            # sbatch run_amazon_robustness.sh $MODEL cv plugin $BOUND se2 "--truncate 0.05" no_ood_trunc0.05_
            # sbatch run_amazon_robustness.sh $MODEL cv dr $BOUND se2 "--truncate 0.05" no_ood_trunc0.05_
            # sbatch run_amazon_robustness.sh $MODEL cv plugin $BOUND se2 "--truncate 0.1" no_ood_trunc0.1_
            # sbatch run_amazon_robustness.sh $MODEL cv dr $BOUND se2 "--truncate 0.1" no_ood_trunc0.1_
    
        done
    done
done
