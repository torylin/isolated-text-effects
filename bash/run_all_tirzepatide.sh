#!/bin/bash

MODELS=(gbm)
BOUND_TYPE=(bound)
STRAT=1

for MODEL in "${MODELS[@]}"; do
    for BOUND in "${BOUND_TYPE[@]}"; do
        ./run_tirzepatide_save_models.sh $MODEL cv dr $BOUND "" no_ood_ $STRAT "--compute-ovb-grid"
    done
done