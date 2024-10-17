#!/bin/bash

MODELS=(lr)
BOUNDS=(bound)
STRAT=2

for MODEL in "${MODELS[@]}"; do
    for BOUND in "${BOUNDS[@]}"; do
        ./run_amazon_robustness_RV0.sh $MODEL cv dr $BOUND se2 "" no_ood_ $STRAT
    done
done