#!/bin/bash
# Long layout-evolution run, intended to run from Thu evening until Mon ~06:00.
# Big 50x50 archive, res_ind measures, dense fitness, checkpointing every 50 gens.
# caffeinate keeps the Mac awake (idle/display/system). NOTE: closing the laptop
# lid will still sleep the machine unless on AC power in clamshell mode.
cd "$(dirname "$0")"
exec caffeinate -dimsu /usr/bin/python3 qd_train.py \
    --policy layout \
    --gens 500000 \
    --emitters 8 --batch 20 --workers 8 \
    --fitness dense \
    --n-actions 1 --ticks-per-action 100000 \
    --measures res_ind \
    --archive-dims 50,50 \
    --save results/weekend_layout_resind.npz \
    --save-every 50 \
    --log results/weekend_layout_resind.log
