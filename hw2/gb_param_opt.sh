#!/bin/bash

lr_list='5e-4 5e-3'

for lr in $lr_list
    do
        echo training model with learning rate "$lr"
        python train_pg_f18.py GB_game -ep 10000 --discount 0.99 -n 4000 -l 2 -s 64 -b 100000 -lr "$lr" --exp_name GB_sb_ndp_rc_rp_discrete_lr_"$lr" -rc -rp --save_models --save_best_model\
         --script_optimizing_dir gb_discrete --num_enemies 4 --gb_discrete --output_activation None --gb_max_speed 10 &
    done

# s_list='16 64'
# for s in $s_list
#     do

#         python train_pg_f18.py GB_game -ep 10000 --discount 0.99 -n 1000 -l 2 -s "$s" -b 30000 -lr 1e-4 --exp_name GB_sb_ndp_rc_rp_discrete_lr_1e-4_s_"$s" -rc -rp --save_models --save_best_model\
#          --script_optimizing_dir gb_discrete --num_enemies 4 --gb_discrete --output_activation None --gb_max_speed 10 &
#     done
# 	# echo All Done

