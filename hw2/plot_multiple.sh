#!/bin/bash
all_dirs_in_dir=data/$1/*
echo plotting $all_dirs_in_dir

python plot.py $all_dirs_in_dir
# for direc in $all_dirs_in_dir
# 	do
# 		# python train_pg_f18.py InvertedPendulum-v2 -ep 1000 --discount 0.9 -n 100 -e 1 -b $bs -lr 5e-3 -rtg --exp_name hc_b500_r5e-3 --script_optimizing_dir opt_b
# 	done

# echo All Done