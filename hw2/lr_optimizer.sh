#!/bin/bash
bs=3000
lr_list='1e0 5e-1 1e-1 '
for lr in $lr_list
	do
	
		echo save file is hc_b"$bs"_lr"$lr"
		python train_pg_f18.py InvertedPendulum-v2 -ep 1000 --discount 0.9 -n 100 -e 1 -b $bs -lr "$lr" -rtg --exp_name hc_b"$bs"_r"$lr" --script_optimizing_dir "$1"_bs"$bs"
	done

echo All Done