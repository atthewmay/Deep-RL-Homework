#!/bin/bash
batch_sizes='1'
bllr_list='1e-4 5e-3 1e-2 5e-2 1e-1'
for batch_size in $batch_sizes
	do 
		for bllr in $bllr_list
			do
				let bs=$batch_size\*10000
				echo bs
					python train_pg_f18.py HalfCheetah-v2 -ep 150 --discount 0.9 -n 100 -e 3 -l 2 -s 32 -b $bs -lr 5e-3 --baseline_lr $bllr -rtg --nn_baseline --exp_name halfc_b"$bs"_bllr"$bllr" --script_optimizing_dir "$1"
			done
	done

	# echo All Done