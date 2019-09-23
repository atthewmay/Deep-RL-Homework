#!/bin/bash
batch_sizes='1 3 5'
lr_list='5e-3 1e-2 2e-2'
for batch_size in $batch_sizes
	do 
		for lr in $lr_list
			do
				let bs=$batch_size\*10000
				echo bs
				echo save file is halfc_b"$bs"_lr"$lr"

				# python train_pg_f18.py InvertedPendulum-v2 -ep 1000 --discount 0.9 -n 100 -e 1 -b $bs -lr "$lr" -rtg --exp_name halfc_b"$bs"_r"$lr" --script_optimizing_dir "$1"_lr"$lr"
				python train_pg_f18.py HalfCheetah-v2 -ep 150 --discount 0.9 -n 100 -e 3 -l 2 -s 32 -b $bs -lr "$lr" -rtg --nn_baseline --exp_name halfc_b"$bs"_r"$lr" --script_optimizing_dir "$1"
			done
	done

	# echo All Done