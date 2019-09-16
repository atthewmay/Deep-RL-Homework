#!/bin/bash
batch_sizes='1 2 3 4 5 6 7 8 9'
lr=5e-2
for batch_size in $batch_sizes
	do
		let bs=$batch_size\*1000
		echo bs
		echo save file is hc_b"$bs"_lr"$lr"
		python train_pg_f18.py InvertedPendulum-v2 -ep 1000 --discount 0.9 -n 100 -e 1 -b $bs -lr "$lr" -rtg --exp_name hc_b"$bs"_r"$lr" --script_optimizing_dir "$1"_lr"$lr"
	done

echo All Done