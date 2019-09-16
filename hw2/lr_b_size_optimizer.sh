#!/bin/bash
batch_sizes='1 2 3 4 5 6 7 8 9'
for batch_size in $batch_sizes
	do
		let a=$batch_size\*1000
		echo $a
	done

echo All Done