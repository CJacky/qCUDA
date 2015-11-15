#!/bin/bash

set -e

cpu=mmul_cpu
gpu=mmul_gpu

make $cpu $gpu elapsed

rm -rf mmul_data_host.txt

for i in $(seq 1 10)
do
	echo "$i run"

	for (( j=16; j<=8192; j=j*2 ))
	do
		echo "    $j dim"
#if [ $j -le 2048 ]; then
#			CPU=`./elapsed ./$cpu $j $j $j`
#		else
#			CPU=0
#		fi
#		GPU=`./elapsed ./$gpu $j $j $j`
#		echo -e "time $j $GPU $CPU" >> mmul_data_host.txt
		GPU=`./$gpu $j $j $j`
		echo -e "$j\t$GPU" >> mmul_data_host.txt
	done

done
