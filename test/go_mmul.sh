#!/bin/bash

set -e

cpu=mmul_cpu
run=mmul_gpu
drv=mmul_gpu_drv

make $cpu $run $drv elapsed

#for i in $(seq 1 10)
#do
#	echo $i run

	for (( j=16; j<=8192; j=j*2 ))
	do
		if [ "$j" -le 2048 ]; then
		CPU=`./elapsed ./$cpu $j $j $j`
		else
		CPU=0
		fi
		RUN=`./elapsed ./$run $j $j $j`
		DRV=`./elapsed ./$drv $j $j $j`
		echo -e "$j\t$RUN\t$DRV\t$CPU"
	done

#done
