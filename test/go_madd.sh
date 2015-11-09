#!/bin/bash

set -e

cpu=madd_cpu
run=madd_gpu
drv=madd_gpu_drv

i=1000

make $cpu $run $drv elapsed

#for i in $(seq 1 10)
#do
#	echo $i run

	for (( j=16; j<=8192; j=j*2 ))
	do
		if [ "$j" -le 2048 ]; then
		CPU=`./elapsed ./$cpu $i $j $j`
		else
		CPU=0
		fi
		RUN=`./elapsed ./$run $i $j $j`
		DRV=`./elapsed ./$drv $i $j $j`
		echo -e "$j\t$RUN\t$DRV\t$CPU"
	done

#done
