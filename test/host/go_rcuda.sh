#!/bin/bash

set -e

################################################################################
function bench_mmul(){
f=$1
make -C matrixMul clean all
rm -f $f

for i in $(seq 1 $2)
do
	echo "##################################################################"
	echo "mmul $i / $2"
	for (( j=$3; j<=$4; j=j*2 ))
	do
		echo "    $j size"
		/usr/bin/time -f "%e" -o tmp_output ./matrixMul/matrixMul -wA=$j -hA=$j -wB=$j -hB=$j > /dev/null
		real=`cat tmp_output`
		echo "$j $real" >> $f
	done
done
rm -f tmp_output
}

################################################################################
function bench_vadd(){
f=$1
make -C vectorAdd clean all
rm -f $f

for i in $(seq 1 $2)
do
	echo "##################################################################"
	echo "vadd $i / $2"
	for (( j=$3; j<=$4; j=j*2 ))
	do
		size=$(($j*1000000))
		echo "    $size size"
		/usr/bin/time -f "%e" -o tmp_output ./vectorAdd/vectorAdd $size > /dev/null
		real=`cat tmp_output`
		echo "$j $real" >> $f
	done
done
rm -f tmp_output
}

################################################################################
function bench_bwth(){
f=$1
make -C bandwidthTest clean all
rm -f $f

for i in $(seq 1 $2)
do
	echo "bwth $i / $2"
	for (( j=$3; j<=$4; j=j*2 ))
	do
		echo "    $j size"
		./bandwidthTest/bandwidthTest -mode=range --start=$j --end=$j --increment=1 > tmp_output
		h2d=`cat tmp_output | sed -n '10p' | awk '{print $2}'`
		d2h=`cat tmp_output | sed -n '15p' | awk '{print $2}'`
		echo "$j $h2d $d2h" >> $f
	done
	echo "##################################################################"
done
rm -f tmp_output
}

if [ $# -eq 0 ]; then
echo "$0 [virtual|native]"
fi

################################################################################
bench_bwth "bwth_${1}_data.txt" 10 1024 1073741824
bench_mmul "mmul_${1}_data.txt" 10 32 4096
bench_vadd "vadd_${1}_data.txt" 10 1 256

