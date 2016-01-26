#!/bin/bash

LOG=/tmp/libcudart_dl_out

function bw()
{
	echo "bw"
	for i in $(seq 1 10)
	do
		for(( j=1024; j<=1073741824; j=j*2 ))
		do
			echo "bw $i $j" 1>&2
			./bw $j
		done
	done
}

function mmul()
{
	echo ""
	echo "mmul"
	for i in $(seq 1 10)
	do
		for(( j=32; j<=4096; j=j*2 ))
		do
			echo "mmul $i $j" 1>&2
			./mmul $j
		done
	done
}

function vadd()
{
	echo ""
	echo "vadd"
	for i in $(seq 1 10)
	do
		for (( j=1048576; j<=268435456; j=j*2 ))
		do
			echo "vadd $i $j" 1>&2
			./vadd $j
		done
	done
}

./bw 10 > /dev/null

bw
mmul
vadd
