#!/bin/bash

p=./aio

function bw()
{
	rm -rf host_bw1.txt host_bw2.txt
#for((i=1024; i<1073741824; i=i*2 ))
	for((i=10; i<=20; i=i*2 ))
	do
		echo "bw $i" 1>&2
		$p bw $i >> host_bw1.txt & P1=$!
		$p bw $i >> host_bw2.txt & P2=$!
		$p bw $i >> host_bw3.txt & P3=$!
		$p bw $i >> host_bw4.txt & P4=$!
		wait $P1
		wait $P2
		wait $P3
		wait $P4
	done
}

bw
#mmul
#vadd


