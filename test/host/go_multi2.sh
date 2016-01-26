#!/bin/bash

p=./aio

function bw()
{
#	for((i=10; i<=20; i=i*2 ))
	rm -rf host_bw2.txt 
	for((i=1024; i<=1073741824; i=i*2 ))
	do
		echo "bw $i" 1>&2
		$p bw $i >> host_bw2.txt & P1=$!
		$p bw $i >> host_bw2.txt & P2=$!
		wait $P1
		wait $P2
	done
}

function mmul()
{
	rm -rf host_mmul2.txt
	for((i=32; i<=4096; i=i*2 ))
	do
		echo "mmul $i" 1>&2
		$p mmul $i >> host_mmul2.txt & P1=$!
		$p mmul $i >> host_mmul2.txt & P2=$!
		wait $P1
		wait $P2
	done
}

function vadd()
{
	rm -rf host_vadd2.txt
	for((i=1048576; i<=134217728; i=i*2 ))
	do
		echo "vadd $i" 1>&2
		$p vadd $i >> host_vadd2.txt & P1=$!
		$p vadd $i >> host_vadd2.txt & P2=$!
		wait $P1
		wait $P2
	done
}

bw
mmul
vadd

