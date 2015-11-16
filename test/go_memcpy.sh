#!/bin/bash

f=memcpy_data_host.txt

make memcpy

rm -rf $f

for i in $(seq 1 10)
do
	echo "$i run"

	for (( j=1; j<=512; j=j*2 ))
	do
		echo "    ${j}KB"
		sp=`./memcpy ${j}k`
		echo "${j}K $sp" >> $f
	done

	for (( j=1; j<=512; j=j*2 ))
	do
		echo "    ${j}MB"
		sp=`./memcpy ${j}m`
		echo "${j}M $sp" >> $f
	done

	j=1
	echo "    ${j}GB"
	sp=`./memcpy ${j}g`
	echo "${j}G $sp" >> $f

done
