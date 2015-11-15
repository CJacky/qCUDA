#!/bin/bash

make memcpy

rm -rf memcpy_data

for i in $(seq 1 10)
do
	echo "$i run"

	for (( j=1; j<=512; j=j*2 ))
	do
		echo "    ${j}KB"
		./memcpy ${j}k >> memcpy_data
		sleep 2
	done

	for (( j=1; j<=1024; j=j*2 ))
	do
		echo "    ${j}MB"
		./memcpy ${j}m >> memcpy_data
		sleep 2
	done

done
