#!/bin/bash

if [ "$#" -ne "1" ]; 
then
	echo "$0 [memcpy program]"
	exit
fi

make $1

#for i in $(seq 1 10)
#do
#	echo $i run

	for (( j=1; j<=512; j=j*2 ))
	do
		./$1 ${j}k
		sleep 2
	done

	for (( j=1; j<=1024; j=j*2 ))
	do
		./$1 ${j}m
		sleep 2
	done

#done
