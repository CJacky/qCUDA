#!/bin/bash

p=/home/cjacky/test/guest

bf1=guest_bw1.txt
bf2=guest_bw2.txt

mf1=guest_mmul1.txt
mf2=guest_mmul2.txt

vf1=guest_vadd1.txt
vf2=guest_vadd2.txt

function bw()
{
	for i in $(seq 1 10)
	do
		$p/bw $1 >> $p/$2
	done
}

function mmul()
{
	for i in $(seq 1 10)
	do
		$p/mmul $1 >> $p/$2
	done
}

function vadd()
{
	for i in $(seq 1 10)
	do
		$p/vadd $1 >> $p/$2
	done
}


case $1 in
	"bw")
	bw $2 $3
	;;

	"mmul")
	mmul $2 $3
	;;

	"vadd")
	vadd $2 $3
	;;

	*)
	ssh vm_ubuntu_14.04.3_0 "$p/bw 10" 
	ssh vm_ubuntu_14.04.3_1 "$p/bw 10" 
	
################################################################################

	ssh vm_ubuntu_14.04.3_0 "rm $p/$bf1" 
	ssh vm_ubuntu_14.04.3_1 "rm $p/$bf2" 
	for((i=1024; i<=1073741824; i=i*2 ))
	do
		echo "bw $i"
		ssh vm_ubuntu_14.04.3_0 "$p/go_multi2.sh bw $i $bf1" & P1=$!
		ssh vm_ubuntu_14.04.3_1 "$p/go_multi2.sh bw $i $bf2" & P2=$!
		wait $P1
		wait $P2
	done
	scp vm_ubuntu_14.04.3_0:$p/$bf1 .
	scp vm_ubuntu_14.04.3_1:$p/$bf2 .

################################################################################

	ssh vm_ubuntu_14.04.3_0 "rm $p/$mf1" 
	ssh vm_ubuntu_14.04.3_1 "rm $p/$mf2" 
	for((i=32; i<=4096; i=i*2 ))
	do
		echo "mmul $i"
		ssh vm_ubuntu_14.04.3_0 "$p/go_multi2.sh mmul $i $mf1" & P1=$!
		ssh vm_ubuntu_14.04.3_1 "$p/go_multi2.sh mmul $i $mf2" & P2=$!
		wait $P1
		wait $P2
	done
	scp vm_ubuntu_14.04.3_0:$p/$mf1 .
	scp vm_ubuntu_14.04.3_1:$p/$mf2 .

################################################################################

	ssh vm_ubuntu_14.04.3_0 "rm $p/$vf1" 
	ssh vm_ubuntu_14.04.3_1 "rm $p/$vf2" 
	for((i=1048576; i<=134217728; i=i*2 ))
	do
		echo "vadd $i"
		ssh vm_ubuntu_14.04.3_0 "$p/go_multi2.sh vadd $i $vf1" & P1=$!
		ssh vm_ubuntu_14.04.3_1 "$p/go_multi2.sh vadd $i $vf2" & P2=$!
		wait $P1
		wait $P2
	done
	scp vm_ubuntu_14.04.3_0:$p/$vf1 .
	scp vm_ubuntu_14.04.3_1:$p/$vf2 .

################################################################################
	;;
esac
