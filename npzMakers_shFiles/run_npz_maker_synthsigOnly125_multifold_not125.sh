#!/bin/bash

percs=(".1" "1." "10.")
#masses=("500" "1000" "2000" "4000" "8000" "16000")

#percs=("10")
#masses=("500" "2000" "4000" "16000")
masses=("1" "4")

for i in "${percs[@]}"
do
    for j in "${masses[@]}"
    do
	python3.6 omni_npz_maker_synthsigOnly125_multifold_not125.py $j $i
    done
done
