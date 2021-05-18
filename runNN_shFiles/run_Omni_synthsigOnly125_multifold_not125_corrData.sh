#!/bin/bash

#percs=("1Perc" "2Perc" "5Perc" "10Perc" "20Perc" "40Perc")
#percs=("0Perc" "1Perc" "10Perc")
#percs=("0Perc")
percs=("0Perc" "1Perc" "10Perc")

#masses=("4000" "2000" "1000" "500")
masses=("1" "4")

for i in "${percs[@]}"
do
    for j in "${masses[@]}"
    do
	python3.6 Omni_synthsigOnly125_multifold_cmd_not125_corrData.py $i $j &> output_${i}_perc_${j}_mass_multifold_synthOnly125_not125_corrData.txt
    done
done
