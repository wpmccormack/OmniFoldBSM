#!/bin/bash

#percs=("1Perc" "2Perc" "5Perc" "10Perc" "20Perc" "40Perc")
#percs=("0Perc" "1Perc" "10Perc")
#percs=("0Perc")
percs=("0Perc" "10Perc")

#masses=("4000" "2000" "1000" "500")
masses=("0" "5")

for i in "${percs[@]}"
do
    for j in "${masses[@]}"
    do
	python3.6 Omni_NOsynthsig_pythia_cmd.py $i $j &> output_${i}_perc_${j}_mass_pythia_NOsynth.txt
    done
done
