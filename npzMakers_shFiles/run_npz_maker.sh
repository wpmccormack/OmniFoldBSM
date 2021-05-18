#!/bin/bash

percs=("1" "2.5" "5" "10" "20" "40")
#masses=("500" "1000" "2000" "4000" "8000" "16000")

#percs=("10")
masses=("4000" "2000" "1000" "500")

for i in "${percs[@]}"
do
    for j in "${masses[@]}"
    do
	python3.6 omni_npz_maker_cmd.py $j $i
    done
done
