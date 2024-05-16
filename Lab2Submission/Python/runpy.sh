#!/bin/bash

PYTHON_SCRIPT="Lab1PyWork.py"
declare -a Ms=(1 2 5 10 25 50 100 150 180 250 500 1000) # max threads/size MxM

declare -a Ns=(1 2 5 10 25 50 100 250 500 1000 2000 3000 5000 7500 10000) # power

RESULTS_FILE="matrix_benchmark_results_python3.csv"
if [ ! -f "$RESULTS_FILE" ] || [ ! -s "$RESULTS_FILE" ]; then
    echo "Method, Matrix Size, Power, Time (milliseconds)" > "$RESULTS_FILE"
fi

for M in "${Ms[@]}"
do
    for N in "${Ns[@]}"
    do
        echo "Running ${PYTHON_SCRIPT} for M=${M}, N=${N}"
        python3 ${PYTHON_SCRIPT} $M $N | grep "Time" | while read -r line; do
            TIME=$(echo "$line" | awk '{print $5}')
            echo "Python Array, $M, $N, $TIME" >> "$RESULTS_FILE"
        done
    done
done

echo "All operations completed."
