#!/bin/bash

# Compile the C++ program
cmake ..
make

# Declare arrays of M and N values
declare -a Ms=(1 2 5 10 25 50 100 150 180 250 500 1000) # size
declare -a Ns=(1 2 5 10 25 50 100 250 500 1000 2000 3000 5000 7500 10000) # powers


# Check if the results file exists, if not, create and add the header
RESULTS_FILE="matrix_benchmark_results.csv"
if [ ! -f "$RESULTS_FILE" ]; then
    echo "Method, Matrix Size, Power,Time (milliseconds)" > "$RESULTS_FILE"
fi

# Run the program with each combination of M and N
for M in "${Ms[@]}"
do
    for N in "${Ns[@]}"
    do
        echo "Running matrixOps for M=${M}, N=${N}"
        ./Lab1CPP $M $N
    done
done

echo "All operations completed."

 cd ../../Lab2CPP-CUDA/build/
 pwd
./run.sh