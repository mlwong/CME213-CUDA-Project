#!/usr/bin/env bash

#SBATCH -p defq

######### ADD YOUR EXECUTION SCRIPT HERE #########
# Set the number of threads
export OMP_NUM_THREADS=4
# Clean up the directory
make clean
# Compile the program
make
# Run
./main