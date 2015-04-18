#!/bin/sh
#SBATCH -p cme213
#SBATCH --gres=gpu:1

module add shared
module add cuda65

# Clean up the directory
make clean
# Compile the program
make
# Run sums
./main