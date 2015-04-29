#!/bin/sh
#SBATCH -p cme213
#SBATCH --gres=gpu:1

module add shared
module add cuda65

make clean
# Compile the program
make
./main -gsb