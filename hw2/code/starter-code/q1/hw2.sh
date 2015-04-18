#!/bin/sh
#SBATCH -p cme213
#SBATCH --gres=gpu:1

module add shared
module add cuda65

########### UPDATE THIS VARIABLES ###############
num_copies=0
##################################################

# Clean up the directory
make clean
# Compile the program
make
# Run sums
./main $num_copies