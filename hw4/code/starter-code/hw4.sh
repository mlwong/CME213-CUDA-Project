#!/bin/sh
#SBATCH -p cme213
#SBATCH --gres=gpu:1

########### UPDATE THESE VARIABLES ###############
cipher_period=11
##################################################

module add shared
module add cuda65

# Clean up the directory
make clean
# Compile the program
make
# Encrypt text
./create_cipher mobydick.txt $cipher_period
# Decipher text
./solve_cipher cipher_text.txt