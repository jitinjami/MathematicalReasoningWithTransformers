#!/bin/bash
#SBATCH --job-name=math
#SBATCH --time=47:59:59
#SBATCH --output=messages/math.out
#SBATCH --error=messages/math.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH -w icsnode11

USERNAME=$(id -un)
#---------------------------
# user parameters, worakble nodes = icsnode09,icsnode10,icsnode11,icsnode12,icsnode14
#----------------------------
WORKING_DIR="/home/jami/MathematicalReasoningWithTransformers" # directory containing the script to be run
#---------------------------
# issue the job
#---------------------------

source ~/.bashrc
conda activate ex
cd $WORKING_DIR
# run the experiment
python3 train.py