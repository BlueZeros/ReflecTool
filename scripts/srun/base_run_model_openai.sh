#!/bin/bash
#SBATCH -J eval_agent
#SBATCH --partition=medai_llm
#SBATCH -N1
#SBATCH --quotatype=auto
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=8
#SBATCH --ntasks-per-node=1    
#SBATCH --mem-per-cpu=8G  
#SBATCH --time=72:00:00
###SBATCH --kill-on-bad-exit=1

DATA_PATH="$1"
OUTPUT_PATH="$2"
TASK_NAME="$3"
TASK_SPLIT="$4"
MODEL="$5"

EXP_NAME=${MODEL}
mkdir -p ${OUTPUT_PATH}/${TASK_SPLIT}/${EXP_NAME}

srun --jobid $SLURM_JOBID python run_model.py \
    --data-path ${DATA_PATH} \
    --output-path ${OUTPUT_PATH} \
    --task-name ${TASK_NAME} \
    --test-split ${TASK_SPLIT} \
    --exp-name ${EXP_NAME} \
    --model ${MODEL} \
    --log-print \
    --prompt-debug \
    --resume

# # > ${OUTPUT_PATH}/${TASK_NAME}/${TASK_SPLIT}/${EXP_NAME}/infer.log &