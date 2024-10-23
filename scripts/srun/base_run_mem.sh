#!/bin/bash
#SBATCH -J eval_agent
#SBATCH --partition=medai_llm
#SBATCH -N1
#SBATCH --quotatype=spot
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --ntasks-per-node=1    
#SBATCH --mem-per-cpu=16G  
#SBATCH --time=72:00:00
#SBATCH -x SH-IDC1-10-140-0-224
###SBATCH --kill-on-bad-exit=1

DATA_PATH="$1"
OUTPUT_PATH="$2"
TASK_NAME="$3"
TASK_SPLIT="$4"
MODEL="$5"
ACTIONS="$6"
MEMORY="$7"
FEWSHOT="$8"

EXP_NAME=${MODEL}-few_shot_${FEWSHOT}-${ACTIONS}
MEMORY=${MEMORY}/${EXP_NAME}
mkdir -p ${OUTPUT_PATH}/${TASK_SPLIT}/${EXP_NAME}

srun --jobid $SLURM_JOBID python run.py \
    --data-path ${DATA_PATH} \
    --output-path ${OUTPUT_PATH} \
    --task-name ${TASK_NAME} \
    --test-split ${TASK_SPLIT} \
    --exp-name ${EXP_NAME} \
    --model ${MODEL} \
    --actions ${ACTIONS} \
    --memory-path ${MEMORY} \
    --max-exec-steps 10 \
    --few-shot ${FEWSHOT} \
    --memory-type task \
    --write-memory \
    --test-number 1000 \
    --log-print \
    --prompt-debug > ${OUTPUT_PATH}/${TASK_SPLIT}/${EXP_NAME}/infer.log

# # > ${OUTPUT_PATH}/${TASK_NAME}/${TASK_SPLIT}/${EXP_NAME}/infer.log &