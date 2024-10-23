#!/bin/bash
#SBATCH --partition=medai_llm
#SBATCH -N1
#SBATCH --quotatype=spot
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --ntasks-per-node=1    
#SBATCH --mem-per-cpu=8G  
#SBATCH --time=72:00:00
#SBATCH -x SH-IDC1-10-140-0-224,SH-IDC1-10-140-0-229,SH-IDC1-10-140-0-170,SH-IDC1-10-140-0-226
###SBATCH --kill-on-bad-exit=1

DATA_PATH="$1"
OUTPUT_PATH="$2"
TASK_NAME="$3"
TASK_SPLIT="$4"
MODEL="$5"
ACTIONS="$6"
MEMORY="$7"
MEMORY_TYPE="$8"
FEWSHOT="$9"
SEARCH="${10}"
REFLECT_NUM="${11}"


EXP_NAME=clinical_${SEARCH}_${REFLECT_NUM}-${MODEL}-few_shot_${FEWSHOT}-${ACTIONS}-${MEMORY_TYPE}_memory
mkdir -p ${OUTPUT_PATH}/${TASK_SPLIT}/${EXP_NAME}

echo $EXP_NAME
echo $TASK_SPLIT
echo $MEMORY_TYPE

srun --jobid $SLURM_JOBID python run.py \
    --data-path ${DATA_PATH} \
    --output-path ${OUTPUT_PATH} \
    --task-name ${TASK_NAME} \
    --test-split ${TASK_SPLIT} \
    --exp-name ${EXP_NAME} \
    --agent "clinical" \
    --model ${MODEL} \
    --actions ${ACTIONS} \
    --action-guide-path "../results/train/${TASK_SPLIT}/train-qwen2-72b-int4-few_shot_0-mm/action_optim/step-200.json" \
    --memory-path ${MEMORY} \
    --memory-type ${MEMORY_TYPE} \
    --clinical-reflect-num ${REFLECT_NUM} \
    --max-exec-steps 15 \
    --force-action \
    --action-search ${SEARCH} \
    --few-shot ${FEWSHOT} \
    --log-print \
    --prompt-debug \
    --resume