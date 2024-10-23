#!/bin/bash
#SBATCH --partition=medai_llm
#SBATCH -N1
#SBATCH --quotatype=spot
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --ntasks-per-node=1    
#SBATCH --mem-per-cpu=8G  
#SBATCH --time=72:00:00
#SBATCH -x SH-IDC1-10-140-0-224,SH-IDC1-10-140-0-173
###SBATCH --kill-on-bad-exit=1

DATA_PATH="$1"
OUTPUT_PATH="$2"
TASK_PATH="$3"
TASK_SPLIT="$4"
MODEL="$5"
ACTIONS="$6"
MEMORY_PATH="$7"
FEWSHOT="$8"

EXP_NAME=${MODEL}-few_shot_${FEWSHOT}-${ACTIONS}
mkdir -p ${OUTPUT_PATH}/${TASK_PATH}-w_tool_optim/${TASK_SPLIT}/${EXP_NAME}
OUTPUT_PATH=${OUTPUT_PATH}/${TASK_PATH}-w_tool_optim

srun --jobid $SLURM_JOBID python run.py \
    --data-path ${DATA_PATH} \
    --output-path ${OUTPUT_PATH} \
    --task-name ${TASK_PATH} \
    --test-split ${TASK_SPLIT} \
    --exp-name ${EXP_NAME} \
    --model ${MODEL} \
    --actions ${ACTIONS} \
    --memory-path ${MEMORY_PATH} \
    --max-exec-steps 10 \
    --few-shot ${FEWSHOT} \
    --resume \
    --load-action-params /mnt/petrelfs/liaoyusheng/projects/ClinicalAgent/ClinicalAgent/results/train/mix_train/qwen2-72b-int4-few_shot_3-all-memory-fix/action_optim/step-500.json