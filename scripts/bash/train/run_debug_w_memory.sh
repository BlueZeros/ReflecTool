DATA_PATH=../data
OUTPUT_PATH=../results/train
MEMORY_PATH=../data/memory

TASK_PATH=train

domain="mimic_iii"
MODEL=qwen2-72b-int4


ACTIONS="all"
FEWSHOT=0

# EXP_NAME=train-${MODEL}-few_shot_${FEWSHOT}-${ACTIONS}
EXP_NAME=train-${MODEL}-few_shot_${FEWSHOT}-${ACTIONS}
mkdir -p ${OUTPUT_PATH}/${domain}/${EXP_NAME}

srun -p medai_llm --quotatype=auto --gres=gpu:1 -x SH-IDC1-10-140-0-224 python train.py \
    --data-path ${DATA_PATH} \
    --output-path ${OUTPUT_PATH} \
    --task-name ${TASK_PATH} \
    --test-split ${domain} \
    --exp-name ${EXP_NAME} \
    --model ${MODEL} \
    --actions ${ACTIONS} \
    --memory-path ${MEMORY_PATH} \
    --max-exec-steps 15 \
    --few-shot ${FEWSHOT} \
    --write-memory \
    --memory-type task \
    --log-print \
    --resume \
    --test-number -1