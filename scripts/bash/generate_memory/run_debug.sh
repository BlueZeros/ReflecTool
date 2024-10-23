DATA_PATH=../data
OUTPUT_PATH=../results/trian
MEMORY=../data/memory

TASK_NAME=train
MEMORY_TYPE=task


domain="emrqa"
MODEL=qwen2-72b-int4

ACTIONS="all"
FEWSHOT=0

EXP_NAME=${MODEL}-few_shot_${FEWSHOT}-${ACTIONS}
MEMORY=${MEMORY}/${EXP_NAME}
mkdir -p ${OUTPUT_PATH}/${domain}/${EXP_NAME}

srun -p medai_llm --quotatype=auto --gres=gpu:1 python run.py \
    --data-path ${DATA_PATH} \
    --output-path ${OUTPUT_PATH} \
    --task-name ${TASK_NAME} \
    --test-split ${domain} \
    --exp-name ${EXP_NAME} \
    --model ${MODEL} \
    --actions ${ACTIONS} \
    --memory-path ${MEMORY} \
    --max-exec-steps 10 \
    --few-shot ${FEWSHOT} \
    --write-memory \
    --memory-type task \
    --log-print \
    --test-number 1000 \
    --prompt-debug