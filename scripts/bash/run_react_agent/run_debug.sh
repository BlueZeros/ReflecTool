DATA_PATH=../data
OUTPUT_PATH=../results/train
MEMORY_PATH=../data/memory

TASK_PATH=train

domain="ehrsql"
MODEL=qwen2-72b-int4
# URL="http://SH-IDC1-10-140-0-173:10002"

# MODEL=qwen2-7b
# URL="http://SH-IDC1-10-140-0-173:10002"


ACTIONS="all"
FEWSHOT=0

EXP_NAME=${MODEL}-few_shot_${FEWSHOT}-${ACTIONS}
mkdir -p ${OUTPUT_PATH}/${TASK_PATH}/${domain}/${EXP_NAME}

# --vllm-serve-url  \

srun -p medai_llm --quotatype=spot --gres=gpu:1 -x SH-IDC1-10-140-0-224 python run.py \
    --data-path ${DATA_PATH} \
    --output-path ${OUTPUT_PATH} \
    --task-name ${TASK_PATH} \
    --test-split ${domain} \
    --exp-name ${EXP_NAME} \
    --model ${MODEL} \
    --actions ${ACTIONS} \
    --memory-path ${MEMORY_PATH} \
    --max-exec-steps 15 \
    --force-action \
    --few-shot ${FEWSHOT} \
    --log-print \
    --test-number 10 \
    --prompt-debug \
    --resume