DATA_PATH=../data
OUTPUT_PATH=../results/test
MEMORY_PATH=../data/memory

TASK_PATH=test

domain="medqa"
MODEL=qwen2-7b
URL="http://SH-IDC1-10-140-0-176:10003"

# MODEL=qwen2-7b
# URL="http://SH-IDC1-10-140-0-173:10002"

ACTIONS="all"
FEWSHOT=1

EXP_NAME=reflectool-${MODEL}-few_shot_${FEWSHOT}-${ACTIONS}-task_standard_memory-test
mkdir -p ${OUTPUT_PATH}/${TASK_PATH}/${domain}/${EXP_NAME}

# --vllm-serve-url  \

srun -p medai_llm --quotatype=spot --gres=gpu:0 -x SH-IDC1-10-140-0-229 python run.py \
    --data-path ${DATA_PATH} \
    --output-path ${OUTPUT_PATH} \
    --task-name ${TASK_PATH} \
    --test-split ${domain} \
    --exp-name ${EXP_NAME} \
    --agent "reflectool" \
    --model ${MODEL} \
    --vllm-serve \
    --vllm-serve-url ${URL} \
    --actions ${ACTIONS} \
    --memory-path ${MEMORY_PATH} \
    --memory-type "task_standard" \
    --max-exec-steps 15 \
    --force-action \
    --action-search "refine" \
    --few-shot ${FEWSHOT} \
    --log-print \
    --test-number -1 \
    --prompt-debug \
    --resume