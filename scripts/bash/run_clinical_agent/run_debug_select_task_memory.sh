DATA_PATH=../data
OUTPUT_PATH=../results/test

TASK_PATH=test

domain="medqa"
MODEL=qwen2-7b
# URL="http://SH-IDC1-10-140-0-173:10002"

# MODEL=qwen2-7b
# URL="http://SH-IDC1-10-140-0-173:10002"


ACTIONS="all"
FEWSHOT=1
MEMORY_PATH=../data/memory/task/train/${domain}/train-qwen2-72b-int4-few_shot_0-all

EXP_NAME=clinical_select-${MODEL}-few_shot_${FEWSHOT}-${ACTIONS}-task_memory
mkdir -p ${OUTPUT_PATH}/${TASK_PATH}/${domain}/${EXP_NAME}

# --vllm-serve-url  \

srun -p medai_llm --quotatype=auto --gres=gpu:1 -x SH-IDC1-10-140-0-229,SH-IDC1-10-140-0-170,SH-IDC1-10-140-0-226 python run.py \
    --data-path ${DATA_PATH} \
    --output-path ${OUTPUT_PATH} \
    --task-name ${TASK_PATH} \
    --test-split ${domain} \
    --exp-name ${EXP_NAME} \
    --agent "clinical" \
    --model ${MODEL} \
    --actions ${ACTIONS} \
    --action-guide-path "../results/train/${domain}/train-qwen2-72b-int4-few_shot_0-all/action_optim/step-290.json" \
    --memory-path ${MEMORY_PATH} \
    --memory-type "task" \
    --max-exec-steps 15 \
    --force-action \
    --action-search "select" \
    --few-shot ${FEWSHOT} \
    --log-print \
    --test-number 1 \
    --prompt-debug \
    --resume