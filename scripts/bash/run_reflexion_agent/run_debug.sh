DATA_PATH=../data
OUTPUT_PATH=../results/test
MEMORY_PATH=../data/memory

TASK_PATH=test

# ("medmentions" "emrqa" "longhealthqa" "medhalt_rht" "longhalt")
domain="longhalt"
MODEL=qwen2-7b
# URL="http://SH-IDC1-10-140-0-173:10003"


ACTIONS="all"
FEWSHOT=1

EXP_NAME=reflexion_${MODEL}-few_shot_${FEWSHOT}-${ACTIONS}
mkdir -p ${OUTPUT_PATH}/${domain}/${EXP_NAME}

srun -p medai_llm --quotatype=auto --gres=gpu:1 python run.py \
    --data-path ${DATA_PATH} \
    --output-path ${OUTPUT_PATH} \
    --task-name ${TASK_PATH} \
    --test-split ${domain} \
    --exp-name ${EXP_NAME} \
    --agent "reflexion" \
    --model ${MODEL} \
    --actions ${ACTIONS} \
    --memory-path ${MEMORY_PATH} \
    --memory-type "reflexion_standard" \
    --max-exec-steps 15 \
    --few-shot ${FEWSHOT} \
    --resume \
    --test-number 1 &