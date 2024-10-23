DATA_PATH=./ClinicalAgentBench
TASK=test

OUTPUT_PATH=./results/${TASK}

DATASET=medqa
MODEL=qwen2-7b
ACTIONS="all_wo_mm" # tool type. note that `all` and `mm` actions should use 2 gpu.
FEWSHOT=1 # num of few-shot demonstration
MEMORY_PATH=${DATA_PATH}/memory/task/long_term_memory/${DATASET}

EXP_NAME=reflectool_refine-${MODEL}-few_shot_${FEWSHOT}-${ACTIONS}-task_memory
mkdir -p ${OUTPUT_PATH}/${TASK}/${DATASET}/${EXP_NAME}

python run.py \
    --data-path ${DATA_PATH} \
    --output-path ${OUTPUT_PATH} \
    --task-name ${TASK} \
    --test-split ${DATASET} \
    --exp-name ${EXP_NAME} \
    --agent "reflectool" \
    --model ${MODEL} \
    --actions ${ACTIONS} \
    --action-guide-path ${DATA_PATH}/memory/task/tool_experience/${DATASET}.json \
    --memory-path ${MEMORY_PATH} \
    --force-action \
    --action-search "refine" \
    --few-shot ${FEWSHOT} \
    --log-print \
    --test-number -1 \
    --prompt-debug \
    --resume