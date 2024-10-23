DATA_PATH=./ClinicalAgentBench
TASK=train

OUTPUT_PATH=./results/${TASK}
MEMORY_PATH=${DATA_PATH}/my_memory

DATASET=medqa
MODEL=qwen2-7b
ACTIONS="all_wo_mm" # tool type. note that `all` and `mm` actions should use 2 gpu.
FEWSHOT=0 # num of few-shot demonstration

EXP_NAME=train-${MODEL}-few_shot_${FEWSHOT}-${ACTIONS}
mkdir -p ${OUTPUT_PATH}/${DATASET}/${EXP_NAME}

python train.py \
    --data-path ${DATA_PATH} \
    --output-path ${OUTPUT_PATH} \
    --task-name ${TASK} \
    --test-split ${DATASET} \
    --exp-name ${EXP_NAME} \
    --model ${MODEL} \
    --actions ${ACTIONS} \
    --memory-path ${MEMORY_PATH} \
    --few-shot ${FEWSHOT} \
    --write-memory \
    --memory-type task \
    --log-print \
    --resume