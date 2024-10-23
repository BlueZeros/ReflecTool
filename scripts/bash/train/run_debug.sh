DATA_PATH=../data
OUTPUT_PATH=../results/train
MEMORY_PATH=../data/memory

TASK_PATH=train

domain="medqa"
MODEL=qwen2-72b-int4
URL="http://SH-IDC1-10-140-0-173:10003"
# MODEL=qwen2-7b
# URL="http://SH-IDC1-10-140-0-223:10003"


ACTIONS="all"
FEWSHOT=0

EXP_NAME=train-${MODEL}-few_shot_${FEWSHOT}-${ACTIONS}
mkdir -p ${OUTPUT_PATH}/${domain}/${EXP_NAME}

srun -p medai_llm --quotatype=auto --gres=gpu:0 -x SH-IDC1-10-140-0-224 python train.py \
    --data-path ${DATA_PATH} \
    --output-path ${OUTPUT_PATH} \
    --task-name ${TASK_PATH} \
    --test-split ${domain} \
    --exp-name ${EXP_NAME} \
    --model ${MODEL} \
    --vllm-serve \
    --vllm-serve-url ${URL} \
    --actions ${ACTIONS} \
    --memory-path ${MEMORY_PATH} \
    --max-exec-steps 15 \
    --few-shot ${FEWSHOT} \
    --log-print \
    --resume \
    --test-number -1