DATA_PATH=../data
OUTPUT_PATH=../results/test
MEMORY_PATH=../data/memory

TASK_PATH=test

domain="slake"
MODEL=qwen2-72b-int4


ACTIONS="mm"
FEWSHOT=1

EXP_NAME=critic_${MODEL}-few_shot_${FEWSHOT}-${ACTIONS}
mkdir -p ${OUTPUT_PATH}/${domain}/${EXP_NAME}

srun -p medai_llm --quotatype=auto --gres=gpu:2 -x SH-IDC1-10-140-0-229 python run.py \
    --data-path ${DATA_PATH} \
    --output-path ${OUTPUT_PATH} \
    --task-name ${TASK_PATH} \
    --test-split ${domain} \
    --exp-name ${EXP_NAME} \
    --agent "critic" \
    --model ${MODEL} \
    --actions ${ACTIONS} \
    --memory-path ${MEMORY_PATH} \
    --memory-type "critic_standard" \
    --max-exec-steps 15 \
    --few-shot ${FEWSHOT} \
    --resume \
    --log-print \
    --test-number -1 \
    --prompt-debug