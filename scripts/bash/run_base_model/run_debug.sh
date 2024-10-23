DATA_PATH=../data
OUTPUT_PATH=../results/test

TASK_PATH=test
MEMORY="../data/memory"

domain="eicu"
MODEL=huatuo-vision-7b

EXP_NAME=${MODEL}
mkdir -p ${OUTPUT_PATH}/${domain}/${EXP_NAME}

srun -p medai_llm --quotatype=auto --gres=gpu:1 python run_model.py \
    --data-path ${DATA_PATH} \
    --output-path ${OUTPUT_PATH} \
    --task-name ${TASK_PATH} \
    --test-split ${domain} \
    --exp-name ${EXP_NAME} \
    --model ${MODEL} \
    --log-print \
    --test-number -1 \
    --prompt-debug