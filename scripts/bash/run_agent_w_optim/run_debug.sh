DATA_PATH=../data
OUTPUT_PATH=../results
MEMORY_PATH=../data/memory
TASK_PATH=test

domain="medqa"
MODEL=qwen2-72b-int4


ACTIONS="all"
FEWSHOT=1

EXP_NAME=${MODEL}-few_shot_${FEWSHOT}-${ACTIONS}
mkdir -p ${OUTPUT_PATH}/${TASK_PATH}-w_tool_optim/${domain}/${EXP_NAME}
OUTPUT_PATH=${OUTPUT_PATH}/${TASK_PATH}-w_tool_optim

srun -p medai_llm --quotatype=auto --gres=gpu:1 python run.py \
    --data-path ${DATA_PATH} \
    --output-path ${OUTPUT_PATH} \
    --task-name ${TASK_PATH} \
    --test-split ${domain} \
    --exp-name ${EXP_NAME} \
    --model ${MODEL} \
    --actions ${ACTIONS} \
    --memory-path ${MEMORY_PATH} \
    --max-exec-steps 10 \
    --few-shot ${FEWSHOT} \
    --log-print \
    --resume \
    --load-action-params /mnt/petrelfs/liaoyusheng/projects/ClinicalAgent/ClinicalAgent/results/train/mix_train/qwen2-72b-int4-few_shot_3-all-memory-fix/action_optim/step-500.json \
    --test-number -1 \
    --prompt-debug