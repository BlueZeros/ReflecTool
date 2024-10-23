DATA_PATH=../data
OUTPUT_PATH=../results/test

## run config
TASK_NAME=test
MEMORY_TYPE="task"

# domains=("mimic_iii" "ehrsql" "eicu")
domain="medcalc"
MODELS=("qwen2-7b" "qwen2-72b-int4")
MEMORY=${DATA_PATH}/memory/${MEMORY_TYPE}/train/${domain}/train-qwen2-72b-int4-few_shot_0-all

SEARCH="select" # should be select or refine
ACTIONS="all"
FEWSHOT=1
for MODEL in "${MODELS[@]}"; do
    sbatch -J eval_${domain} ../scripts/srun/base_run_clinical_w_memory.sh $DATA_PATH $OUTPUT_PATH $TASK_NAME $domain $MODEL $ACTIONS $MEMORY $MEMORY_TYPE $FEWSHOT ${SEARCH} & sleep 3
done

# MODEL=qwen2-7b

# ACTIONS="all"
# FEWSHOT=1
# for domain in "${domains[@]}"; do
#     MEMORY=${DATA_PATH}/memory/${MEMORY_TYPE}/train/${domain}/train-qwen2-72b-int4-few_shot_0-all
#     sbatch ../scripts/srun/base_run_clinical_w_memory.sh $DATA_PATH $OUTPUT_PATH $TASK_NAME $domain $MODEL $ACTIONS $MEMORY $MEMORY_TYPE $FEWSHOT & sleep 3
# done

# ACTIONS="all"
# FEWSHOT=0
# for domain in "${domains[@]}"; do
#     sbatch ../scripts/srun/base_run.sh $DATA_PATH $OUTPUT_PATH $TASK_PATH $domain $MODEL $ACTIONS $MEMORY $FEWSHOT & sleep 2
# done

