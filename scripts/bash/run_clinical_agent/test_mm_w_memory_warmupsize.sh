DATA_PATH=../data
OUTPUT_PATH=../results/ablations

## run config
TASK_NAME=ablations
MEMORY_TYPE="task"

# domains=("mimic_iii" "ehrsql" "eicu")
# domains=("mimic_iii" "eicu")
domains=("vqarad")
MODEL="qwen2-72b-int4"
WARM_UP_SIZES=("0" "10" "50" "100")

SEARCH="select" # should be select or refine
ACTIONS="mm"
FEWSHOT=1
for domain in "${domains[@]}"; do
    MEMORY=${DATA_PATH}/memory/${MEMORY_TYPE}/train/${domain}/train-qwen2-72b-int4-few_shot_0-mm
    
    for WARM_UP_SIZE in "${WARM_UP_SIZES[@]}"; do
        sbatch -J eval_${domain} -x SH-IDC1-10-140-1-159 --gres=gpu:2 ../scripts/srun/base_run_clinical_w_mm_memory_warmupsize.sh $DATA_PATH $OUTPUT_PATH $TASK_NAME $domain $MODEL $ACTIONS $MEMORY $MEMORY_TYPE $FEWSHOT ${SEARCH} ${WARM_UP_SIZE} & sleep 3
    done
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

