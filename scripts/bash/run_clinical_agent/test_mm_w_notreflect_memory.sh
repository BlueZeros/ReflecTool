DATA_PATH=../data
OUTPUT_PATH=../results/ablations

## run config
TASK_NAME=ablations
MEMORY_TYPE="task"

# domains=("mimic_iii" "ehrsql" "eicu")
# domains=("mimic_iii" "eicu")
domains=("vqarad")
MODELS=("qwen2-72b-int4")

# SEARCH="select" # should be select or refine
ACTIONS="mm"
FEWSHOT=1
# for domain in "${domains[@]}"; do
#     MEMORY=${DATA_PATH}/memory/${MEMORY_TYPE}/train/${domain}/train-qwen2-72b-int4-few_shot_0-mm
#     for MODEL in "${MODELS[@]}"; do
#         sbatch -J eval_${domain} -x SH-IDC1-10-140-1-159 --gres=gpu:2 ../scripts/srun/base_run_clinical_w_notreflect_mm_memory.sh $DATA_PATH $OUTPUT_PATH $TASK_NAME $domain $MODEL $ACTIONS $MEMORY $MEMORY_TYPE $FEWSHOT ${SEARCH} & sleep 3
#     done
# done

SEARCH="select" # should be select or refine
for domain in "${domains[@]}"; do
    MEMORY=${DATA_PATH}/memory/${MEMORY_TYPE}/train/${domain}/train-qwen2-72b-int4-few_shot_0-mm
    for MODEL in "${MODELS[@]}"; do
        sbatch -J eval_${domain} -x SH-IDC1-10-140-1-159 --gres=gpu:2 ../scripts/srun/base_run_clinical_w_notreflect_mm_memory.sh $DATA_PATH $OUTPUT_PATH $TASK_NAME $domain $MODEL $ACTIONS $MEMORY $MEMORY_TYPE $FEWSHOT ${SEARCH} & sleep 3
    done
done