DATA_PATH=../data
OUTPUT_PATH=../results/ablations

## run config
TASK_NAME=ablations
MEMORY_TYPE="standard"
ACTIONS="all"
FEWSHOT=1

# domains=("mimic_iii" "ehrsql" "eicu")
# domains=("mimic_iii" "eicu")
domains=("medhalt_rht")
MODEL="qwen2-72b-int4"

SEARCH="refine" # should be select or refine
# REFLECT_NUMS=("1" "2" "4" "8" "16" "32")
REFLECT_NUMS=("2")

for domain in "${domains[@]}"; do
    MEMORY=${DATA_PATH}/memory
    for REFLECT_NUM in "${REFLECT_NUMS[@]}"; do
        sbatch -J eval_${domain} -x SH-IDC1-10-140-1-159,SH-IDC1-10-140-0-228 ../scripts/srun/base_run_clinical_w_memory_reflect_num.sh $DATA_PATH $OUTPUT_PATH $TASK_NAME $domain $MODEL $ACTIONS $MEMORY $MEMORY_TYPE $FEWSHOT ${SEARCH} ${REFLECT_NUM} & sleep 3
    done
done


SEARCH="select" # should be select or refine
# REFLECT_NUMS=("1" "2" "4" "8" "16" "32")
REFLECT_NUMS=("2")

for domain in "${domains[@]}"; do
    MEMORY=${DATA_PATH}/memory
    for REFLECT_NUM in "${REFLECT_NUMS[@]}"; do
        sbatch -J eval_${domain} -x SH-IDC1-10-140-1-159,SH-IDC1-10-140-0-228 ../scripts/srun/base_run_clinical_w_memory_reflect_num.sh $DATA_PATH $OUTPUT_PATH $TASK_NAME $domain $MODEL $ACTIONS $MEMORY $MEMORY_TYPE $FEWSHOT ${SEARCH} ${REFLECT_NUM} & sleep 3
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

