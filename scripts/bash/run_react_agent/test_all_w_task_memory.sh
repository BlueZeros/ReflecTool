## root config
DATA_PATH=../data
OUTPUT_PATH=../results

## run config
TASK_NAME=test
MEMORY=${DATA_PATH}/memory/
MOMORY_TYPE="task_standard"

domains=("ehrsql" "mimic_iii" "eicu")
MODEL=qwen2-72b-int4

ACTIONS="all"
FEWSHOT=1
for domain in "${domains[@]}"; do
    sbatch -x SH-IDC1-10-140-0-177 ../scripts/srun/base_run.sh $DATA_PATH $OUTPUT_PATH $TASK_NAME $domain $MODEL $ACTIONS $MEMORY $MOMORY_TYPE $FEWSHOT & sleep 10
done

# ACTIONS="all"
# FEWSHOT=0
# for domain in "${domains[@]}"; do
#     sbatch ../scripts/srun/base_run.sh $DATA_PATH $OUTPUT_PATH $TASK_PATH $domain $MODEL $ACTIONS $MEMORY $FEWSHOT & sleep 2
# done

