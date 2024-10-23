DATA_PATH=../data
OUTPUT_PATH=../results/trian
MEMORY=../data/memory

TASK_NAME=train
MEMORY_TYPE=task

domains=("medcalc" "ehrsql" "emrqa" "medmentions")
MODEL=qwen2-72b-int4


ACTIONS="all"
FEWSHOT=0
for domain in "${domains[@]}"; do
    sbatch ../scripts/srun/base_run_mem.sh $DATA_PATH $OUTPUT_PATH $TASK_NAME $domain $MODEL $ACTIONS $MEMORY $FEWSHOT & sleep 60
done

# ACTIONS="all"
# FEWSHOT=1
# for domain in "${domains[@]}"; do
#     sbatch ../scripts/srun/base_run.sh $DATA_PATH $OUTPUT_PATH $TASK_PATH $domain $MODEL $ACTIONS $MEMORY $FEWSHOT &
# done

# ACTIONS="all"
# FEWSHOT=0
# for domain in "${domains[@]}"; do
#     sbatch ../scripts/srun/base_run.sh $DATA_PATH $OUTPUT_PATH $TASK_PATH $domain $MODEL $ACTIONS $MEMORY $FEWSHOT &
# done

