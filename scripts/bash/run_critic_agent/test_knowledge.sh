DATA_PATH=../data
OUTPUT_PATH=../results/test

TASK_PATH=test
MEMORY="./memory/medqa_memory"

domains=("medqa" "mmlu" "pubmedqa" "bioasq")
# MODEL=llama3.1-8b
MODEL=qwen2-7b
# MODEL=qwen1.5-32b-int4
# MODEL=llama3-70b-int4
# ACTIONS="base"
# FEWSHOT=1
# for domain in "${domains[@]}"; do
#     sbatch ../scripts/srun/base_run.sh $DATA_PATH $OUTPUT_PATH $TASK_PATH $domain $MODEL $ACTIONS $MEMORY $FEWSHOT &
# done

ACTIONS="all"
FEWSHOT=1
for domain in "${domains[@]}"; do
    sbatch ../scripts/srun/base_run_critic.sh $DATA_PATH $OUTPUT_PATH $TASK_PATH $domain $MODEL $ACTIONS $MEMORY $FEWSHOT & sleep 30
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

