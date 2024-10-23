DATA_PATH=../data
OUTPUT_PATH=../results/train

TASK_PATH=train
MEMORY="../data/memory"

domains=("medhalt_rht")
ACTIONS="all"
FEWSHOT=0

MODEL=qwen2-72b-int4
for domain in "${domains[@]}"; do
    sbatch -J train_${domain} -x SH-IDC1-10-140-0-229  ../scripts/srun/base_run_train_w_memory.sh $DATA_PATH $OUTPUT_PATH $TASK_PATH $domain $MODEL $ACTIONS $MEMORY $FEWSHOT & sleep 3
done

# domains=("pubmedqa" "medqa" "mmlu" "bioasq" "medmentions" "emrqa" "longhealthqa" "medhalt_rht" "longhalt")
# MODEL=qwen2-72b-int4
# for domain in "${domains[@]}"; do
#     sbatch -J eval_${domain} -x SH-IDC1-10-140-0-225 ../scripts/srun/base_run_reflexion.sh $DATA_PATH $OUTPUT_PATH $TASK_PATH $domain $MODEL $ACTIONS $MEMORY $FEWSHOT & sleep 3
# done

# -x SH-IDC1-10-140-0-177,SH-IDC1-10-140-0-229,SH-IDC1-10-140-0-225

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

