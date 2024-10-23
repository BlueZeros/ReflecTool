DATA_PATH=../data
OUTPUT_PATH=../results/test

TASK_PATH=test
MEMORY="../data/memory"

# domains=("ehrsql" "mimic_iii" "eicu" "ehr_halt")
domains=("omnimedqa")
ACTIONS="mm"
FEWSHOT=1

# MODEL=qwen2-72b-int4
# for domain in "${domains[@]}"; do
#     sbatch -J eval_${domain} -x SH-IDC1-10-140-0-177 ../scripts/srun/base_run_critic.sh $DATA_PATH $OUTPUT_PATH $TASK_PATH $domain $MODEL $ACTIONS $MEMORY $FEWSHOT & sleep 30
# done

MODEL=qwen2-72b-int4
for domain in "${domains[@]}"; do
    sbatch -J eval_${domain} --gres=gpu:2 -x SH-IDC1-10-140-0-229 ../scripts/srun/base_run_critic.sh $DATA_PATH $OUTPUT_PATH $TASK_PATH $domain $MODEL $ACTIONS $MEMORY $FEWSHOT & sleep 3
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

