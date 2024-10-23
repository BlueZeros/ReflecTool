DATA_PATH=../data
OUTPUT_PATH=../results

TASK_NAME=test
MEMORY="../data/memory/"

# domains=("mmlu" "bioasq" "pubmedqa" "medcalc" "ehrsql" "medmentions" "emrqa") 
domains=("pubmedqa")
MODEL=qwen2-72b-int4


ACTIONS="all"
FEWSHOT=1
for domain in "${domains[@]}"; do
    sbatch -J "eval_${domain}" ../scripts/srun/base_run_w_optim.sh $DATA_PATH $OUTPUT_PATH $TASK_NAME $domain $MODEL $ACTIONS $MEMORY $FEWSHOT & sleep 10
done

# ACTIONS="all"
# FEWSHOT=0
# for domain in "${domains[@]}"; do
#     sbatch ../scripts/srun/base_run.sh $DATA_PATH $OUTPUT_PATH $TASK_PATH $domain $MODEL $ACTIONS $MEMORY $FEWSHOT & sleep 2
# done

