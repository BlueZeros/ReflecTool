DATA_PATH=../data
OUTPUT_PATH=../results/test

TASK_PATH=test
MEMORY="./memory/medqa_memory"

# domains=("medqa" "mmlu" "pubmedqa" "bioasq" "medhalt_rht" "medcalc" "slake" "omnimedqa" "vqarad" "ehrsql") 
# domains=("slake" "omnimedqa" "vqarad")
# domains=("medmentions" "mimic_iii" "ehr_halt" "longhalt")
domain=medmentions
MODELS=(qwen2-7b qwen2-72b-int4)


ACTIONS="all"
FEWSHOT=1
for MODEL in "${MODELS[@]}"; do
    sbatch ../scripts/srun/base_run.sh $DATA_PATH $OUTPUT_PATH $TASK_PATH $domain $MODEL $ACTIONS $MEMORY $FEWSHOT & sleep 10
done

# ACTIONS="all"
# FEWSHOT=0
# for domain in "${domains[@]}"; do
#     sbatch ../scripts/srun/base_run.sh $DATA_PATH $OUTPUT_PATH $TASK_PATH $domain $MODEL $ACTIONS $MEMORY $FEWSHOT & sleep 2
# done

