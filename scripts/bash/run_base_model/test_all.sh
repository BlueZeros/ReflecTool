DATA_PATH=../data
OUTPUT_PATH=../results/test

TASK_PATH=test

domains=("medqa" "mmlu" "pubmedqa" "bioasq" "medhalt_rht" "medcalc" "slake" "omnimedqa" "vqarad" "ehrsql") 
# domains=("medmentions")
MODEL=huatuo
# MODEL=medllama3-8b

for domain in "${domains[@]}"; do
    sbatch ../scripts/srun/base_run_model.sh $DATA_PATH $OUTPUT_PATH $TASK_PATH $domain $MODEL & sleep 2
done

