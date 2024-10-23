DATA_PATH=../data
OUTPUT_PATH=../results

TASK_PATH=test
MEMORY="./memory/medqa_memory"

domains=("mmlu" "medqa" "pubmedqa" "bioasq")
# MODEL=llama3.1-8b
MODEL=meditron-7b

for domain in "${domains[@]}"; do
    sbatch ../scripts/srun/base_run_model.sh $DATA_PATH $OUTPUT_PATH $TASK_PATH $domain $MODEL & sleep 1
done

