DATA_PATH=../data
OUTPUT_PATH=../results

TASK_PATH=test
MEMORY="./memory/medqa_memory"

domains=("medvqa_halt")
# MODEL=llama3.1-8b
MODEL=huatuo-vision-34b

for domain in "${domains[@]}"; do
    sbatch ../scripts/srun/base_run_model.sh $DATA_PATH $OUTPUT_PATH $TASK_PATH $domain $MODEL & sleep 1
done

