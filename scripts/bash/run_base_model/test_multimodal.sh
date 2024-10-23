DATA_PATH=../data
OUTPUT_PATH=../results

TASK_PATH=test

domains=("omnimedqa")
# MODEL=llama3.1-8b
MODELS=("huatuo-vision-34b")

for domain in "${domains[@]}"; do
    for MODEL in "${MODELS[@]}"; do
        sbatch ../scripts/srun/base_run_model.sh $DATA_PATH $OUTPUT_PATH $TASK_PATH $domain $MODEL & sleep 1
    done
done

