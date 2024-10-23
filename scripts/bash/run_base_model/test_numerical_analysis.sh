DATA_PATH=../data
OUTPUT_PATH=../results

TASK_PATH=test

domains=("ehrsql")
# MODEL=llama3.1-8b
# MODELS=("llama3.1-8b" "qwen2-7b" "llama3-8b" "medllama3-8b" "huatuo-vision-7b" "qwen2-72b-int4" "llama3.1-70b-int4")
MODELS=("qwen2-7b") # "huatuo-vision-34b")

for domain in "${domains[@]}"; do
    for MODEL in "${MODELS[@]}"; do
        sbatch ../scripts/srun/base_run_model.sh $DATA_PATH $OUTPUT_PATH $TASK_PATH $domain $MODEL & sleep 1
    done
done

