DATA_PATH=../data
OUTPUT_PATH=../results/test
TASK_PATH=test

domain="omnimedqa"

# MODELS=("qwen2-7b" "qwen2-72b-int4" "medllama3-8b" "llama3-8b" "llama3.1-8b" "llama3.1-70b-int4")
MODELS=("gpt-4o-mini")
# MODELS=("gpt-3.5-turbo")
# MODEL=medllama3-8b

for MODEL in "${MODELS[@]}"; do
    sbatch ../scripts/srun/base_run_model_openai.sh $DATA_PATH $OUTPUT_PATH $TASK_PATH $domain $MODEL & sleep 3
done

