DATA_PATH=../data
OUTPUT_PATH=../results/test
TASK_PATH=test

# MODELS=("qwen2-7b" "qwen2-72b-int4" "llama3.1-8b" "llama3.1-70b-int4")
MODELS=("gpt-4o-mini")
# MODELS=("minicpm-v-2.6" "huatuo-vision-7b" "huatuo-vision-34b" "internvl-chat-v1.5")
# MODELS=("llama3.1-8b" "llama3-8b")

domain="omnimedqa"
for MODEL in "${MODELS[@]}"; do
    sbatch ../scripts/srun/base_run_model.sh $DATA_PATH $OUTPUT_PATH $TASK_PATH $domain $MODEL & sleep 3
done

# domain="mimic_iii"
# for MODEL in "${MODELS[@]}"; do
#     sbatch ../scripts/srun/base_run_model.sh $DATA_PATH $OUTPUT_PATH $TASK_PATH $domain $MODEL & sleep 60
# done

