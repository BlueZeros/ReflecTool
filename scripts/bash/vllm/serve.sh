MODEL=/mnt/hwfile/medai/LLMModels/Model/Qwen2-7B-Instruct-lys
# MODEL=/mnt/hwfile/medai/LLMModels/Model/Qwen2-72B-Instruct-GPTQ-Int4

srun -p medai_llm --quotatype=auto --gres=gpu:1 vllm serve ${MODEL} \
    --port 10003 \
    --dtype auto 


