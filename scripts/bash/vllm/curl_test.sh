curl http://SH-IDC1-10-140-0-173:10002/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
            "model": "/mnt/hwfile/medai/LLMModels/Model/Qwen2-7B-Instruct-lys",
            "prompt": "San Francisco is a",
            "max_tokens": 512,
            "temperature": 0
        }'