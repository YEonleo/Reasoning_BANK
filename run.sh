# api_key 

python run_baseline.py \
  --model gpt-5-nano \
  --api_key "임의의 API 키" \
  --base_dir test \
  --tasks_path test/사전과제.json

python run_enhanced.py \
  --model gpt-5-nano \
  --api_key "임의의 API 키" \
  --base_dir test \
  --tasks_path test/사전과제.json \
  --bank_path memory/bank.json