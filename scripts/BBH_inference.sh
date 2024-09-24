base_dir="/path/to/Abstraction-of-Thought"


cd ${base_dir}/code

model_name="Llama-3-8B-AoT"
model_path="Abstraction-of-Thought/Llama-3-8B-AoT"
tokenizer_path=${model_path}

## Zero-Shot Setting
prompt_type=Zero_prompt
example_num=0

## Few-Shot Setting 
# prompt_type=AoT_prompt
# example_num=3

python bbh_runner.py --call_type 'vllm' --max_new_tokens 2048 --batch_size 16 --stop_word "<|im_end|>" \
--tokenizer_path ${tokenizer_path} --model_path ${model_path} --query_type "chat-add" \
--prompt_type ${prompt_type} --example_num ${example_num} \
--data_path "${base_dir}/data/BIG-Bench-Hard/bbh" \
--save_path "${base_dir}/exp/${model_name}.${prompt_type}.shot${example_num}.jsonl"


