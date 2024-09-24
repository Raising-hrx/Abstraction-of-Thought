import json
import re
from tqdm import tqdm
import numpy as np
import argparse
import os
import sys

from BBH_Master import BBH_Master

def chunk(datas, batch_size):
    for i in range(0, len(datas), batch_size):
        yield datas[i:i+batch_size]

def get_params():
    parser = argparse.ArgumentParser(description='')

    # dateset
    parser.add_argument("--data_path", type=str, default='') 
    parser.add_argument("--save_path", type=str, default='') 
    
    # model
    parser.add_argument("--call_type", type=str, default='api', choices=['api', 'llm', 'vllm']) 
    parser.add_argument("--model_name", type=str, default='***')

    # for LLM model
    parser.add_argument("--local_dir", type=str, default='')

    # for API model
    parser.add_argument("--proxy", type=str, default='')  

    # for Custom LLM model
    parser.add_argument("--tokenizer_path", type=str, default='')  
    parser.add_argument("--model_path", type=str, default='')  
    parser.add_argument("--load_type", type=str, default='CausalLM')  
    parser.add_argument("--query_type", type=str, default=None)  

    # for vLLM model
    parser.add_argument("--tensor_parallel_size", type=int, default=8)
    parser.add_argument("--stop_word", type=str, default=None)

    # for vLLM api 
    parser.add_argument("--api_url", type=str, default=None)  
    
    # prompt
    parser.add_argument("--prompt_type", type=str, default='') 
    parser.add_argument("--example_num", type=int, default=0)  

    # generate args
    parser.add_argument("--max_new_tokens", type=int, default=1024)  
    parser.add_argument("--return_n", type=int, default=1)

    # batch
    parser.add_argument("--batch_size", type=int, default=1)  

    # skipping task names
    parser.add_argument("--skip_tasks", nargs='+', type=str, default=None) 


    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    
    return args

  
if __name__ == '__main__':
    args = get_params()

    if args.call_type == 'api':
        from OpenaiAPI import OpenaiAPIMaster
        # API
        api_str = open("api_key.txt").readlines()[0].strip()
        if len(api_str.split()) > 1:
            api_key, base_url = api_str.split()
        else:
            api_key, base_url = api_str, None

        if args.proxy:
            os.environ['http_proxy'] = args.proxy
            os.environ['https_proxy'] = args.proxy

        model_name = args.model_name
        master = OpenaiAPIMaster(model_name, api_key, base_url)
        master.handshake()

    elif args.call_type == 'llm':
        from LLM_Master import LLM_Master
        master = LLM_Master(args.model_name, args.local_dir)

    elif args.call_type == 'vllm':
        from vLLM_Master import vLLM_Master
        from vllm import SamplingParams
        master = vLLM_Master(args.tokenizer_path, 
                                args.model_path,
                                tensor_parallel_size = args.tensor_parallel_size,
                                query_type = args.query_type)

    else: 
        raise NotImplementedError


    # data
    bbh_master = BBH_Master()
    bbh_master.load_datas(args.data_path)

    # prompt
    prompt_type = args.prompt_type
    example_num = args.example_num
    save_path = args.save_path
    
    batch_size = args.batch_size

    if prompt_type in ['Official_cot_prompt', 'Official_standard_prompt', 'Official_cot_prompt_no_system', 'Official_standard_prompt_no_system']:
        
        prompt_path = args.data_path.replace('bbh', 'cot-prompts')
        if prompt_type in ['Official_cot_prompt', 'Official_cot_prompt_no_system']:
            use_cot = True
        else:
            use_cot = False
        bbh_master.load_bbh_official_prompt(prompt_path, use_cot=use_cot)
        
        bbh_all_task_prompts = bbh_master.bbh_official_prompt

        if prompt_type in ['Official_cot_prompt_no_system', 'Official_standard_prompt_no_system']:
            for task, prompt in bbh_all_task_prompts.items():
                bbh_all_task_prompts[task] = {
                    'system_prompt': "",
                    'examples': prompt['examples'],
                }
            
    elif prompt_type in ['Zero_prompt', 'Zero_prompt_boxed_system']:
        system_prompt = {
            'Zero_prompt': "",
            'Zero_prompt_boxed_system': "Answer the question and put the final answer in \\boxed{}.",
            'Zero_prompt_step_boxed_system': "Answer the question. Think step by step and put the final answer in \\boxed{}.",
        }[prompt_type]

        bbh_all_task_prompts = {}
        for task in bbh_master.bbh_tasks:
            bbh_all_task_prompts[task] = {
                'system_prompt': system_prompt,
                'examples': [],
            }
            
    elif prompt_type in ['AoT_prompt']:
        path = f"../data/prompts/BBH_AoT_prompt/bbh_prompt.json"
        bbh_all_task_prompts = json.load(open(path))

    else:
        raise NotImplementedError

    print(bbh_all_task_prompts)

    # skip_tasks
    skip_tasks = []
    if args.skip_tasks:
        if type(args.skip_tasks) == str:
            skip_tasks = [args.skip_tasks]
        elif type(args.skip_tasks) == list:
            skip_tasks = args.skip_tasks
        assert all([task in BBH_Master().bbh_tasks for task in skip_tasks])
    print(f"skip_tasks: {skip_tasks}")
    print(f"not skip_tasks: {[task for task in BBH_Master().bbh_tasks if task not in skip_tasks]}")

    if args.stop_word:
        assert args.call_type == 'vllm'
        print(f"use stop word: |{args.stop_word}|")
    
    predicted_datas = {}
    for i, task_name in enumerate(bbh_master.bbh_datas.keys()):
        # if i < 3: continue
        print(f"Evaluating {i+1}/{len(bbh_master.bbh_datas)} {task_name}")

        if task_name in skip_tasks:
            print(f"Skipping task {task_name}")
            continue

        datas = bbh_master.bbh_datas[task_name]

        prompt = bbh_all_task_prompts[task_name]
        system_prompt = prompt['system_prompt']
        examples = prompt['examples'][:example_num]

        if "gpt-4" in args.model_name:
            system_prompt = "You are a strong step-by-step reasoner. Please follow your historical response format for the new questions."

        if batch_size == 1:
            for item in tqdm(datas):
                sample = item['input']

                if args.call_type == 'api':
                    response_str = master.get_response(sample, system_prompt, examples)

                elif args.call_type == 'llm':
                    query = master.make_query(sample, system_prompt, examples)
                    response_str = master.get_response(query)
                
                elif args.call_type == 'vllm':
                    sampling_params = SamplingParams(
                            skip_special_tokens=False,
                            spaces_between_special_tokens=False,
                            temperature=0.0,
                            top_k=-1,
                            top_p=1.0,
                            max_tokens=args.max_new_tokens,
                            stop=args.stop_word,
                        )
                    response_str = master.get_response(sample, system_prompt, examples, sampling_params)
                    if args.stop_word:
                        response_str = response_str+args.stop_word if not response_str.endswith(args.stop_word) else response_str

                else:
                    raise NotImplementedError

                item['prediction'] = response_str

        else:
            print(f"Batch Size: {batch_size}")
            total = (len(datas) + batch_size - 1) // batch_size
            for batch in tqdm(chunk(datas, batch_size), total=total):
                samples = [item['input'] for item in batch]

                if args.call_type == 'api':
                    response_strs = master.get_response_batch(samples, system_prompt, examples)

                elif args.call_type == 'llm':
                    raise NotImplementedError

                elif args.call_type == 'vllm':
                    sampling_params = SamplingParams(
                            skip_special_tokens=False,
                            spaces_between_special_tokens=False,
                            temperature=0.0,
                            top_k=-1,
                            top_p=1.0,
                            max_tokens=args.max_new_tokens,
                            stop=args.stop_word
                        )
                    response_strs = master.get_response_batch(samples, system_prompt, examples, sampling_params)
                    if args.stop_word:
                        response_strs = [
                            response_str+args.stop_word if not response_str.endswith(args.stop_word) else response_str
                            for response_str in response_strs
                        ]
                        
                else:
                    raise NotImplementedError

                for item, response_str in zip(batch, response_strs):
                    item['prediction'] = response_str

        predicted_datas[task_name] = datas

        os.makedirs(os.path.dirname(save_path),exist_ok=True)
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(predicted_datas, f)
            # f.writelines([json.dumps(item, ensure_ascii=False)+'\n' for item in datas])

    os.makedirs(os.path.dirname(save_path),exist_ok=True)
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(predicted_datas, f)