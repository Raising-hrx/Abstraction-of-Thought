import os
import json
import time

from vllm import LLM, SamplingParams
 
class vLLM_Master():
    def __init__(self, tokenizer_path, model_path, tensor_parallel_size = 8, query_type="concat", dtype="float32"):
        self.tokenizer_path = tokenizer_path
        self.model_path = model_path
        self.query_type = query_type
        self.tensor_parallel_size = tensor_parallel_size
        self.dtype = dtype
        
        self.build_model()
        self.case_test()
    
    def build_model(self):
        start_time = time.time()

        llm = LLM(
            model = self.model_path,
            tokenizer = self.tokenizer_path,
            tensor_parallel_size=self.tensor_parallel_size,
            dtype = self.dtype
        )

        sampling_params = SamplingParams(
                skip_special_tokens=False,
                spaces_between_special_tokens=False,
                temperature=0.0,
                top_k=-1,
                top_p=1.0,
                max_tokens=1024,
            )

        self.model = llm
        self.sampling_params = sampling_params
        
        end_time = time.time()
        print(f"Load {self.model_path}: {end_time-start_time:.2f}s")
    
    def make_query(self, user_input, system_prompt, examples):
        query_type = self.query_type

        if query_type == 'concat':
            query = system_prompt
            for example in examples:
                query += f"{example['user']} {example['assistant']}\n"
            query += user_input  
            
        elif query_type == 'chat':
            def format(role, message):
                return f"<|im_start|>{role}\n{message}<|im_end|>\n"

            query = format('system', system_prompt)
            for example in examples:
                query += format('user', example['user'])
                query += format('assistant', example['assistant'])
            query += format('user', user_input)

        elif query_type == 'chat-add':
            def format(role, message):
                return f"<|im_start|>{role}\n{message}<|im_end|>\n"

            query = format('system', system_prompt)
            for example in examples:
                query += format('user', example['user'])
                query += format('assistant', example['assistant'])
            query += format('user', user_input)
            
            query += "<|im_start|>assistant\n"

        elif query_type in ['lamma2-chat', 'codellama-instruct']:
            # ref: llama2-chat: https://gpus.llm-utils.org/llama-2-prompt-template/
            # ref: codellama-instruct: https://github.com/facebookresearch/codellama/blob/main/llama/generation.py#L319-L361

            query = ""
            query += f"[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n" # tokenizer add the first <s>
            for example in examples:
                query += f"{example['user']} [/INST] {example['assistant']}</s><s>[INST] "
            query += f"{user_input} [/INST]"

        elif query_type in ['lamma3-instruct']:
            # ref: https://github.com/meta-llama/llama3/blob/main/llama/tokenizer.py#L202
            # ref: https://huggingface.co/blog/llama3#how-to-prompt-llama-3    
        
            query = ""
            query += f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n" # will tokenizer add the first <|begin_of_text|> ??
            for example in examples:
                query += f"{example['user']} <|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n {example['assistant']} <|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
            query += f"{user_input} <|eot_id|>"

            query += "<|start_header_id|>assistant<|end_header_id|>\n\n"

        elif query_type in ['mistral-instruct']:
            # ref: https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2
            query = ""
            query += f"[INST] {system_prompt} "
            for example in examples:
                query += f"{example['user']} [/INST] {example['assistant']}</s> [INST] "
            query += f"{user_input} [/INST]"

        else:
            raise NotImplementedError
    
        return query

    def make_query_forced(self, user_input, forced_output, system_prompt, examples):
        # add forced_output to the input query
        query_type = self.query_type

        query = self.make_query(user_input, system_prompt, examples)
        if query_type == 'concat':
            query += " " + forced_output
            
        elif query_type == 'chat':
            query += f"<|im_start|>assistant\n{forced_output}"

        else:
            raise NotImplementedError
    
        return query

    def get_response(self, user_input, system_prompt, examples, custom_sampling_params=None, return_detail = False):
        use_sampling_params = custom_sampling_params if custom_sampling_params else self.sampling_params 
        
        query = self.make_query(user_input, system_prompt, examples)
        output = self.model.generate(query, use_sampling_params, use_tqdm = False) 
        # outputs: [RequestOutput(request_id=0, prompt='You are a robot.\nHello!', prompt_token_ids=[***], prompt_logprobs=None, outputs=[CompletionOutput(index=0, text=" I'm a robot.", token_ids=[***], cumulative_logprob=-22.24315240478336, logprobs=None, finish_reason=length)], finished=True)]
        
        if return_detail:
            return output

        output_str = output[0].outputs[0].text
        return output_str

    def get_response_forced(self, user_input, forced_output, system_prompt, examples, custom_sampling_params=None, return_detail = False):
        use_sampling_params = custom_sampling_params if custom_sampling_params else self.sampling_params 
        
        query = self.make_query_forced(user_input, forced_output, system_prompt, examples)
        output = self.model.generate(query, use_sampling_params, use_tqdm = False) 
        
        if return_detail:
            return output

        output_str = output[0].outputs[0].text
        return output_str
    
    def get_response_batch(self, user_inputs, system_prompt, examples, custom_sampling_params=None):
        assert type(user_inputs) == list
        
        use_sampling_params = custom_sampling_params if custom_sampling_params else self.sampling_params 

        querys = [self.make_query(user_input, system_prompt, examples) for user_input in user_inputs]
        outputs = self.model.generate(querys, use_sampling_params, use_tqdm = False) 
        
        output_strs = []
        for i in range(len(user_inputs)):
            output_str = outputs[i].outputs[0].text
            output_strs.append(output_str)

        return output_strs

    def case_test(self):
        system_prompt = "You are a robot.\n"
        sample = "Hello!"
        response = self.get_response(sample, system_prompt=system_prompt, examples = [])
        return True if len(response.strip()) > 0 else False