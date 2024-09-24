# Ref: https://platform.openai.com/docs/api-reference/introduction

import json
import openai
from openai import OpenAI, AsyncOpenAI
import time
import os
import asyncio

# import nest_asyncio
# nest_asyncio.apply()

class OpenaiAPIMaster:
    def __init__(self, model_name = 'gpt-4', api_key = None, base_url = None):
        self.model_name = model_name
        
        if api_key is None:
            if openai.api_key is None:
                print("Please set the api_key")
        else:
            openai.api_key = api_key

        if base_url is None:
            self.client = OpenAI(api_key=api_key)
        else:
            self.client = OpenAI(api_key=api_key, base_url=base_url)
            
        self.api_key = api_key
        self.base_url = base_url
        
        print(f"api_key: {api_key}")
        print(f"base_url: {base_url}")
        print(f"model_name: {model_name}")
        print(f"http/https proxy: {os.environ.get('http_proxy', 'No proxy')} / {os.environ.get('https_proxy', 'No proxy')}")
    
    def chat_query(self, query, retry = True, generate_args = {}):
        
        done = False
        while not done:
            
            try:
                messages = []
                if isinstance(query, str):
                    messages.append(
                        {"role": "user", "content": query},
                    )
                elif isinstance(query, list):
                    messages += query
                else:
                    raise ValueError("Unsupported query: {0}".format(query))

                # get response
                response = self.client.chat.completions.create(
                                model = self.model_name,
                                messages = messages,
                                **generate_args
                )
                
                done = True
                
            except Exception as e:
                print(str(e))
                response = None
                done = True
                    
                retry_key_words = ['Rate limit reached', 'Request timed out', 'Connection', '429', '500', '503']
                if any([word in str(e) for word in retry_key_words]):
                    if retry:
                        time.sleep(1)
                        print('Retrying......')
                        done = False
                    else:
                        done = True  
                
                not_retry_key_words = ['400']
                if any([word in str(e) for word in not_retry_key_words]):
                    print("Skip this query")
                    done = True
            
        return response
    
    def handshake(self):
        start_time = time.time()
        
        self.chat_query("hello!")
        
        run_time = time.time() - start_time
        print(f"{run_time:.2f} seconds")
        
    def get_response(self, user_input, system_prompt, examples, generate_args = None):
        use_generate_args = generate_args if generate_args else {'temperature': 0.0, 'n': 1}
            
        query = [{"role": "system", "content": system_prompt}]
        for example in examples:
            query += [
                {"role": "user", "content": example['user']},
                {"role": "assistant", "content": example['assistant']},    
            ]
        query += [{"role": "user", "content": user_input}]

        response = self.chat_query(query, generate_args = use_generate_args)

        if not response:
            response_str = ""
        elif len(response.choices) == 1:
            response_str = response.choices[0].message.content
        else:
            response_str = [choice.message.content for choice in response.choices]

        return response_str
    
    
    ### ------ async functions ------
    def init_client_async(self):
        if (not hasattr(self, 'client_async')) or (self.client_async is None):
            if self.base_url is None:
                self.client_async = AsyncOpenAI(api_key=self.api_key)
            else:
                self.client_async = AsyncOpenAI(api_key=self.api_key, base_url=self.base_url)
            
    
    async def chat_query_async(self, query, retry = True, generate_args = {}):
        self.init_client_async()
        
        done = False
        while not done:
            
            try:
                messages = []
                if isinstance(query, str):
                    messages.append(
                        {"role": "user", "content": query},
                    )
                elif isinstance(query, list):
                    messages += query
                else:
                    raise ValueError("Unsupported query: {0}".format(query))

                # get response
                # response = await self.client.chat.completions.create(
                response = await self.client_async.chat.completions.create(
                                model = self.model_name,
                                messages = messages,
                                **generate_args
                )
                
                done = True
                
            except Exception as e:
                print(str(e))
                response = None
                done = True
                    
                retry_key_words = ['Rate limit reached', 'Request timed out', 'Connection', '429', '500', '503']
                if any([word in str(e) for word in retry_key_words]):
                    if retry:
                        time.sleep(1)
                        print('Retrying......')
                        done = False
                    else:
                        done = True  

                not_retry_key_words = ['400']
                if any([word in str(e) for word in not_retry_key_words]):
                    print("Skip this query")
                    done = True
                
        return response
    
    async def get_response_async(self, user_input, system_prompt, examples, generate_args = None):
        use_generate_args = generate_args if generate_args else {'temperature': 0.0, 'n': 1}

        query = [{"role": "system", "content": system_prompt}]
        for example in examples:
            query += [
                {"role": "user", "content": example['user']},
                {"role": "assistant", "content": example['assistant']},    
            ]
        query += [{"role": "user", "content": user_input}]

        response = await self.chat_query_async(query, generate_args = use_generate_args)
        response_str = response.choices[0].message.content if response else ""
        
        return response_str
    
    async def get_response_batch_async(self, user_inputs, system_prompt, examples, generate_args = None):
        tasks = [self.get_response_async(user_input, system_prompt, examples, generate_args) for user_input in user_inputs]
        responses = await asyncio.gather(*tasks)
        return responses
    
    def get_response_batch(self, user_inputs, system_prompt, examples, generate_args = None):
        assert type(user_inputs) == list
        return asyncio.run(self.get_response_batch_async(user_inputs, system_prompt, examples, generate_args))



if __name__ == '__main__':
    api_str = open("api_key.txt").readlines()[0].strip()
    if len(api_str.split()) > 1:
        api_key, base_url = api_str.split()
    else:
        api_key, base_url = api_str, None

    os.environ['http_proxy'] = "http://127.0.0.1:8891"
    os.environ['https_proxy'] = "http://127.0.0.1:8891"

    model_name = "gpt-4"
    api_master = OpenaiAPIMaster(model_name, api_key, base_url)
    api_master.handshake()
    