import os
import json
import re
from tqdm import tqdm
from collections import defaultdict

from BBH_Master import BBH_Master
from Parser import ProgramParser
from answer_extractor import AnswerExtractor, chunk

import pandas as pd
pd.set_option('display.max_colwidth', None)

def extract_and_evaluate(path, extract_meth, overwrite = False):

    save_path = path.replace(".jsonl", ".extracted.jsonl")

    if os.path.exists(save_path) and not overwrite:
        pass
        # print("The extracted file exists")

    else:
        bbh_master = BBH_Master()
        bbh_master.load_datas_json(path)

        answer_extractor = AnswerExtractor()
        for task_name, datas in bbh_master.bbh_datas.items():
            # print(task_name)
            for batch_item in chunk(datas, batch_size=32):
                batch_s = [item['prediction'] for item in batch_item]
                if extract_method == 'aot':
                    batch_preds = answer_extractor.batch_extract_aot(batch_s)
                    for item, pred_dict in zip(batch_item, batch_preds):
                        item['pred'] = pred_dict['pred']
                        item['pred_dict'] = pred_dict
                elif extract_method == 'the_answer_is':
                    batch_preds = answer_extractor.batch_extract_the_answer_is(batch_s)
                    for item, pred in zip(batch_item, batch_preds):
                        item['pred'] = pred
                elif extract_method == 'first_line':
                    batch_preds = answer_extractor.batch_extract_first_line(batch_s)
                    for item, pred in zip(batch_item, batch_preds):
                        item['pred'] = pred  
                elif extract_method == 'zero_v1':
                    batch_preds = answer_extractor.batch_extract_zero_v1(batch_s)
                    for item, pred in zip(batch_item, batch_preds):
                        item['pred'] = pred  

            bbh_master.bbh_datas[task_name] = datas

        json.dump(bbh_master.bbh_datas, open(save_path, 'w'))
        print(f"Save the extracted file to {save_path}")

    bbh_master = BBH_Master()
    bbh_master.load_datas_json(save_path)

    result = bbh_master.evaluate_all_task()
    
    return result

exp_info = [
    ("Llama-3-8B-CoT.Zero_prompt.shot0.jsonl", "the_answer_is"),
    ("Llama-3-8B-AoT.Zero_prompt.shot0.jsonl", "aot"),
]

collector = []
collector_html = []

for path, extract_method in exp_info:
    base_dir = "/path/to/Abstraction-of-Thought/exp"
    read_path = os.path.join(base_dir, path)
    
    result = extract_and_evaluate(read_path, extract_method, overwrite=False)
    d = {}
    d['path'] = path
    d['NLP'] = result['NLP']
    d['Alg'] = result['Algorithmic']
    d['All'] = result['All']
    collector.append(d.copy())

    d.update(result['report'])
    collector_html.append(d.copy())

df = pd.json_normalize(collector)
print(df)

df_html = pd.json_normalize(collector_html)
save_path = os.path.join(base_dir, 'results.html')
df_html.to_html(save_path)
print(f"Saved to html: {save_path}")
