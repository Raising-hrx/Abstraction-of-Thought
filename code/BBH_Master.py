import json
import os
import re
import numpy as np

class BBH_Master():
    def __init__(self):
        
        self.bbh_tasks = ['boolean_expressions',
                             'causal_judgement',
                             'date_understanding',
                             'disambiguation_qa',
                             'dyck_languages',
                             'formal_fallacies',
                             'geometric_shapes',
                             'hyperbaton',
                             'logical_deduction_five_objects',
                             'logical_deduction_seven_objects',
                             'logical_deduction_three_objects',
                             'movie_recommendation',
                             'multistep_arithmetic_two',
                             'navigate',
                             'object_counting',
                             'penguins_in_a_table',
                             'reasoning_about_colored_objects',
                             'ruin_names',
                             'salient_translation_error_detection',
                             'snarks',
                             'sports_understanding',
                             'temporal_sequences',
                             'tracking_shuffled_objects_five_objects',
                             'tracking_shuffled_objects_seven_objects',
                             'tracking_shuffled_objects_three_objects',
                             'web_of_lies',
                             'word_sorting']
        self.NLP_tasks = ['causal_judgement', 'date_understanding', 'disambiguation_qa', 'formal_fallacies', 'hyperbaton', 'movie_recommendation', 'penguins_in_a_table', 'reasoning_about_colored_objects', 'ruin_names', 'salient_translation_error_detection', 'snarks', 'sports_understanding']
        self.Algorithmic_tasks = ['boolean_expressions', 'dyck_languages', 'geometric_shapes', 'logical_deduction', 'multistep_arithmetic_two', 'navigate', 'object_counting', 'temporal_sequences', 'tracking_shuffled_objects', 'web_of_lies', 'word_sorting']
        
    
    def match_task_name(self, input_string):
        pattern = re.compile('|'.join(map(re.escape, self.bbh_tasks)), re.IGNORECASE)
        match = pattern.search(input_string)
        return match.group() if match else None
    
    def load_datas(self, bbh_dir):
        # bbh_dir = "../data/BIG-Bench-Hard/bbh"
        
        bbh_datas = {}
        for file in os.listdir(bbh_dir):
            if not file.endswith('.json'): continue
            task_name = file.split('.')[0].strip()
            if task_name not in self.bbh_tasks: continue
            
            path = os.path.join(bbh_dir, file)
            datas = json.load(open(path))['examples']

            bbh_datas[task_name] = datas
            
        self.bbh_datas = bbh_datas
        print(self.__class__.__name__, f"Load BIG-Bench-Hard from {bbh_dir}")
        
    def load_datas_json(self, path):
        bbh_datas = json.load(open(path))
        self.bbh_datas = bbh_datas
        print(self.__class__.__name__, f"Load json file from {path}")
        
        
        not_loaded_tasks = [task for task in self.bbh_tasks if task not in self.bbh_datas]
        if not_loaded_tasks:
            print(self.__class__.__name__, f"The following {len(not_loaded_tasks)} are NOT loaded: {not_loaded_tasks}")
        
    def load_bbh_official_prompt(self, cot_prompt_dir, use_cot=False):
        # cot_prompt_dir = "../data/BIG-Bench-Hard/cot-prompts" # from official bbh repo
        
        bbh_official_prompt = {}
        for task_name in self.bbh_datas:
            path = os.path.join(cot_prompt_dir, f"{task_name}.txt")
            
            prompt_txt = open(path).read()
            prompt_txt = prompt_txt.split('-----\n')[-1]

            system_prompt = prompt_txt.split('\n\nQ: ')[0]

            examples = []
            for line in prompt_txt.replace(system_prompt, "").strip().split('\n\n'):
                
                if use_cot:
                    examples.append({
                        'user': line.split("\nA: ", maxsplit=1)[0].replace("Q: ",""),
                        'assistant': line.split("\nA: ", maxsplit=1)[1],
                    })
                else:
                    examples.append({
                        'user': line.split("\nA: ", maxsplit=1)[0].replace("Q: ",""),
                        'assistant': line.split("\nA: ", maxsplit=1)[1].split("So the answer is")[-1].strip('.'),
                    })
            
            bbh_official_prompt[task_name] = {
                'system_prompt':system_prompt,
                'examples':examples,
            }
        self.bbh_official_prompt = bbh_official_prompt
        print(self.__class__.__name__, f"Load BIG-Bench-Hard Official prompt from {cot_prompt_dir}. use_cot: {use_cot}")
        
    def load_single_task_data(self, path, process_func = None):
        task_name = self.match_task_name(path)
        if task_name is None:
            print("The path should contain the name of the task.")
            return None
        
        datas = json.load(open(path))
        if 'outputs' in datas:
            datas = datas['outputs']
            
        if process_func:
            for item in datas:
                process_func(item)
            
        bbh_datas = {}
        bbh_datas[task_name] = datas
        self.bbh_datas = bbh_datas
        
    def evaluate_single_task(self, task_name, datas, return_info = False, pred_func = None):
        
        # multiple choice; consider (X)
        if task_name in ['date_understanding', 'disambiguation_qa', 'geometric_shapes', 'hyperbaton', 'logical_deduction_five_objects', 'logical_deduction_seven_objects', 'logical_deduction_three_objects',  'movie_recommendation', 'penguins_in_a_table', 'reasoning_about_colored_objects', 'ruin_names', 'salient_translation_error_detection', 'snarks', 'temporal_sequences', 'tracking_shuffled_objects_five_objects', 'tracking_shuffled_objects_seven_objects', 'tracking_shuffled_objects_three_objects']:
            def compare(a,b):
                if len(a.replace('(','').replace(')','').strip()) == 1: # the pred could be single "A"
                    return a.replace('(','').replace(')','').strip().lower() == b.replace('(','').replace(')','').strip().lower()
                
                return set(re.findall(r'\(([A-Za-z]+)\)', a)) == set(re.findall(r'\(([A-Za-z]+)\)', b))
        
        # Exact match; condsider only a-z and A-Z
        elif task_name in ['boolean_expressions', 'causal_judgement',  'formal_fallacies',  'navigate',  'sports_understanding', 'web_of_lies', 'word_sorting']:
            def compare(a,b):
                if len(b.split()) == 1: # the target is a single word like "No" "valid"
                    # return re.sub(r'[^a-zA-Z]', '', b).lower() in re.sub(r'[^a-zA-Z]', '', a).lower() # not good: "No" can in "connot"
                    return re.sub(r'[^a-zA-Z]', '', a).lower().startswith(re.sub(r'[^a-zA-Z]', '', b).lower())
                return re.sub(r'[^a-zA-Z]', '', a).lower() == re.sub(r'[^a-zA-Z]', '', b).lower()

        # number
        elif task_name in ['multistep_arithmetic_two', 'object_counting']:
            def compare(a,b):
                try:
                    return float(a.strip('.')) == float(b.strip('.'))
                except ValueError:
                    return False
            
        # dyck_languages
        elif task_name in ['dyck_languages']:
            def compare(a,b):
                return re.sub(r'[^(){}<>[\]]', '', a) == re.sub(r'[^(){}<>[\]]', '', b)

        if not pred_func:
            metrics = [compare(str(item['pred']), item['target']) for item in datas]
        else:
            metrics = [compare(str(pred_func(item)), item['target']) for item in datas]

        metric = np.mean(metrics)

        if return_info:
            return metric, metrics
        else:
            return metric

    def evaluate_all_task(self, pred_func = None):
        
        all_datas = self.bbh_datas
        
        metrics = {}
        
        metrics['raw'] = {}
        for task_name in self.bbh_tasks:
            accuracy = self.evaluate_single_task(task_name, all_datas[task_name], pred_func = pred_func)
            metrics['raw'][task_name] = accuracy
        
        metrics['report'] = {}
        for task_name in ['boolean_expressions', 'causal_judgement', 'date_understanding', 'disambiguation_qa', 'dyck_languages', 'formal_fallacies', 'geometric_shapes', 'hyperbaton', 'movie_recommendation', 'multistep_arithmetic_two', 'navigate', 'object_counting', 'penguins_in_a_table', 'reasoning_about_colored_objects', 'ruin_names', 'salient_translation_error_detection', 'snarks', 'sports_understanding', 'temporal_sequences', 'web_of_lies', 'word_sorting']:
            metrics['report'][task_name] = metrics['raw'][task_name]
        metrics['report']['logical_deduction'] = np.mean([metrics['raw'][sub] for sub in ['logical_deduction_five_objects','logical_deduction_seven_objects','logical_deduction_three_objects']])
        metrics['report']['tracking_shuffled_objects'] = np.mean([metrics['raw'][sub] for sub in ['tracking_shuffled_objects_five_objects','tracking_shuffled_objects_seven_objects','tracking_shuffled_objects_three_objects']])
        
        NLP_tasks = self.NLP_tasks
        Algorithmic_tasks = self.Algorithmic_tasks

        metrics['NLP'] = round(np.mean([metrics['report'][sub] for sub in NLP_tasks])*100, 1)
        metrics['Algorithmic'] = round(np.mean([metrics['report'][sub] for sub in Algorithmic_tasks])*100, 1)
        metrics['All'] = round(np.mean([metrics['report'][sub] for sub in NLP_tasks+Algorithmic_tasks])*100, 1)
        
        metrics['report'] = {k:round(v*100,1) for k,v in metrics['report'].items()}
        
        return metrics
        


if __name__ == "__main__":

    # -----------  Our implemented metrics match the official metrics -----------
    bbh_master = BBH_Master()
    for task_name in bbh_master.bbh_tasks:
    #     p = f"../data/BIG-Bench-Hard/code-davinci-002-outputs/code-davinci-002-direct/{task_name}_few_shot_template_0-255000.json"
    #     pp = f"../data/BIG-Bench-Hard/code-davinci-002-outputs/code-davinci-002-direct/{task_name}_few_shot_template_0-255000_eval_metrics.jsonl"
    #     bbh_master.load_single_task_data(p, process_func=lambda x: x.update({'pred': x['prediction']}))

        p = f"../data/BIG-Bench-Hard/code-davinci-002-outputs/code-davinci-002-cot/{task_name}_few_shot_template_0-255000.json"
        pp = f"../data/BIG-Bench-Hard/code-davinci-002-outputs/code-davinci-002-cot/{task_name}_few_shot_template_0-255000_eval_metrics.jsonl"
        bbh_master.load_single_task_data(p, process_func=lambda x: x.update({'pred': x['prediction'].split("the answer is")[-1]}))
        
        my_eval = bbh_master.evaluate_single_task(task_name, bbh_master.bbh_datas[task_name])
        official_eval = json.load(open(pp))['accuracy']
        print(my_eval*100 == official_eval, task_name, my_eval, official_eval)
    
