import os
import json
import re

from python_executor import PythonExecutor
from Parser import ProgramParser

def chunk(datas, batch_size):
    for i in range(0, len(datas), batch_size):
        yield datas[i:i+batch_size]
        
class AnswerExtractor:
    def __init__(self):
        self.python_executor = PythonExecutor(get_answer_from_stdout=True)
        self.import_str = "import sys\nimport time\nimport itertools\nfrom itertools import accumulate, product, permutations, combinations\nimport collections\nfrom collections import Counter, OrderedDict, deque, defaultdict, ChainMap\nfrom functools import lru_cache\nimport math\nfrom math import sqrt, sin, cos, tan, ceil, fabs, floor, gcd, exp, log, log2\nimport fractions\nfrom typing import List, Tuple\nimport numpy as np\nimport random\nimport heapq\nfrom heapq import *\nimport re\nfrom datetime import datetime, timedelta\n"

    def clean_special_token(self, s):
        remove_special_tokens = [
            "<\|im_start\|>\s*assistant\n", # \s* mean any (or no) space or newline'\n'
            "<\|im_start\|>\s*system\n",
            "<\|im_start\|>\s*user\n",
            "<\|im_end\|>\s*</s>",
            "</s>",
            "<\|im_start\|>",
            "<\|im_end\|>",
            "\[INST\]",
            "\[/INST\]",
            "<<SYS>>",
            "<\|end_of_text\|>",      
        ]

        for token in remove_special_tokens:
            s = re.sub(token, "", s)
        return s
    
    def clean_text_md(self, s):
        matches = re.findall(r'\\text{(.*)}', s)
        return matches[0] if matches else s

    def extract_boxed(self, s, greedy=True):
        if not s: return ""
        if greedy:
            matches = re.findall(r'\\boxed{(.*)}', s)
        else:
            matches = re.findall(r'\\boxed{(.*?)}', s)
        return matches[0] if matches else ""

    def extract_python(self, s, add_import = True):
        return self.extract_python_batch([s], add_import=add_import)[0]
    
    def extract_python_batch(self, batch_s, add_import = True):
        # the input must be: ```python\n *** ```
        batch_s = [s if s else "" for s in batch_s]

        batch_code = [(i, ProgramParser.extract_python_code(s)) for i, s in enumerate(batch_s)]
        valid_codes = [(i, code) for i, code in batch_code if code]

        batch_predictions = [("","fail to extract code") for _ in range(len(batch_s))]
        
        if valid_codes:
            indices, codes = zip(*valid_codes)
            if add_import:
                codes = [self.import_str + code for code in codes]
            predictions = self.python_executor.batch_apply(codes)
            for i, prediction in zip(indices, predictions):
                        batch_predictions[i] = prediction
                    
        return batch_predictions
    
    def extract_aot(self, s):
        return self.batch_extract_aot([s])[0]
    
    def batch_extract_aot(self, batch_s, add_import = True):
        assert type(batch_s) == list
        text_preds = [self.extract_boxed(s) for s in batch_s]
        code_preds_ = self.extract_python_batch(batch_s, add_import=add_import)
        code_preds = [p_[0] if p_[1] == 'Done' else "" for p_ in code_preds_ ]
        
        results = []
        for s, text_pred, code_pred in zip(batch_s, text_preds, code_preds):
            # pred = code_pred if code_pred else text_pred
            # pred = pred if pred else s
            pred = code_pred or text_pred or s
            
            results.append({
                'text_pred': text_pred,
                'code_pred': code_pred,
                'pred': pred
            })
            
        return results
    
    def batch_extract_the_answer_is(self, batch_s):
        assert type(batch_s) == list
        batch_s = [s if s else "" for s in batch_s]
        results = []
        for s in batch_s:
            s = self.clean_special_token(s)
            if not s: results.append("")
            matches = re.findall('the answer is (.*)', s, re.IGNORECASE)
            results.append(matches[0].strip('.').strip() if matches else s)
        return results
    
    def batch_extract_first_line(self, batch_s):
        assert type(batch_s) == list
        batch_s = [s if s else "" for s in batch_s]
        results = []
        for s in batch_s:
            s = self.clean_special_token(s)
            if not s: results.append("")
            lines = s.split('\n')
            results.append(lines[0].strip('.').strip() if lines else s)
        return results
    
    def batch_extract_zero_v1(self, batch_s):
        assert type(batch_s) == list
        batch_s = [s if s else "" for s in batch_s]
        
        results = []
        for s in batch_s:
            s = self.clean_special_token(s)
            if not s: results.append("")

            # try \\boxed{}
            boxed_pred = self.extract_boxed(s)
            boxed_pred = self.clean_text_md(boxed_pred)

            # try "answer is"
            matches = re.findall('answer is (.*)', s, re.IGNORECASE)
            answer_is_pred = matches[0].strip('.').strip() if matches else ""

            pred = boxed_pred or answer_is_pred or s
            results.append(pred)
        return results

    @classmethod
    def compare_func(cls, a, b):
        if a == b:
            return True
        
        if a.lower().strip() == b.lower().strip():
            return True

        # (X)
        if set(re.findall(r'\(([A-Za-z]+)\)', a)) and set(re.findall(r'\(([A-Za-z]+)\)', b)) and \
            set(re.findall(r'\(([A-Za-z]+)\)', a)) == set(re.findall(r'\(([A-Za-z]+)\)', b)):
            return True

        # condsider only a-z and A-Z
        if re.sub(r'[^a-zA-Z]', '', a).lower() == re.sub(r'[^a-zA-Z]', '', b).lower() or \
            re.sub(r'[^a-zA-Z]', '', a).lower().startswith(re.sub(r'[^a-zA-Z]', '', b).lower()) or \
            re.sub(r'[^a-zA-Z]', '', b).lower().startswith(re.sub(r'[^a-zA-Z]', '', a).lower()):
            return True

        # math
        try:
            if float(a.strip('.')) == float(b.strip('.')):
                return True
        except:
            pass

        return False
    
    @classmethod
    def majority_vote(cls, preds):
        counter = {}
        for pred in preds:

            for i in counter.keys():
                if cls.compare_func(pred, i):
                    counter[i] += 1
                    break
            else:
                counter[pred] = 1

        sorted_counter = sorted(counter.items(), key=lambda x:x[1], reverse=True)

        return sorted_counter[0][0]
