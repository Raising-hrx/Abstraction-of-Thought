import json
import re
import autopep8

import redbaron
from redbaron import RedBaron

class TextParser:
    def __init__(self, response_str, MASK_TOKEN = "<FILL_ME>"):
        self.response_str = response_str
        self.parsed = None
        self.MASK_TOKEN = MASK_TOKEN
    
    def parse(self):
        if self.parsed:
            return self.parsed
        
        lines = self.response_str.splitlines(keepends=True)

        parsed = []

        current_head = None
        current_body = []
        for line in lines:
            if re.match(r'Step \d+', line):
                if current_head is not None:
                    parsed.append({'type': 'head', 'text': current_head})
                    parsed.append({'type': 'body', 'text': ''.join(current_body)})
                current_head = line
                current_body = []
            else:
                current_body.append(line)

        if current_head is not None:
            parsed.append({'type': 'head', 'text': current_head})
            parsed.append({'type': 'body', 'text': ''.join(current_body)})

        self.parsed = parsed
        return parsed

    def get_abstract_plan(self):
        if self.parsed is None:
            self.parse()
            
        abstract_plan = ""
        for i in self.parsed:
            if i['type'] != 'head':
                continue
            abstract_plan += i['text']
            
        return abstract_plan
    
    def get_body_infill_pairs(self):
        if self.parsed is None:
            self.parse()
            
        body_infill_pairs = []
        for idx, i in enumerate(self.parsed):
            if i['type'] != "body":
                continue
            
            input_plan = "".join([i_['text'] for i_ in self.parsed[:idx]])
            input_plan += self.MASK_TOKEN
            input_plan += "".join([i_['text'] for i_ in self.parsed[idx+1:] if i_['type']=='head'])
            
            body_infill_pairs.append({
                'input_plan': input_plan,
                'output': i['text']
            })
            
        return body_infill_pairs
    
    def get_head_infill_pairs(self):
        if self.parsed is None:
            self.parse()
    
        head_infill_pairs = []
        for idx, i in enumerate(self.parsed):
            if i['type'] != "head":
                continue
                
            input_plan = "".join([i_['text'] for i_ in self.parsed[:idx]])
            input_plan += self.MASK_TOKEN
            input_plan += "".join([i_['text'] for i_ in self.parsed[idx+1:]])
            
            head_infill_pairs.append({
                'input_plan': input_plan,
                'output': i['text']
            })
            
        return head_infill_pairs
    

class ProgramParser:
    def __init__(self, python_str, MASK_TOKEN = "<FILL_ME>"):
        self.python_str = python_str
        self.parsed = None
        self.MASK_TOKEN = MASK_TOKEN
        
        self.lines = self.python_str.splitlines(keepends=True)

    @staticmethod
    def extract_python_code(text, fix_code=True):
        pattern = r'```python(.*?)```'
        match = re.search(pattern, text, re.DOTALL)
        if match:
            python_str = match.group(1).strip()
            if fix_code:
                python_str = autopep8.fix_code(python_str) # fix the code 
            return python_str
        else:
            return None
        
    @staticmethod
    def add_python_box(s):
        if not s.startswith("```python"):
            s = "```python\n" + s
        if not s.endswith("```"):
            s = s + "\n```"
        return s

    def get_node_box(self, node):
        line_start, column_start = node.absolute_bounding_box.top_left.to_tuple()
        line_end, column_end = node.absolute_bounding_box.bottom_right.to_tuple()

        line_start, line_end = line_start-1, line_end-1
        column_start, column_end = column_start-1, column_end-1

        return line_start, line_end, column_start, column_end

    def get_node_pos(self, node):
        lines = self.lines
        line_start, line_end, column_start, column_end = self.get_node_box(node)

        start_idx = sum([len(lines[i]) for i in range(line_start)]) + column_start
        end_idx = sum([len(lines[i]) for i in range(line_end)]) + column_end + 1 

        if node.type in ["comment"]:
            node_str = node.dumps()
            
        else:
            # add the start indentation
            start_idx -= len(node.indentation) 
            # remove the end indentation
            s = node.dumps()
            end_indentation = len(s) - len(s.rstrip(' '))
            end_idx -= end_indentation

            node_str = node.indentation + node.dumps()
            node_str = node_str.rstrip(' ')
            
        if node_str != self.python_str[start_idx:end_idx]:
            print(f"\n|{node_str}| vs. |{self.python_str[start_idx:end_idx]}|")
            print("type:", node.type)
#         assert node_str == self.python_str[start_idx:end_idx]
            
        return [start_idx, end_idx]

    def parse_defnode(self, defnode):
        assert defnode.type == 'def'

        node_start, node_end = self.get_node_pos(defnode)

        comment = None
        normal = []

        for sub_node in defnode.value:
            if sub_node.type in ['string'] and comment is None:
                comment = self.get_node_pos(sub_node)
            else:
                normal.append(self.get_node_pos(sub_node))

        parse_info = {
            'head': [node_start, normal[0][0]],
            'body': [normal[0][0], node_end],
        }
        return parse_info

    def parse_classnode(self, classnode):
        assert classnode.type == 'class'

        node_start, node_end = self.get_node_pos(classnode)

        parse_info = {}
        first_def_idx = None
        sub_def_infos = []

        for sub_node in classnode.value:
            if sub_node.type in ['def'] and first_def_idx is None:
                first_def_idx = self.get_node_pos(sub_node)[0]

            if sub_node.type in ['def']:
                sub_def_infos.append(self.parse_defnode(sub_node))

        if sub_def_infos:
            sub_def_infos[-1]['body'][1] = node_end

        else:
            first_def_idx = node_end

        parse_info = {
            'head': (node_start, first_def_idx),
            'sub_def_infos': sub_def_infos
        }

        return parse_info

    def is_main_function(self, def_parse_info):
        # if 'main' in the function name or comment, we consider it as the main function
        a,b = def_parse_info['head']
        func_head = self.python_str[a:b]
        return (' main ' in func_head) or ('main(' in func_head)
    
    def parse(self):
        if self.parsed:
            return self.parsed
        
        try:
            red = RedBaron(self.python_str)
        except Exception as e:
            print("RedBaron Parse Fail: ", str(e))
            print(f"RedBaron Parse Fail python_str: {self.python_str}")
            return []

        parsed = []

        for node in red:
            if node.type in ['def']:
                parse_info = self.parse_defnode(node)
                
                if self.is_main_function(parse_info):
                    parsed.append({'type': 'head','pos': parse_info['head']})
                    parsed.append({'type': 'head','pos': parse_info['body']})
                else:
                    parsed.append({'type': 'head','pos': parse_info['head']})
                    parsed.append({'type': 'body','pos': parse_info['body']})

            elif node.type in ['class']:
                parse_info = self.parse_classnode(node)
                parsed.append({'type': 'outline','pos': parse_info['head']})
                for sub_def_info in parse_info['sub_def_infos']:
                    if self.is_main_function(sub_def_info):
                        parsed.append({'type': 'head','pos': sub_def_info['head']})
                        parsed.append({'type': 'head','pos': sub_def_info['body']}) 
                    else:
                        parsed.append({'type': 'head','pos': sub_def_info['head']})
                        parsed.append({'type': 'body','pos': sub_def_info['body']}) 

            else:
                node_pos = self.get_node_pos(node)
                parsed.append({'type': 'outline','pos': node_pos})

        self.parsed = parsed
        return parsed

    def get_str_indentation(self, s):
        return ' ' * (len(s) - len(s.lstrip(' ')))

    def get_abstract_plan(self):
        if self.parsed is None:
            self.parse()
        
        abstrat_plan = self.python_str
        for i in reversed(self.parsed):
            if i['type'] == 'body':
                a,b = i['pos']
                abstrat_plan = abstrat_plan[:a] + f"{self.get_str_indentation(abstrat_plan[a:b])}pass\n\n" + abstrat_plan[b:]
        return abstrat_plan

    def get_body_infill_pairs(self):
        if self.parsed is None:
            self.parse()
            
        body_infill_pairs = []

        body_num = len([i for i in self.parsed if i['type']=='body'])
        for target_idx in range(body_num):
            pass_num = 0
            input_plan = self.python_str
            output_body = ""
            for i in reversed(self.parsed):
                if i['type'] != 'body':
                    continue
                if pass_num < target_idx:
                    # not target body, we replace with pass
                    a,b = i['pos']
                    input_plan = input_plan[:a] + f"{self.get_str_indentation(input_plan[a:b])}pass\n\n" + input_plan[b:]
                    pass_num += 1
                else:
                    # the target body, we replace with MASK_TOKEN 
                    a,b = i['pos']
                    output_body = input_plan[a:b]
                    input_plan = input_plan[:a] + f"{self.MASK_TOKEN}" + input_plan[b:]
                    break
            body_infill_pairs.append({
                'input_plan': input_plan,
                'output': output_body,
            })

        body_infill_pairs = body_infill_pairs[::-1]

        return body_infill_pairs

    def get_head_infill_pairs(self):
        if self.parsed is None:
            self.parse()
            
        head_infill_pairs = []

        input_plan = ""
        output_head = ""
        for i in self.parsed:
            if i['type'] != 'head':
                continue
            input_plan = self.python_str
            a,b = i['pos']
            output_head = input_plan[a:b]
            input_plan = input_plan[:a] + f"{self.MASK_TOKEN}" + input_plan[b:]

            if (' main ' in output_head) or ('main(' in output_head):
                # we do not infill the main function's head, or the code below main()
                break
                
            head_infill_pairs.append({
                'input_plan': input_plan,
                'output': output_head,
            })

        return head_infill_pairs

if __name__ == "__main__":

    code = '''def swap_positions(players, positions, swap_sequence):
    """
    This function takes a list of players and a list of positions, and a list of tuples representing 
    the sequence of swaps. It updates the lists of players and positions according to the swap sequence 
    and returns the updated lists.
    """
    for swap in swap_sequence:
        players[players.index(swap[0]), players[players.index(swap[1]), positions[players.index(swap[0]), positions[players.index(swap[1])] = positions[players.index(swap[1]), positions[players.index(swap[0])]
    return players, positions

def main():
    """
    The main function that calls the other functions and prints the answer.
    """
    players = ['Alice', 'Bob', 'Claire', 'Dave', 'Eve']
    positions = ['right winger', 'striker', 'benchwarmer',
                 'fullback', 'left midfielder']
    swap_sequence = [('Dave', 'Bob'), ('Claire', 'Eve'), ('Eve', 'Bob'),
                      ('Alice', 'Claire'), ('Alice', 'Dave')]
    players, positions = swap_positions(players, positions, swap_sequence)
    answer = positions.index('benchwarmer')  # Claire is the only benchwarmer left
    print('(C)' if answer == 2 else '(D)')


main()
'''
    print(ProgramParser(code).parse())