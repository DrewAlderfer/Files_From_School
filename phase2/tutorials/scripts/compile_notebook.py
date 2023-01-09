import glob
import json
import os
import re


def notebook_parser(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        notebook = json.load(file)
        for cell in notebook['cells']:
            if cell['cell_type'] == 'code':
                line_gen = (line for line in cell['source'] if not line.startswith("#"))
                for line in line_gen: 
                    line = line.rstrip()
                    line = line.encode(encoding='ascii', errors='namereplace')
                    line = line.decode()
                    yield f"{line}\n"

def main():
    """
    Main function of this automation script.
    """
    itm = r"C:/Users/Drew Alderfer/code/flatiron/projects//phase3/labs/25-07-dsc-roc-curves-and-auc-lab/solution_index.ipynb"
                  # r"C:/Users/Drew Alderfer/code/flatiron/projects/phase3/tutorials/"]
                  # r"C:/Users/Drew Alderfer/code/flatiron/NYC-DS-091922/Phase_1/",
                  # r"C:/Users/Drew Alderfer/code/flatiron/NYC-DS-091922/Phase_2/",
                  # r"C:/Users/Drew Alderfer/code/flatiron/projects/phase2/labs/",
                  # r"C:/Users/Drew Alderfer/code/flatiron/projects/phase2/tutorials/"]
    with open('./code.py', 'w') as file:
        code = []
        for line in notebook_parser(itm):
            code.append(line)
        file.writelines(code)

if __name__ == '__main__':
    main()
