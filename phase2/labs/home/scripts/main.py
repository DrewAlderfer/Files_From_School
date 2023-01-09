import os
import json
from shutil import copy2

def print_notebook(file_path):
    file_path = f"../{file_path}/index.ipynb"
    try:
        with open(file_path, 'r', encoding='UTF-8') as file:
            data = json.load(file)
    except FileNotFoundError:
        error = f"Didn't find an index.ipynb in {file_path}."
        return error
    else:
        result = []
        for cell in data['cells']:
            last = ""
            if cell['cell_type'] == "code":
                first = "```python"
                last = "```"
                yield first
            for line in cell['source']:
                line = line.rstrip()
                line = line.encode(encoding='ascii', errors='namereplace')
                line = line.decode()
                yield line
            yield last

def get_code(jup_json_obj):
    for cell in jup_json_obj:
        if cell['cell_type'] == "code":
            for line in cell['source']:
                line = line.rstrip()
                line = line.encode(encoding='ascii', errors='namereplace')
                line = line.decode()
                yield line

def main():
    base_dir = "../../"
    labs = [topic for topic in os.listdir("../../.") if topic[:2].isdigit()] 
    topics = set([lesson[:2] for lesson in os.listdir("../../.") if lesson[:2].isdigit()])
    print(labs[:5])
    print(topics)
    toc = {}
    for topic in topics:
        module = []
        for lesson in labs:
            if topic == lesson[:2]:
                module.append(lesson)
        toc[topic] = module

    for topic in toc.items():
        for lesson in topic[1]:
            file_path = f"{base_dir}/{lesson}/index.ipynb"
            try:
                with open(file_path, 'r', encoding='UTF-8') as file:
                    jup_nb = json.load(file)
            except FileNotFoundError:
                error = f"Didn't find an index.ipynb in {file_path}."
                return error
            for line in get_code(jup_nb['cells']):
                print(line)

if __name__ == '__main__':
    main()
