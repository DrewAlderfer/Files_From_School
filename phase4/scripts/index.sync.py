# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.0
#   kernelspec:
#     display_name: Python 3 (learn-env)
#     language: python
#     name: python3
# ---

# %%
from glob import glob, iglob
import os
import json
from os.path import isdir
import pprint as pp
import re
from shutil import copy
from combine_books import combine_books
pp.PrettyPrinter(indent=4)


# %%
# %load_ext autoreload
# %autoreload 2
# %aimport combine_books

# %%
def print_notebook(nb_path):
    try:
        with open(nb_path, 'r', encoding='UTF-8') as file:
            data = json.load(file)
    except FileNotFoundError as err:
        raise err
    else:
        for cell in data['cells']:
            last = ""
            if cell['cell_type'] == "code":
                first = "```python"
                last = "```"
                yield first
            for src_line in cell['source']:
                src_line = src_line.rstrip()
                src_line = src_line.encode(encoding='ascii', errors='namereplace')
                src_line = src_line.decode()
                yield src_line
            yield last

# %%
def get_cells(nb_file):
    with open(nb_file, "r", encoding='utf-8') as book:
        note_book = json.load(book)
    return note_book['cells']

# %%
topics = set()
for itm in iglob("C:/Users/Drew Alderfer/code/flatiron/projects/phase4/tutorials/**"):
    itm = os.path.basename(itm)[:2]
    if itm.isdigit():
        topics.add(itm)

# %%
home_dirs = [f"{itm}-00-Home" for itm in topics]
home_dirs

# %%
topic_dirs = []
for index_file in iglob("C:/Users/Drew Alderfer/code/flatiron/projects/phase4/tutorials/**/index.ipynb"):
    topic_dirs.append(os.path.relpath(index_file))
pp.pprint(topic_dirs)

# %%
for index, topic in enumerate(topics):
    lines = []
    os.chdir("C:/Users/Drew Alderfer/code/flatiron/projects/phase4/tutorials/")
    os.makedirs(home_dirs[index], exist_ok=True)
    os.makedirs(f"./{home_dirs[index]}/images", exist_ok=True)
    print(f"Working on:\n./{home_dirs[index]}/")
    found_files  = glob(f"./{topic}*/*[!.sync].ipynb")
    for file_path in found_files:
        if not re.search(r'solution', file_path):
            continue
        # print(file_path)
        search_path = os.path.split(file_path)
        # if images:=glob(f"{search_path[0]}/images/**"):
        #     for img in images:
        #         print(".", end="")
        #         copy(img, os.path.abspath(f"./{home_dirs[index]}/images/"))
        if data_file:=glob(f"{search_path[0]}/*[!.ipynb][!.md][!images]"):
            for data in data_file:
                if os.path.isdir(data):
                    continue
                print(data)
                copy(data, os.path.abspath(f"./{home_dirs[index]}/"))
    #     for line in print_notebook(os.path.abspath(file_path)):
    #         lines.append(line)
    #         lines.append("\n")
    #     lines.append("\n-----File-Boundary-----\n")
    # with open(f"./{home_dirs[index]}/index.md", "w", encoding='utf-8') as home_file:
    #     print(f"\nWriting file:\n./{home_dirs[index]}/index.md")
    #     home_file.writelines(lines)
    print("-" * 15)

# %%
with open("../tutorials/33-06-dsc-hierarchical-agglomerative-clustering-codealong/solution_index.ipynb", "r") as file:
    nb = json.load(file)


# %%
for key in nb.keys():
    if key != "cells":
        print(key)
        pp.pprint(nb[key], indent=4)


# %%
target_dirs = ["C:/Users/Drew Alderfer/code/flatiron/projects/phase4/labs", "C:/Users/Drew Alderfer/code/flatiron/projects/phase4/tutorials"]
for tar_dir in target_dirs:
    combine_books(tar_dir, copy_imgs=True, copy_data_files=True)


