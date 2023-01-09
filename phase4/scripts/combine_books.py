import os
import re
import json
from glob import iglob, glob
from shutil import copy

def get_cells(nb_file:str):
    with open(nb_file, "r", encoding='utf-8') as book:
        note_book = json.load(book)
    return note_book['cells']

def get_metadata(nb_file:str):
    result = {}
    with open(nb_file, "r", encoding='utf-8') as book:
        note_book = json.load(book)
        for key, val in note_book.items():
            if key != 'cells':
                result.update({key: val})
    return result


def combine_books(dir_path:str, copy_imgs:bool=False, copy_data_files:bool=False):

    print("Attempt 2")
    dir_path = os.path.abspath(dir_path)
    topics = set()
    for itm in iglob(f"{dir_path}/**"):
        itm = os.path.basename(itm)[:2]
        if itm.isdigit():
            topics.add(itm)

    home_dirs = [f"{itm}-00-Home" for itm in topics]

    topic_dirs = []
    for index_file in iglob(f"{dir_path}/**/index.ipynb"):
        topic_dirs.append(os.path.relpath(index_file))

    for index, topic in enumerate(topics):
        cells = []
        comb_nb = {}
        os.chdir(dir_path)
        os.makedirs(home_dirs[index], exist_ok=True)
        print(f"Working on:\n./{home_dirs[index]}/")
        found_files  = glob(f"./{topic}*/*[!.sync].ipynb")
        meta = False
        for file_path in found_files:
            if re.search(r'-Home', file_path):
                continue
            if not meta:
                comb_nb.update(get_metadata(os.path.abspath(file_path)))
                meta = True
            if not re.search(r'solution', file_path):
                continue
            if not re.search(r'solution', file_path):
                continue
            print(file_path)
            search_path = os.path.split(file_path)
            if copy_imgs:
                os.makedirs(f"{home_dirs[index]}/images/", exist_ok=True)
                images = glob(f"{search_path[0]}/images/**")
                for img in images:
                    print(".", end="")
                    copy(img, os.path.abspath(f"./{home_dirs[index]}/images/"))
            if copy_data_files:
                data_file = glob(f"{search_path[0]}/*[!.ipynb][!.md][!images]")
                for data in data_file:
                    if os.path.isdir(data):
                        continue
                    print(data)
                    copy(data, os.path.abspath(f"./{home_dirs[index]}/"))
            cells.extend(get_cells(file_path))
        comb_nb['cells'] = cells
        with open(f"./{home_dirs[index]}/index.ipynb", "w", encoding='utf-8') as home_file:
            print(f"\nWriting file:\n./{home_dirs[index]}/index.ipynb")
            json.dump(comb_nb, home_file)
        print("-" * 15)
