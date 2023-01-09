import glob
import ast
import json
import os
import re

# TO DO:
# - Refactor the Parser to split up the main() func and the parsing operations.
#   probably this is just throwing it into a generator function and feeding it
#   file paths to parse.
#
# - Fix the Import|From parsing to capture all forms of an import. The structure
#   will probably be similar to the obj_method regex
#
# - Start figuring out the comparator that will expand aliases and create the 
#   basic buidling block of the documentation output.

def method_parse(line):
    """
    Method Parse takes a string, searches is for a pattern, and returns a match as a dictionary item.
    """
    # object_pattern = r"([A-Za-z][A-Za-z0-9]+)\."
    # method_pattern = r"\.([A-Za-z][A-Za-z0-9]+)\(((?:[A-Za-z0-9 \\/=\'\"]{1,}[,\s]{0,2}){0,5})\)"
    obj_method_pattern = r"([A-Za-z][A-Za-z0-9]+)\.([A-Za-z][A-Za-z0-9]+)\(((?:[A-Za-z0-9 \\/=\'\"]{1,}[,\s]{0,2}){0,5})\)"
    methods = re.search(obj_method_pattern, line)
    # format I want these returned as:
    #     {
    #     capture: <match object>
    #     class_name: <class>
    #     method_name: <class method>
    #     args: [arg[0], arg[1], ...]
    #     }
    if methods:
        result = {'capture': methods.group(),
                  'class_name': methods.groups()[0],
                  'method_name': methods.groups()[1]}
        if methods.groups()[2] != "":
            result.update({'args': []})
            args = methods.groups()[2].split(',')
            for arg in args:
                result['args'].append(arg)
        return result
    return None

def parse_lib(str_input, capture=None):
    if "." in str_input:
        str_input = str_input.split('.')
        # I want to return the library and class if it's split and just the library if it's not
        return {'library':str_input[0], 'class':"".join(str_input[1:])}
    return {'library':str_input}

def make_alias(line, cap_list):
    return { line : ".".join([itm for itm in cap_list[0:-2] if itm not in ["from", "import", "as"]])}


def import_parse(line):
    # import_pattern = r"(import|from)\s([A-Za-z][A-Za-z0-9.]+)\s{0,1}(as|import)?\s{0,1}([A-Za-z][A-Za-z0-9._]+)?"
    import_pattern_01 = r"((import|from)\s{1,}([A-Za-z][A-Za-z0-9.]+)){1,}.*"
    # import_pattern_02 = r"((as|import)[\s,]{0,}([A-Za-z][A-Za-z0-9.]+)){1,}"
    import_pattern = import_pattern_01
    imports = re.search(import_pattern, line)
    if imports:
        result = {}
        result.update({'capture':imports.group()})
        print(result['capture'])
        capture_list = imports.group().split()
        fork_dict = {'from': parse_lib, 'import': parse_lib, 'as': make_alias}
        for identifier in capture_list:
            if identifier in ["from", "import", "as"]:
                slot = identifier
                continue
            result.update(fork_dict[slot](identifier, capture_list))
        #     print(result)
        return result
    return None


# { 
#  topics: [
#      {topic_name: "string",
#       libs: [],
#       captures: [],}
#      ]



            # lib = imports.groups()[1]
            # class_name = imports.groups()[3]
            # class_set.add((lib, class_name))
        # if "as" in imports.groups():
        #     print(imports.groups())
            # lib = imports.groups()[1]
            # alias = imports.groups()[3]
            # alias_set[lib] =  alias

# def get_dir_itms(dir=None, search=None):
#     pass

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
                    yield line

def main():
    """
    Main function of this automation script.
    """
    search_dir = [r"C:/Users/Drew Alderfer/code/flatiron/projects/phase3/labs/"] # ,
                  # r"C:/Users/Drew Alderfer/code/flatiron/projects/phase3/tutorials/"]
                  # r"C:/Users/Drew Alderfer/code/flatiron/NYC-DS-091922/Phase_1/",
                  # r"C:/Users/Drew Alderfer/code/flatiron/NYC-DS-091922/Phase_2/",
                  # r"C:/Users/Drew Alderfer/code/flatiron/projects/phase2/labs/",
                  # r"C:/Users/Drew Alderfer/code/flatiron/projects/phase2/tutorials/"]
    import_set = {}
    method_set = {}
    for r_dir in search_dir:
        os.chdir(r_dir)
        for itm in glob.iglob('./**/*[!sync].ipynb', recursive=True):
            if itm[2:7] == "Phase":
                path_split = itm.split('_')
                for split in path_split:
                    if "Topic" in split:
                        topic = split
                group_id = f"{topic}_{os.path.basename(itm)[:-6]}"
            else:
                group_id = f"{itm[2:7]}"
            import_set.update({group_id: {}})
            import_set[group_id].update({'libs': []})
            method_set.update({group_id: []})
            notebook = []
            for line in notebook_parser(itm):
                new_items = import_parse(line)
                if not new_items is None:
                    import_set[group_id]['libs'].append(new_items)
                    import_set[group_id].update(new_items)
                    import_set[group_id]['libs'].append(new_items['library'])
                    import_set[group_id].pop('library')
                    # print(import_parse(line))
                if not method_parse(line) is None:
                    method_set[group_id].append(method_parse(line))
            if len(method_set[group_id]) == 0:
                method_set.popitem()
            if len(import_set[group_id]) == 1:
                import_set.pop(group_id)
            elif len(import_set[group_id]['libs']) ==  0:
                import_set.pop(group_id)

    for name, content in import_set.items():
        print(f"\n{name}")
        for k, v in content.items():
            print(f"{k} = {v}")
        print("\n")
        # for items in content.items():
        #     print(items)

    # for name, content in method_set.items():
    #     print("\n" + "-" * 30)
    #     print(f"{name}")
    #     for entry in content:
    #         print("-" * 30)
    #         for k, v in entry.items():
    #             print(f"{k}: {v}")




                            # print(f"{line_num}  {line}")
    # print("Classes: ")
    # for (a, b) in class_set:
    #     print(f"From Library: {a}\nClass Name: {b}")
    # print("Library Aliases")
    # for (x, y) in alias_set.items():
    #     print(f"Library: {x}\nAlias: {y}")


if __name__ == '__main__':
    main()
