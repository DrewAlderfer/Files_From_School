import os
import ast
import glob
import json



class Visitor(ast.NodeVisitor):

    def visit_For(self, node: ast.AST, target_nodes=tuple()):
        method_result = []
        import_result = {"imports": set(), "modules": set(), "names": set()}
        asName = {}
        for child in ast.walk(node):
            if isinstance(child, (ast.Constant, ast.Name, ast.List, ast.Store, ast.Load,
                                  ast.Module, ast.BinOp, ast.Subscript, ast.arguments, ast.Add,
                                  ast.Sub, ast.keyword, ast.Attribute, ast.arg)):
                continue

            if isinstance(child, (ast.alias)):
                value = child.asname
                if value is None:
                    value = child.name
                asName[child.name] = value

            if isinstance(child, (ast.Expr, ast.Call, ast.Assign)):
                for branch in ast.walk(child):
                    if "id" in branch._fields:
                        if branch.id in list(asName.values()):
                            method_result.append((branch.id, ast.unparse(child)))
                            continue
                continue

            if isinstance(child, (ast.ImportFrom, ast.Import)):
                for branch in ast.walk(child):
                    if "module" in branch._fields:
                        import_result['modules'].add(branch.module)
                        continue
                    if "name" in branch._fields:
                        import_result['names'].add(branch.name)
                        if "asname" in branch._fields:
                            import_result['names'].add(branch.asname)
                import_result["imports"].add(ast.unparse(child))
                continue

        return (asName, method_result, import_result)


def notebook_parser(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        notebook = json.load(file)
        for cell in notebook['cells']:
            if cell['cell_type'] == 'code':
                line_gen = (line for line in cell['source'] if not line.startswith("#"))
                for line in line_gen: 
                    # line = line.rstrip()
                    # line = line.encode(encoding='ascii', errors='namereplace')
                    # line = line.decode()
                    if line.startswith("%"):
                        line = ""
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
    method_set = []
    names_dict = {}
    import_list = set()
    modules = set()
    names =   set()
    count = 0
    for r_dir in search_dir:
        os.chdir(r_dir)
        for itm in glob.iglob('./**/*[!sync].ipynb', recursive=True):
            if itm[2:7] == "Phase":
                topic = ""
                path_split = itm.split('_')
                for split in path_split:
                    if "Topic" in split:
                        topic = split
                group_id = f"{topic}_{os.path.basename(itm)[:-6]}"
            else:
                group_id = f"{itm[2:7]}"
            result = "" 
            for line in notebook_parser(itm):
                result += f"{line}\n"
            try:
                node = ast.parse(result)
            except SyntaxError as err:
                # print(f"{itm}: contains a syntax error:\n{err}")
                count += 1
                continue
            parsed_items = Visitor().visit_For(node, target_nodes=(ast.AST))
            names_dict.update(parsed_items[0])
            method_set.extend(parsed_items[1])
            import_list.update(parsed_items[2]['imports'])
            modules.update(parsed_items[2]['modules'])
            names.update(parsed_items[2]['names'])
            # import_list.append((group_id, Visitor().visit_For(node, target_nodes=(ast.Import, ast.ImportFrom))))
            # Visitor().visit_For(node, target_nodes=(ast.Assign))
            # method_set.append((group_id, Visitor().visit_For(node, target_nodes=(ast.Call))))

    name_iter = list(names)
    while None in name_iter:
        name_iter.remove(None)

    method_set = list(set(method_set))
    name_iter.sort(key=lambda x : x.lower())
    method_set.sort(key=lambda x : (x[0].lower(), x[1].lower()))
    input_dict = {k: v for k, v in sorted(names_dict.items(), key=lambda x : x[1].lower())}

    final_dict = {}
    for k, v in input_dict.items():
        final_dict[f"{k} as {v}"] = []
        for tup in method_set:
            if tup[0] == v:
                out = tup[1].encode(encoding='ascii', errors='namereplace')
                out = out.decode()
                final_dict[f"{k} as {v}"].append(out)
    
    comb_dict = {}
    for itm in import_list:
        split_import = itm.split()
        name_list = []
        dict_key = split_import[1]
        if not dict_key in comb_dict.keys():
            comb_dict.update({dict_key: set()})
        for name in names:
            if name in split_import:
                name_list.append(name)
        comb_dict[dict_key].update(name_list)

    import_lines = []
    for k, v in comb_dict.items():
        if k in v:
            v.remove(k)
            if len(v) == 0:
                import_lines.append(f"import {k}")
                continue
            for itm in v:
                import_lines.append(f"import {k} as {itm}")
            continue
        if len(v) > 3:
            import_lines.append(f"from {k} import (")
            for itm in v:
                  import_lines.append(f"\t{itm},")
            import_lines.append(")")
            continue
        imp_str = f"from {k} import"
        v = list(v)
        for itm in range(len(v)-1):
            imp_str += f" {v[itm]},"
        imp_str += f" {v[-1]}"

    file_lines = []
    for line in import_lines:
        file_lines.append(line)

    for k, v in final_dict.items():
        file_lines.append("#" + "-" * 100)
        file_lines.append(f"# {k}")
        file_lines.append("#" + "-" * 100)
        for itm in v:
            file_lines.append(itm)
        file_lines.append("#" + "-" * 100)
    
    # inspect = [f"{line}\n" for line in file_lines]
    # print("-" * 100)
    # print("\n" * 10)
    # for line in inspect:
    #     print(line, end="")
    
    os.chdir('C:/Users/Drew Alderfer/code/flatiron/projects/phase2/tutorials/scripts/')
    with open("ast_output.py", 'w') as file:
        file.writelines([f"{line}\n" for line in file_lines])

    # flat_list =accuracy = (23 + 97) / (23 + 97 + 1 + 9) []
    # for topic, items in import_list:
    #     if len(items) > 0:
    #         flat_list.append(f"# {topic}") 
    #         for line in items:
    #             flat_list.append(line) 
    # for each in list(set(flat_list)):
    #     print(each)
    # flat_methods = []
    # for topic, items in method_set:
    #     if len(items) > 0:
    #         flat_methods.append(f"# {topic}") 
    #         for line in items:
    #             flat_methods.append(line) 
    # for each in list(set(flat_methods)):
    #     print(each)
    

if __name__ == "__main__":
    main()
