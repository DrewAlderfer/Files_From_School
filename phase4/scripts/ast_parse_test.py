import ast
import glob
import inspect
import json
import os
import pprint
from typing import BinaryIO, Dict, Union

pp = pprint.PrettyPrinter(indent=4)

from ast import NodeTransformer, iter_child_nodes, iter_fields


class Visitor(ast.NodeVisitor):
    def __init__(self, node: ast.AST, doc_name: str = ""):
        self.doc_node = node
        self.doc_name = doc_name

    def get_imports(self):
        results = {self.doc_name: {}}
        for child in ast.walk(self.doc_node):
            if isinstance(child, ast.Import):
                for alias in child.names:
                    results[self.doc_name][alias.name] = {
                            "type": "Import",
                            "name": alias.name
                            }
                    if "asname" in alias._fields:
                        results[self.doc_name][alias.name]["asname"] = alias.asname
        return results

    def get_importfrom(self):
        results = {self.doc_name: {}}
        for child in ast.walk(self.doc_node):
            if isinstance(child, ast.ImportFrom):
                namespace = child.module.split('.')
                results[self.doc_name].update({
                    child.module: {
                        'library': namespace[0],
                        'subpackage': ".".join(namespace[1:]),
                        'classes': {},
                        'type': 'ImportFrom'
                        }
                })
                for alias in child.names:
                    results[self.doc_name][child.module]['classes'].update({
                            alias.name:  {'name': alias.name}
                            })
                    if "asname" in alias._fields:
                        results[self.doc_name][child.module]['classes'][alias.name].update({'asname': alias.asname})
        return results

class FuncTransform(NodeTransformer):

    def __init__(self, node: ast.AST, doc_name:str, call_map:dict):
        self.doc_node = node
        self.doc_name = doc_name
        self.func_ids = call_map

        self.get_assignments()

    def get_callable(self):
        result = []
        for child in ast.walk(self.doc_node):
            if isinstance(child, ast.Call) and isinstance(child.func, ast.Attribute):
                # print("\n")
                # print("-" * 60)
                # print(f"\n<>SOURCE INPUT<>\n{ast.unparse(child)}")
                # print(f"\nINPUT NODE:\n{ast.dump(child, indent=4)}")
                found_node = self.format_call(child)
                if isinstance(found_node, ast.AST):
                    # print(f"\nFORMATTED NODE:\n{ast.dump(found_node, indent=4)}")
                    # print(f"\n<>SOURCE OUT<>\n{ast.unparse(found_node)}")
                    result.append(ast.unparse(found_node))
        return result

    def get_assignments(self):
        result = ""
        for child in ast.walk(self.doc_node):
            if isinstance(child, ast.Assign) and isinstance(child.value, ast.Call):
                # print("-" * 60)
                # print(f"\nINPUT:\n{ast.unparse(child)}")
                # print(f"\nTOP-LEVEL NODE:\n{ast.dump(child, indent=4)}\n")
                result = self.format_call(child.value)
                for name in ast.walk(result):
                    if isinstance(name, ast.Name):
                        # print(f"<>\n{name.id}\n<>")
                        func_name = name.id
                for target in child.targets:
                    if isinstance(target, ast.Name):
                        self.func_ids.update({target.id: func_name})
                        # print(f"()(){target.id}: {func_name}()()")
                # print(f"\nRESULT NODE:\n{ast.dump(result, indent=4)}")
                # print(f"\nOUTPUT:\n{ast.unparse(result)}")
                # print("^" * len(ast.unparse(result)))


    def format_call(self, call_node:ast.Call):
        # print(f"CALL TO FORMAT:\n{ast.unparse(call_node)}")
        # print(f"\nCALL NODE:\n{ast.dump(call_node, indent=4)}")
        for subnode in iter_child_nodes(call_node.func):
            if isinstance(subnode, ast.Name):
                if subnode.id in self.func_ids.keys():
                    super().generic_visit(subnode)
                    subnode.id = self.func_ids[subnode.id]
            if isinstance(subnode, ast.Call):
                super().generic_visit(subnode)
                subnode.args = []
                subnode.keywords = []
        super().generic_visit(call_node)
        call_node.args = []
        call_node.keywords = []
        return call_node


class SourceCollection:
    def __init__(self, dir_paths:list, find:str="./**/solution_index.ipynb"):
        self.search_dir = dir_paths
        self.search = find
        self.books, self.source_nodes = self._get_doc_lines()

    def _get_doc_lines(self):
        books = []
        nodes = []
        for r_dir in self.search_dir:
            os.chdir(r_dir)
            for itm in glob.iglob(self.search, recursive=True):
                doc_num = os.path.split(itm)[0][2:7]
                doc_path = os.path.abspath(itm)
                result = ""
                for line in self.notebook_parser(itm):
                    result += f"{line}\n"
                try:
                    node = ast.parse(result)
                except SyntaxError:
                    continue
                books.append((doc_num, doc_path, result))
                nodes.append((doc_num, node))
        return books, nodes

    def notebook_parser(self, file_path):
        with open(file_path, "r", encoding="utf-8") as file:
            notebook = json.load(file)
            for cell in notebook["cells"]:
                if cell["cell_type"] == "code":
                    line_gen = (line for line in cell["source"] if not line.startswith("#"))
                    for line in line_gen:
                        # line = line.rstrip()
                        # line = line.encode(encoding='ascii', errors='namereplace')
                        # line = line.decode()
                        if line.startswith("%"):
                            line = ""
                        yield line


def format_import(data:dict):
    result = f"import {data['name']}"
    if data['asname']:
        result += f" as {data['asname']}"
    if 'dual' in data.keys():
        data.update({'library': data['name']})
        sec_line = format_importfrom(data)
        result += f"\n{sec_line}"
    return result

def format_importfrom(data:dict):
    result = f"from {data['library']}"
    if 'subpackage' in data.keys():
        result += f".{data['subpackage']}"
    result += " import "
    class_num = len(data['classes'].keys()) - 1
    for itm in data['classes'].values():
        result += f"{itm['name']}"
        if class_num > 0:
            result += ", "
            class_num -= 1
    return result

def reduce_imports(data:dict):
    result = {}
    for doc in data.values():
        for lib, lib_info in doc.items():
            if lib in result:
                if lib_info['type'] == 'ImportFrom':
                    if not "classes" in result[lib].keys():
                        result[lib]['dual'] = True
                        result[lib]['classes'] = {}
                    for class_id, class_info in lib_info['classes'].items():
                        result[lib]['classes'].update({class_id: class_info})
                    continue
                result[lib].update(lib_info)
                continue
            result[lib] = lib_info
    return result

def format_header(data:dict):
    result = set()
    for import_info in data.values():
        if import_info['type'] == 'Import':
            # pp.pprint(value)
            result.add(format_import(import_info))
        if import_info['type'] == 'ImportFrom':
            # pp.pprint(value)
            result.add(format_importfrom(import_info))
    result = list(result)
    result.sort(key = lambda x : x.split()[1].lower())
    return result

def map_callables(data:dict):
    result = {}
    for info in data.values():
        if "asname" in info.keys():
            if info['asname']:
                result.update({info['asname']:info['name']})
            else:
                result.update({info['name']:info['name']})
        if "classes" in info.keys():
            for class_id, class_info in info['classes'].items():
                result.update({class_id: class_info['name']})
    return result

def main():
    """
    Main function of this automation script.
    """
    search_dir = [
        r"C:/Users/Drew Alderfer/code/flatiron/projects/phase4/labs/"
    ]
    # r"C:/Users/Drew Alderfer/code/flatiron/NYC-DS-091922/Phase_1/",
    # r"C:/Users/Drew Alderfer/code/flatiron/NYC-DS-091922/Phase_2/",
    # r"C:/Users/Drew Alderfer/code/flatiron/projects/phase2/labs/",
    # r"C:/Users/Drew Alderfer/code/flatiron/projects/phase2/tutorials/"]
    imports = {}
    importfroms = {}
    source = SourceCollection(search_dir)

    for node in source.source_nodes:
        visitor = Visitor(node[1], node[0])
        imports.update(visitor.get_imports())
        importfroms.update(visitor.get_importfrom())

    for key, value in importfroms.items():
        imports[key].update(value)

    imports = reduce_imports(imports)
    call_lookup = map_callables(imports)

    # pp.pprint(call_lookup)
    callables = set()
    call_map = []
    for number, node in source.source_nodes:
        func_visitor = FuncTransform(node, number, call_lookup)
        callables.update(func_visitor.get_callable())
        call_map.extend(func_visitor.func_ids.items())

    header = format_header(imports)
    pp.pprint(header)
    with open("C:/Users/Drew Alderfer/code/flatiron/projects/phase4/scripts/nb_imports.json", 'w') as file:
        json.dump(imports, file, indent=4)

    get_docs_fmt = []
    for func in callables:
        print(f"inpsect.signature({func[:-2]})")

if __name__ == "__main__":
    main()
