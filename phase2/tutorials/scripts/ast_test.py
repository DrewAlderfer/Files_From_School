import ast

class Visitor(ast.NodeVisitor):

    def visit_For(self, node: ast.AST, py_list):
        for child in ast.walk(node):
            if isinstance(child, (ast.Call, ast.Import, ast.ImportFrom)):
                py_list.append(ast.unparse(child))
        self.generic_visit(node)

def main():
    with open("./code.py", "r") as file:
        code = file.read()

    node = ast.parse(code)
    result = []
    Visitor().visit_For(node, result)
    for ind in range(len(result)):
        result[ind] += "\n\n"
    with open('./test_code.py', 'w') as log:
        log.writelines(result)

if __name__ == "__main__":
    main()
