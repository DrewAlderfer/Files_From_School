import os
import json

def print_notebook(file_path):
    file_path = f"{file_path}/solution_index.ipynb"
    try:
        with open(file_path, 'r', encoding='UTF-8') as file:
            data = json.load(file)
    except FileNotFoundError:
        error = f"Didn't find an index.ipynb in {file_path}."
        return error
    else:
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

def main():
    base = "C:/Users/Drew Alderfer/code/flatiron/projects/phase3/labs"
    topics = glob("C:/Users/Drew Alderfer/code/flatiron/projects/phase4/tutorials/**")
    for num in topics:
        dirs = [f"{base}/{name}" for name in os.listdir(base) if name.startswith(num)]
        home = f"{base}/home/"
        os.makedirs(home, exist_ok=True)
        tar_file = f"{home}/{num}index.md"
        lines = []
        for topic in dirs:
            print(topic)
            for line in print_notebook(topic):
                lines.append(line)
                lines.append("\n")
            lines.append("\n-----File-Boundary-----\n")

        with open(tar_file, 'w', encoding='UTF-8') as index:
            index.writelines(lines)

if __name__ == '__main__':
    main()
