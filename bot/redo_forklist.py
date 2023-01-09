import subprocess


def git_clone(repo_url, local_dir):
    cmd = ['git', 'clone', repo_url]
    process =  subprocess.run(cmd, cwd=local_dir, check=True)
    return process

def get_line(file):
    for line in file:
        yield line

BASE_URL = "https://github.com/DrewAlderfer/"
BASE_DIR = "C:/Users/Drew Alderfer/code/flatiron/projects/phase3/"

def main():
    new_lines = []
    with open('forked_repos.txt', 'r+', encoding='UTF-8') as log:
        count = 0
        for repo_name in get_line(log):
            repo_name = repo_name.rstrip()
            lab_check = repo_name[-3:]
            if lab_check == "lab":
                cur_dir = "labs/"
            elif repo_name[-3:] == "---":
                new_lines.append(repo_name)
                continue
            else:
                cur_dir = "tutorials/"
            cwd = BASE_DIR + cur_dir
            repo_url = BASE_URL + repo_name
            # print(f"trying to clone:\n{repo_url}\nto local dir:\n{cwd}")
            job = git_clone(repo_url=repo_url, local_dir=cwd)
            if isinstance(job, subprocess.CompletedProcess):
                print(job.stdout)
            new_lines.append(repo_name + "---")
            count += 1
        print(f"Finished Clone loop after cloning {count} repos.")

    with open("./forked_repos.txt", "w") as new_file:
        new_file.writelines(new_lines)




if __name__ == '__main__':
    main()

