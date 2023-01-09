import os
from glob import iglob
import shutil
import subprocess
from githubtool import GhubCon

def onerror(func, path, exc_info):
    """
    Error handler for ``shutil.rmtree``.

    If the error is due to an access error (read only file)
    it attempts to add write permission and then retries.

    If the error is for another reason it re-raises the error.

    Usage : ``shutil.rmtree(path, onerror=onerror)``
    """
    import stat
    # Is the error an access error?
    if not os.access(path, os.W_OK):
        os.chmod(path, stat.S_IWUSR)
        func(path)
    else:
        raise

def construct_entries(phase):
    result = []
    folders = ['labs', 'tutorials']
    for folder in folders:
        for lesson in iglob(phase + f"/{folder}/**"):
            if not os.path.basename(lesson)[:2].isnumeric():
                continue
            lesson_name = " ".join(os.path.basename(lesson)
                                   .replace('dsc', '')[7:-3]
                                   .split('-')).title()
            key = f"{os.path.basename(lesson)[:2]}_{os.path.basename(lesson)[3:5]}"
            entry = {
                    key:{
                        'phase': os.path.basename(phase),
                        'topic_number': os.path.basename(lesson)[:2],
                        'lesson_number': os.path.basename(lesson)[3:5],
                        'lesson_name': lesson_name,
                        'lesson_url': None,
                        "upstream_repo": None,
                        "lesson_type": None,
                        "upstream_user": "learn-co-curriculum",
                        "repo_name": os.path.basename(lesson),
                        "repo_url": f"http://www.github.com/DrewAlderfer/{os.path.basename(lesson)}",
                        "forked": True
                        }
                    }

            result.append(entry)
    return result

def rm_dir(dir_path, phase):
    cmd = ['rm', '-Recurse', '-Force', './' + os.path.basename(dir_path)]
    print(cmd)
    process = subprocess.run(cmd, cwd=phase, check=True)
    return process

def clean_and_update(project_dir:str="C:/Users/Drew Alderfer/code/flatiron/projects", clean:bool=False):
    github = GhubCon()
    repos = github.get_my_repos()
    local_path = project_dir
    result = []

    for phase in iglob(f"{local_path}/phase*"):
        if os.path.exists(phase + "/labs"):
            result.extend(construct_entries(phase))
            continue
        os.chdir(phase)
        
        if clean:
            for lesson in iglob(phase + "/**"):
                print(f"attempting to remove: {os.path.basename(lesson)}")
                if lesson in repos:
                    github.delete_repo('DrewAlderfer', os.path.basename(lesson))
                shutil.rmtree('./' + os.path.basename(lesson), onerror=onerror)
                if not os.path.exists(lesson):
                    print(f"Sucessfully deleted: {os.path.basename(lesson)}")

    return result

def local_update_db(clean:bool=False):
    update = clean_and_update(clean=clean)
    with open("C:/Users/Drew Alderfer/code/webbot/repo_db.json", "r+") as file:
        repos = json.load(file)

        for lesson, info in update.values():
             
            for key, val in entry.items():
                if not val:
                    continue

    
