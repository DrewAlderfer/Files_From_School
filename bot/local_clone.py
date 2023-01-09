import os
import subprocess
import json
from glob import iglob


def git_clone(repo_url, local_dir):
    cmd = ['git', 'clone', repo_url]
    process =  subprocess.run(cmd, cwd=local_dir, check=True)
    return process

def local_clone(repo_db:dict, clone:bool=False):
    """Take a dictionary database of school assignments and then clones the ones that are not cloned locally."""
    print(json.dumps(repo_db, indent=4))

    BASE_DIR = os.getenv('FLATIRON_BASE_DIR')
    local_lessons = []
    local_info = {}
    for itm in iglob(f"{BASE_DIR}/**/**/**/"):
        parts = itm.split('\\')
        phase = ""
        local_lesson = ""
        for part in parts:
            if 'phase' in part:
                phase = part
            if '-dsc-' in part:
                local_lesson = part[:5].replace("-", "_")
        if phase and local_lesson:
            local_lessons.append(local_lesson)
            if local_lesson in local_info.keys():
                local_info[local_lesson].update({'local_dir':itm, 'cloned':True})
                continue
            local_info[local_lesson] = {'local_dir':itm, 'cloned':True}

    for lesson, dict in repo_db.items():
        if lesson in local_lessons:
            dict.update(local_info[lesson])
            continue

        if clone:
            print(f"cloning lesson: {dict['topic_num']}-{dict['lesson_num']}\nfrom repo: {dict['repo_url']}")
            phase = dict['phase'] 
            cwd = f"{BASE_DIR}{phase}/{type}s/"
            print(cwd)
            job = git_clone(repo_url=dict['repo_url'], local_dir=cwd)
            if isinstance(job, subprocess.CompletedProcess):
                print(job.stdout)
              

    return repo_db

    
