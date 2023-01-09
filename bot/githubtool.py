import os
from github import Github
from typing import Tuple
import subprocess
from glob import iglob
import time

from github.GithubException import GithubException

class GhubCon:
    def __init__(self):
        self.token = os.getenv('GIT_TOKEN')
        self.conn = Github(self.token)
        self.my_repos = self.get_my_repos() 


    def get_my_repos(self):
        repos = []
        for repo in self.conn.get_user().get_repos():
            repos.append(repo.name)
        return repos

    def fork_check(self, new_name:str):
        if new_name in self.my_repos:
            return True
        return False

    def fork_repos(self, repo_db:dict) -> Tuple[dict, int, int]:
        """Pass in a dictionary describing repos to fork and this function will check them against 
            current repos, then fork them if they don't already exist."""
        forked = 0
        tries = 0
        for lesson_key, dict in repo_db.items(): 
            tries += 1
            user = dict['upstream_user']
            lesson = dict['repo_name'][6:]
            new_name = dict['repo_name']

            check = self.fork_check(new_name)
            if check:
                dict.update({"repo_url": f"http://www.github.com/DrewAlderfer/{new_name}", "forked":True})
                tries += 1
                continue
            print(f"forking {new_name}")
            target_repo = self.conn.get_user(user).get_repo(lesson)
            try:
                new_repo = target_repo.create_fork(organization={"name":new_name})
                time.sleep(1)
                if new_repo:
                    dict.update({"repo_url": f"http://www.github.com/DrewAlderfer/{new_name}", "forked":True})
                    forked += 1
            except GithubException:
                time.sleep(5)
                new_repo = target_repo.create_fork(organization={"name":new_name})
                if new_repo:
                    dict.update({"repo_url": f"http://www.github.com/DrewAlderfer/{new_name}", "forked":True})
                    forked += 1

        return repo_db, forked, tries

    def git_clone(self, repo_url, local_dir):
        cmd = ['git', 'clone', repo_url]
        process =  subprocess.run(cmd, cwd=local_dir, capture_output=True, check=True)
        return process

    def delete_repo(self, user, repo_name):
        self.conn.get_user(user).get_repo(repo_name).delete()

    def local_update(self, repo_db:dict, clone:bool=False):
        """
        Take a dictionary database of school assignments and then clones the ones that are not
        cloned locally.
        """

        # print(json.dumps(repo_db, indent=4))

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
                dict.update({'local_dir': local_info[lesson]['local_dir'], 'cloned': True})
                continue

            if clone:
                # print(f"cloning lesson: {dict['topic_number']}-{dict['lesson_number']}\nfrom repo: {dict['repo_url']}")
                phase = dict['phase']
                lesson_type = dict['lesson_type']
                cwd = f"{BASE_DIR}{phase}/{lesson_type}s/"
                if not os.path.exists(cwd):
                    os.mkdir(cwd)
                try:
                    job = self.git_clone(repo_url=dict['repo_url'], local_dir=cwd)
                    if isinstance(job, subprocess.CompletedProcess):
                        print(job.stdout.decode())
                except subprocess.CalledProcessError as error:
                    print(error.stderr)
                    continue


        return repo_db
