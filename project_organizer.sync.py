# %%
import os
import json
import shutil
from glob import iglob
import time
from typing import Tuple, Union, List
import pprint as pp

from github.GithubException import UnknownObjectException

from bot.githubtool import GhubCon

# %%
# %load_ext autoreload

# %%
# %autoreload 2
# aimport bot.githubtool

# %%
pp.PrettyPrinter(indent=4)

# %%
gh = GhubCon()

# %%
gh_repos = gh.my_repos

# %%
def clean_up_repos(github_repos):
    """
    Looks for repos matching ^[0-9]{2}-[0-9]{2} in github repos and then deletes them.

    Returns:
        A list of repos that matched the pattern but where not deleted for whatever reason.
    """
    target_repos = [x for x in github_repos if x[:2].isdigit() and x[3:4].isdigit()]
    result = target_repos.copy()
    for repo in target_repos:
        time.sleep(.05)
        try:
            gh.conn.get_user('DrewAlderfer').get_repo(repo).delete()
            print(f"deleting {repo}")
            result.remove(repo)
        except UnknownObjectException:
            continue

    return result

# %%
def match_local_with_db(github:GhubCon, module_range:Tuple[int, int]=(0, 0)) -> None:
    """
    Grabs information from the lesson database and forks the repos for those lessons, clones them
    to the local directory and then deletes them from github.

    returns:
        None
    """


    with open("../lesson_db.json", "r") as file:
        repo_db = json.load(file)

    lo_num, hi_num = module_range
    target_repos = {k: v for k, v in repo_db.items() if int(k[:2]) > lo_num and int(k[:2]) < hi_num}

    # fork the target repos
    repos, _, _ = github.fork_repos(repo_db=target_repos)
    # clone the target repos to the local projects folder
    repos = github.local_update(repos, True)
    # clean up github repos
    for repo in target_repos.values():
        time.sleep(.05)
        try:
            gh.conn.get_user('DrewAlderfer').get_repo(repo['repo_name']).delete()
            print(f"deleting {repo['repo_name']}")
        except UnknownObjectException:
            continue


