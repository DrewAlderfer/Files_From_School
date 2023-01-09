# %%
import os
import json
import shutil
from glob import iglob
import time
import pprint as pp

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
print(len(gh_repos))
gh_repos

# %%
with open("C:/Users/Drew Alderfer/code/flatiron/projects/lesson_db.json", "r") as file:
    repo_db = json.load(file)

# %%
target_repos = {k: v for k, v in repo_db.items() if int(k[:2]) < 15 and int(k[:2]) > 10}
print(len(target_repos))
print(list(target_repos.values())[-1])


# %%
repos, forked, tries = gh.fork_repos(repo_db=target_repos)
print(f"Forked {forked} projects from {tries} overall tries.")
repos = gh.local_update(repos, True)

# %%
for repo in repo_names:
    print(f"deleting {repo}")
    gh.conn.get_user('DrewAlderfer').get_repo(repo).delete()
    time.sleep(.05)

# %%
keys = list(repo_db.keys())
keys.sort()
sorted_db = {}
for key in keys:
    sorted_db.update({f"{key}":repo_db[key]})

# %%
for i in range(10):
    pp.pprint(list(sorted_db.items())[i])


