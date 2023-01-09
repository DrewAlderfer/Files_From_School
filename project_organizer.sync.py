# %%
import os
import json
import shutil
from glob import iglob
from github import Github
import pprint as pp

# %%
pp.PrettyPrinter(indent=4)

# %%
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
gh = GhubCon()

# %%
print(len(gh.my_repos))

# %%
with open("C:/Users/Drew Alderfer/code/webbot/repo_db.json", "r") as file:
    repo_db = json.load(file)
p_r = list(repo_db.values())
for ind in range(5):
    pp.pprint(p_r[ind])

# %%
records = []
for val in repo_db.values():
    records.append(val['repo_name'])

def find_mismatches(list1, list2):
    result = []
    record_nums, record_names = [[x[:6] for x in list1 if x not in list2],
                                [x[6:] for x in list1 if x not in list2]]
    repo_nums, repos = [[x[:6] for x in list2 if x not in list1],
                        [x[6:] for x in list2 if x not in list1]]

    for i, name in enumerate(repos):
        if name in record_names:
            idx = record_names.index(name)
            result.append((record_nums[idx], repo_nums[i], name))

    return result

# %%
new_records = find_mismatches(records, gh.my_repos)
new_records

# %%
delete_list = [f"{x[1]}{x[2]}" for x in new_records]
for repo in delete_list:
    gh.conn.get_user('DrewAlderfer').get_repo(repo).delete()

# %%
count = 0
for record in new_records:
    num = record[0][:-1].replace('-', '_')
    repo_db[num]['forked'] = False
    repo_db[num]['cloned'] = False
    if count < 10:
        pp.pprint(repo_db[num])
    count += 1

# %%
with open("C:/Users/Drew Alderfer/code/webbot/repo_db.json", "w") as db:
    old_repos.update(repo_db)
    json.dump(old_repos, db, indent=4)
