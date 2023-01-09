import os

from github import Github


def fork_check(new_name=None, test_list=None):
    if new_name in test_list:
        return True
    return False

def extract_repos(file):
    for row in file:
        row = row.rstrip().split(",")
        repo_info = row[-1].rstrip().split("/")
        user_name, repo_name = repo_info[-2:]
        topic_num, lesson_num = row[0:2]
        yield [topic_num, lesson_num, user_name, repo_name]


def main():

    token = os.getenv('GIT_TOKEN')
    ghub = Github(token)
    forked_list = []
    my_repos = []
    for repo in ghub.get_user().get_repos():
        my_repos.append(repo.name)

    with open("./repo_list.txt", 'r', encoding='utf-8') as file:
        forked = 0
        tries = 0
        for row in extract_repos(file):
            tries += 1
            user = row[2]
            lesson = row[3]
            new_name = f"{row[0]}-{row[1]}-{row[3]}"

            check = fork_check(new_name=new_name, test_list=my_repos)
            if check is True:
                print("You already have that one.")
                # forked_list.append(f"{new_name}---\n")        # Uncomment this line if you mess up your fork_list.txt file again
                continue
            target_repo = ghub.get_user(user).get_repo(lesson)
            new_repo = target_repo.create_fork(organization={"name":new_name})
            forked_list.append(f"{new_repo.name}\n")
            forked += 1

    with open('forked_repos.txt', 'w', encoding='UTF-8') as log:
        for x in forked_list:
            log.write(x)

    finished = print(f"completed {forked} forks of {tries} total on the list.")
    return finished




if __name__ == '__main__':
    main()

