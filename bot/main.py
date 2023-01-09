import argparse
import json
import os
import time

import tabulate
from find_repo_links import make_repo_list
from Helpers import ingest_topic, string_processor, title_maker
from githubtool import GhubCon
from local_clone import local_clone

# pyright: basic, reportUnreachable=false
def main():
    # TO DO
    # Add a verbose argument to the command line args so that you can turn
    # the debuging messages on and off.
    parser = argparse.ArgumentParser()

    parser.add_argument('--topics', type=str, default="", help='Pass topic numbers as a String')
    parser.add_argument('--phase', type=str, default="", help='Example: for Phase 3 enter: --phase "3"\nused as:\nDTSC-LIVE-091922-P{phase_num}\nwhich becomes: DTSC-LIVE-091922-P3')
    parser.add_argument('--fork', action='store_true')
    parser.add_argument('-v', '--verbose', action='store_true', default=False, help='Turn on print messages.')
    parser.add_argument('-c', '--clone', action='store_true')
    parser.add_argument('--sort_local', action='store_true')
    # parser.add_argument('--fork', type=bool)
    args = parser.parse_args()

    # FOR FUTURE TO DO REF::
    # xpath to the phase links on the dashboard:
    #
    #       //div[contains(@class, "DashboardCard")]/a[contains(., "Phase")]
    #

    # verbose = args.verbose
    sort_local = args.sort_local
    phase_num = None
    topics = None
    if not sort_local:
        phase_num = args.phase
        topics = args.topics.split()
        for ind, num in enumerate(topics):
            topics[ind] = f"{int(num):02d}"

        if len(topics) < 1:
            error_message = print(
            """You need to pass an argument with topic numbers separated by spaces in a string.
                Example:
                        --topics \"12 13 14\" 
            """)
            return error_message
        if phase_num == "":
            return print("Please try again and include a phase number in the arguments.")

    fork = args.fork
    clone = args.clone
    verbose = args.verbose
    # Check to ensure that arguments were passed

    with open("C:/Users/Drew Alderfer/code/flatiron/projects/lesson_db.json", "r") as repo_db:
        repos = json.load(repo_db)

    found_repos = repos
    if not sort_local:
        found_repos = make_repo_list(topics, phase_num)

    for itm in found_repos.values():
        try:
            repo_info = itm['upstream_repo'].rstrip().split("/")
        except KeyError:
            itm['upstream_repo'] = f"https://www.github.com/{itm['upstream_user']}/{itm['repo_url'].split('/')[-1][6:]}"
            repo_info = itm['upstream_repo'].rstrip().split("/")
        user_name, repo_name = repo_info[-2:]
        lesson_type = "tutorial"
        if "lab" in repo_name.split("-"):
            lesson_type = "lab"
        new_name = f"{itm['topic_number']}-{itm['lesson_number']}-{repo_name}"
        itm.update({"lesson_type": lesson_type, "upstream_user":user_name, "repo_name":new_name})

    for lesson, dict in found_repos.items():
        for key, value in dict.items():
            if lesson in repos.keys() and value is not None:
                repos[lesson].update({key:value})
                continue
            repos[lesson] = dict

    with open("C:/Users/Drew Alderfer/code/flatiron/projects/lesson_db.json", "w") as repo_db:
        json.dump(repos, repo_db, indent=4)

    ghub = GhubCon()
    if fork:
        repos, forked, tries = ghub.fork_repos(repo_db=repos)
        print(f"Forked {forked} projects from {tries} overall tries.")
    if clone:
        repos = ghub.local_update(repos, clone)

    with open("C:/Users/Drew Alderfer/code/flatiron/projects/lesson_db.json", "w") as repo_db:
        json.dump(repos, repo_db, indent=4)

    success = print("We did it. We got everything.")
    return success

if __name__ == '__main__':

    main()
