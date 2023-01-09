import os
import time

from selenium import webdriver
from selenium.common.exceptions import NoSuchElementException
from selenium.common.exceptions import InvalidArgumentException
from webdriver_manager.chrome import ChromeDriverManager

from Helpers import ingest_topic, string_processor, title_maker

def make_repo_list(topics, phase_num):
    result = {} 
    with webdriver.Chrome(executable_path=ChromeDriverManager().install()) as driver:
        # Go to flatiron canvas site
        driver.get("https://learning.flatironschool.com")
        driver.find_element_by_partial_link_text("Log In").click()

        # define the phase number to be returned in the dictionary entries
        phase = f"phase{phase_num}"

        # Grab Username and Password from local env variables
        can_uname = os.getenv('canvas_user')
        can_passw = os.getenv('canvas_pass')
        # Login to the dashboard
        driver.find_element_by_id('user-email').send_keys(can_uname)
        driver.find_elements_by_name("user[password]")[0].send_keys(can_passw)
        driver.find_element_by_name("commit").click()

        # Wait for the page to fully load
        time.sleep(1)

        # Select the course and list of modules
        # TO DO:
        #       Make this part take a command line argument so that you can assign
        #       a course identifier. Rn, it will only work with the phase 2 cousre.
        driver.find_element_by_xpath(f'//div[contains(@class, "DashboardCard")]/a[contains(., "Phase {phase_num}")]').click()
        time.sleep(1)
        driver.find_element_by_xpath("//a[@class='modules']").click()
        time.sleep(1)
        lesson_links = driver.find_elements_by_xpath("//a[@class='ig-title title item_link']")

        # Now that I have the page link objects I send them out to get cleaned up
        # and some logic to give them 'ID numbers' based on the topic number and
        # their order in that topic. This is important to pass to GitHub when I 
        # fork the repos and makes organizing and navigating the folders much
        # easier on my local machine.
        links = string_processor(lesson_links)
        # print(f"printing links:\n\n{links}\n\n")
        lessons = title_maker(links)
        topic_info = []
        # ingest_topic takes in the topic numbers and the list of links and pairs
        # each with its url.
        for topic in topics:
            # print(f"topic to be ingested: {topic}\n")
            topic_info.append(ingest_topic(lessons, topic, driver))
            # print("-" * 100)
        # print(f"printing topic_info:\n\n{topic_info}")
        # Now I am going to unpack my topic info tuple list and fork each repo in
        # succession.

        # Set Log-in status to False so that the driver will log-in the first loop
        # but after should remain logged in.
        print(f"Formatting topic data for: {len(topic_info)} topics.")
        for topic in topic_info:
            count = 0
            errors = []
            # print(topic)
            print(f"Starting Topic {topic[0][0][:2]}:")

            for (page_id, page_name, page_url) in topic:
                print("[]", end="")
                try:
                    driver.get(page_url)
                except InvalidArgumentException:
                    errors.append("Unable to get the Url at:")
                    errors.append(f"{page_url}\n From {page_id}{page_name}")
                    count += 1
                    continue
                try:
                    driver.find_element_by_xpath('//a[img[@id="repo-img"]]').get_attribute('href')
                except NoSuchElementException:
                    errors.append(f"Didn't find a repo for {page_name}")
                    count += 1
                    continue
                else:
                    url = driver.find_element_by_xpath('//a[img[@id="repo-img"]]').get_attribute('href')
                    result.update({
                        f"{page_id[:2]}_{count:02}": {
                                'phase': phase,
                                'topic_number':page_id[:2],
                                'lesson_number':f"{count:02}",
                                'lesson_name':page_name,
                                'lesson_url':page_url,
                                "lesson_type": None,
                                'upstream_repo':url,
                                "upstream_user": None,
                                "repo_name": None,
                                "forked": False,
                                "local_dir": None,
                                "cloned": False
                                }
                            })
                    count += 1
            print(f"\nCaught {len(errors)} errors:")
            for error in errors:
                print(f"\n{error}")
    return result
