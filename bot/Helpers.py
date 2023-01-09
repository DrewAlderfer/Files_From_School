from selenium.common.exceptions import NoSuchElementException

def title_maker(string_list):
    """takes a list of strings, reformats them as titles and then returns a list of tuples."""

    result = []
    topic_num = 0 
    lesson_num = 0  

    for ind, itm in enumerate(string_list):
        if "Topic" in itm:
            words = itm.split()
            topic_num = words[1]
            lesson_num = 0
        itm = itm.replace("  ", " ")
        # itm = "-".join(itm.split())
        result.append((f"{int(topic_num):02d}-{int(lesson_num):02d}", itm))
        lesson_num += 1
    return result

def string_processor(link_list):
    """Takes a list of links and returns a cleaned list of topic titles."""    

    result = []

    for elem in link_list:
        link_text = elem.get_attribute('innerHTML')

        if len(link_text) > 0:
            try:
                link_text.encode('ascii')
            except UnicodeEncodeError:
                phrase = link_text.split()
                new_phrase = []
                for word in phrase:
                    try:
                        word = word.encode(encoding='ascii', errors='namereplace')
                        word = str(word, encoding='UTF-8', errors='strict')
                    except UnicodeEncodeError:
                        pass
                    else:
                        if not word.startswith('\\N'):
                            new_phrase.append(word)
                result.append(" ".join(new_phrase))
            else:
                result.append(link_text.strip())
                
    return result


def ingest_topic(topic_list, topic, driver):
    """"""
    result = []
    topic_pages =  (x for x in topic_list if (x[0][:2] == topic)) 
    # print(f"topic_pages:\n{topic_pages}")
    for (id, page) in topic_pages:
        link = ""
        try:
            link = driver.find_element_by_link_text(page).get_attribute('href')
        except NoSuchElementException:
            repair = page.split()
            page = " ".join(repair)
            try:
                link = driver.find_element_by_link_text(page).get_attribute('href')
            except NoSuchElementException:
                link = "*** Caught an Error ***"
            finally:
                result.append((id, page, link))
        else:
            result.append((id, page, link))

    return result

