import argparse


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument('--topics', type=str, default="", help='Pass topic numbers as a String')
    args = parser.parse_args()

    topics = args.topics.split()

    if len(topics) < 1:
        error_message = print(
        """You need to pass an argument with topic numbers separated by spaces in a string.
            Example:
                    --topics \"12 13 14\" 
        """)
        return error_message
    else: 
        print(topics)


if __name__ == '__main__':
    main()
