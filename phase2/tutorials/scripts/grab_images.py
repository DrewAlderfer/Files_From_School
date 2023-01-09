import os
from shutil import copy2, copytree, SameFileError

def main():
    base = "C:/Users/Drew Alderfer/code/flatiron/projects/phase3/tutorials"
    top_dir = [f"{base}/{name}" for name in os.listdir(base) if name[:2].isdigit()]
    home = f"{base}/home"
    img_dir = f"{home}/images"
    os.makedirs(img_dir, exist_ok=True)
    for dir in top_dir:
        print(dir)
    count = 0
    copied = 0
    for img in top_dir:
        topic = img 
        target_dir = f"{img}/images"
        try:
            grab_files = os.listdir(target_dir)
        except FileNotFoundError:
            print(f"Didn't find a images directory in Topic {topic} @ {target_dir}.")
            count += 1
        else:
            # image_dest = [folder for folder in top_dir if folder.startswith(f"../{topic}-00-")][0]
            # image_dest = image_dest + "/images"
            image_dest = img_dir
            if os.path.isdir(image_dest) is False:
                os.mkdir(image_dest)
            count += 1
            for itm in grab_files:
                print(f"{itm} going to : {image_dest}")
                if os.path.isdir(f"{target_dir}/{itm}"):
                    try:
                        copytree(src=f"{target_dir}/{itm}", dst=image_dest)
                    except FileExistsError:
                        pass
                else:
                    try:
                        copy2(src=f"{target_dir}/{itm}", dst=image_dest)
                    except SameFileError:
                        continue
                copied += 1

    out_message = print(f"Copied {copied} files from {count} folders.\n\nThe end.")
    return out_message

if __name__ == '__main__':
    main()
