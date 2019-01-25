import shutil, os, sys
from detect import main

def main(directory):
    file_list = os.listdir(directory)

if __name__ == '__main__':
    dir = sys.argv[1]
    main(dir)