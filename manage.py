#!/usr/bin/env python3

import sys
import os

def install_dependencies():
    os.system("python3 -m venv .venv")
    os.system("source .venv/bin/activate")
    os.system("pip install -r requirements.txt")

def download_dataset():
    os.system("python3 src/dataset/download.py")

def setup():
    install_dependencies()
    download_dataset()

def help():
    print("Possible commands:")
    print(" python3 manage.py i_dep        # Install project dependencies")
    print(" python3 manage.py down_data    # Download project dataset")
    print(" python3 manage.py setup        # Setup project")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        help()
    else:
        cmd = sys.argv[1]
        match(cmd):
            case "i_dep": install_dependencies()
            case "down_data": download_dataset()
            case "setup": setup()
            case "help": help()
            case _: help()
