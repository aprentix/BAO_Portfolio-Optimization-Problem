#!/usr/bin/env python3

import sys
import os
import platform
import subprocess
import shutil
import argparse


def get_python_command():
    python_cmd = sys.executable
    if not python_cmd:
        # Try to find python command
        if shutil.which("python3"):
            python_cmd = "python3"
        elif shutil.which("python"):
            python_cmd = "python"
        else:
            raise RuntimeError("Could not find Python executable")
    return python_cmd


def get_pip_command(venv_path=None):
    if venv_path:
        if platform.system() == "Windows":
            pip_paths = [
                os.path.join(venv_path, "Scripts", "pip.exe"),
                os.path.join(venv_path, "Scripts", "pip3.exe")
            ]
        else:
            pip_paths = [
                os.path.join(venv_path, "bin", "pip"),
                os.path.join(venv_path, "bin", "pip3")
            ]

        for pip_path in pip_paths:
            if os.path.exists(pip_path):
                return pip_path

        # If direct path not found, try using python -m pip
        python_path = get_venv_python_path(venv_path)
        return [python_path, "-m", "pip"]
    else:
        # Check system pip
        if shutil.which("pip3"):
            return "pip3"
        elif shutil.which("pip"):
            return "pip"
        else:
            # Use python -m pip as fallback
            return [get_python_command(), "-m", "pip"]


def get_venv_python_path(venv_path):
    if platform.system() == "Windows":
        return os.path.join(venv_path, "Scripts", "python.exe")
    else:
        return os.path.join(venv_path, "bin", "python")


def install_dependencies():
    python_cmd = get_python_command()
    venv_path = ".venv"

    # Create virtual environment
    subprocess.run([python_cmd, "-m", "venv", venv_path], check=True)

    # Get pip command inside virtual environment
    pip_cmd = get_pip_command(venv_path)

    # Install requirements
    if isinstance(pip_cmd, list):
        subprocess.run(pip_cmd + ["install", "-r",
                       "requirements.txt"], check=True)
    else:
        if platform.system() == "Windows":
            subprocess.run([pip_cmd, "install", "-r",
                           "requirements.txt"], check=True)
        else:
            subprocess.run(
                ["bash", "-c", f"source {venv_path}/bin/activate && {pip_cmd} install -r requirements.txt"], check=True)

    print('[-] Please activate entrainment with')


def download_dataset():
    venv_python = get_venv_python_path(".venv")
    subprocess.run([venv_python, "src/dataset/download.py"], check=True)


def setup():
    install_dependencies()
    download_dataset()


def main():
    python_cmd = "python" if platform.system() == "Windows" else get_python_command()

    parser = argparse.ArgumentParser(
        description="Project manager script.",
        epilog=f"Example: {python_cmd} --setup"
    )

    parser.add_argument(
        '-s', '--setup', action="store_true", help="Setup project (dependencies + dataset)")
    parser.add_argument('-i', '--i-depend', action="store_true",
                        help="Install project dependencies")
    parser.add_argument('-d', '--down-data', action="store_true",
                        help="Download project dataset")

    args = parser.parse_args()

    if args.setup:
        setup()
    elif args.i_depend:
        install_dependencies()
    elif args.down_data:
        download_dataset()


if __name__ == "__main__":
    main()
