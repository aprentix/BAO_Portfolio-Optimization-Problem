#!/usr/bin/env python3

import sys
import os
import platform
import subprocess
import shutil
import argparse


def get_python_command():
    """Gets the Python command available in the system."""
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
    """Gets the pip command as a list for consistent usage."""
    if venv_path:
        if platform.system() == "Windows":
            python_path = os.path.join(venv_path, "Scripts", "python.exe")
        else:
            python_path = os.path.join(venv_path, "bin", "python")

        # Verify that the Python executable exists
        if not os.path.exists(python_path):
            raise RuntimeError(
                f"Python not found in virtual environment: {python_path}")

        return [python_path, "-m", "pip"]
    else:
        # Use python -m pip as standard
        return [get_python_command(), "-m", "pip"]


def get_venv_python_path(venv_path):
    """Gets the path to the Python executable inside the virtual environment."""
    if platform.system() == "Windows":
        return os.path.join(venv_path, "Scripts", "python.exe")
    else:
        return os.path.join(venv_path, "bin", "python")


def check_requirements_file():
    """Verifies that the requirements.txt file exists."""
    if not os.path.exists("requirements.txt"):
        raise FileNotFoundError(
            "requirements.txt file not found in the current directory")


def check_dataset_script():
    """Verifies that the dataset download script exists."""
    dataset_script = os.path.join("pop", "dataset", "download.py")
    if not os.path.exists(dataset_script):
        raise FileNotFoundError(
            f"Dataset download script not found: {dataset_script}")


def install_dependencies():
    """Installs project dependencies in a virtual environment."""
    try:
        # Verify requirements.txt
        check_requirements_file()

        python_cmd = get_python_command()
        venv_path = ".venv"

        # Create virtual environment if it doesn't exist
        if not os.path.exists(venv_path):
            print(f"[+] Creating virtual environment in {venv_path}...")
            subprocess.run([python_cmd, "-m", "venv", venv_path], check=True)
        else:
            print(f"[*] Using existing virtual environment in {venv_path}")

        # Get pip command inside the virtual environment
        pip_cmd = get_pip_command(venv_path)

        # Install requirements
        print("[+] Installing dependencies from requirements.txt...")
        subprocess.run(pip_cmd + ["install", "-r",
                       "requirements.txt"], check=True)

        print(
            f"[-] Please activate the virtual environment with: {'".venv\\Scripts\\activate"' if platform.system() == 'Windows' else 'source .venv/bin/activate'}")

        return True
    except subprocess.CalledProcessError as e:
        print(f"[!] Error installing dependencies: {e}")
        return False
    except FileNotFoundError as e:
        print(f"[!] Error: {e}")
        return False
    except Exception as e:
        print(f"[!] Unexpected error: {e}")
        return False


def download_dataset():
    """Downloads the project dataset."""
    try:
        # Verify dataset script
        check_dataset_script()

        venv_python = get_venv_python_path(".venv")
        if not os.path.exists(venv_python):
            print(
                "[!] Virtual environment not found. Please run '--setup' or '--i-depend' first")
            return False

        print("[+] Downloading dataset...")
        subprocess.run([venv_python, "pop/dataset/download.py"], check=True)
        print("[+] Dataset successfully downloaded")
        return True
    except subprocess.CalledProcessError as e:
        print(f"[!] Error downloading dataset: {e}")
        return False
    except FileNotFoundError as e:
        print(f"[!] Error: {e}")
        return False
    except Exception as e:
        print(f"[!] Unexpected error: {e}")
        return False


def setup():
    """Sets up the complete project (dependencies + dataset)."""
    if install_dependencies():
        download_dataset()


def main():
    """Main function that handles command line arguments."""

    parser = argparse.ArgumentParser(
        description="Project management script.",
        epilog="Example: %(prog)s --help"
    )

    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        '-s', '--setup', action="store_true", help="Setup project (dependencies + dataset)")
    group.add_argument('-i', '--i-depend', action="store_true",
                       help="Install project dependencies")
    group.add_argument('-d', '--down-data', action="store_true",
                       help="Download project dataset")

    args = parser.parse_args()

    if args.setup:
        setup()
    elif args.i_depend:
        install_dependencies()
    elif args.down_data:
        download_dataset()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
