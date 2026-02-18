"""
Setup script for Brain Tumor Detection project.
Creates a Python 3.11 virtual environment and installs requirements.
"""

import subprocess
import sys
import os

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
VENV_DIR = os.path.join(PROJECT_DIR, "venv")
REQUIREMENTS = os.path.join(PROJECT_DIR, "requirements.txt")


def find_python311():
    """Try to locate a Python 3.11 interpreter."""
    candidates = ["py -3.11", "python3.11", "python"]
    for cmd in candidates:
        try:
            result = subprocess.run(
                cmd.split() + ["--version"],
                capture_output=True, text=True
            )
            if result.returncode == 0 and "3.11" in result.stdout:
                return cmd.split()
        except FileNotFoundError:
            continue
    return None


def main():
    print("=" * 60)
    print("  Brain Tumor Detection - Environment Setup")
    print("=" * 60)
    print()

    # Find Python 3.11
    print("[INFO] Looking for Python 3.11...")
    python_cmd = find_python311()
    if python_cmd is None:
        print("[ERROR] Python 3.11 not found.")
        print("        Install it from https://www.python.org/downloads/release/python-3119/")
        sys.exit(1)

    version = subprocess.run(
        python_cmd + ["--version"], capture_output=True, text=True
    ).stdout.strip()
    print(f"[OK]   Found {version}")
    print()

    # Create virtual environment
    if os.path.exists(os.path.join(VENV_DIR, "Scripts", "python.exe")):
        print("[INFO] Virtual environment already exists, skipping creation.")
    else:
        print("[INFO] Creating virtual environment...")
        result = subprocess.run(python_cmd + ["-m", "venv", VENV_DIR])
        if result.returncode != 0:
            print("[ERROR] Failed to create virtual environment.")
            sys.exit(1)
        print("[OK]   Virtual environment created at ./venv")
    print()

    # Determine python path inside venv
    venv_python = os.path.join(VENV_DIR, "Scripts", "python.exe")
    if not os.path.exists(venv_python):
        print("[ERROR] python not found in virtual environment.")
        sys.exit(1)

    # Upgrade pip
    print("[INFO] Upgrading pip...")
    subprocess.run([venv_python, "-m", "pip", "install", "--upgrade", "pip"], check=True)
    print()

    # Install requirements
    print(f"[INFO] Installing requirements from {os.path.basename(REQUIREMENTS)}...")
    result = subprocess.run([venv_python, "-m", "pip", "install", "-r", REQUIREMENTS])
    if result.returncode != 0:
        print("[ERROR] Some packages failed to install.")
        sys.exit(1)
    print()

    print("=" * 60)
    print("[OK] Setup complete! You can now run train.bat")
    print("=" * 60)


if __name__ == "__main__":
    main()
