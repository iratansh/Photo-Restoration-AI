"""
Installation script for the photo restoration system
"""
import subprocess
import sys

def check_python_version():
    if not (sys.version_info.major == 3 and sys.version_info.minor in [8, 9]):
        raise SystemError("This project requires Python 3.8 or 3.9")

def install_requirements():
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])

def setup():
    check_python_version()
    install_requirements()
    print("Setup completed successfully!")

if __name__ == "__main__":
    setup()