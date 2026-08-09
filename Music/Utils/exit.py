import sys
from rich import print


def exit_application(message: str):
    print(message)
    sys.exit("close application")
