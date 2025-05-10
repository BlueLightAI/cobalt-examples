import argparse

from cobalt.config import check_license
from cobalt.set_license import setup_license

parser = argparse.ArgumentParser(prog="cobalt")
parser.add_argument("command", choices=["setup-license", "check-license"])
args = parser.parse_args()


if args.command == "setup-license":
    setup_license()
elif args.command == "check-license":
    print("Checking license...")
    check_license()
