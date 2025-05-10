import getpass
import os
import re
import shutil
from enum import Enum

import requests
from mapper import check_key
from mapper.neighborgraph import DEFAULT_KEY

from cobalt import check_license
from cobalt.config import (
    CONFIG_FILE_PATH,
    LAMBDA_VERSION,
    REGISTER_API_URL,
    is_colab_environment,
    is_vscode,
    load_config,
    save_config,
)


class EventType(Enum):
    REGISTER = "registration"
    VERIFY = "verification"


class RegisterType(Enum):
    NONCOMMERCIAL = "n"
    TRIAL = "t"


def register_license():
    """Registers this installation of Cobalt for noncommercial or trial usage.

    Requests your name and email address and configures a license key. If you
    have already registered Cobalt on a different computer, this will link your
    computer with the previous registration.
    """
    print("Thank you for registering this copy of BluelightAI Cobalt!")
    config = load_config()
    hint_for_input = (
        "\nInput field is at the top of the window \u2191" if is_vscode() else ""
    )
    print(
        "Enter 'n' to register for noncommercial use, "
        f"or 't' to register for commercial trial use:{hint_for_input}"
    )
    register_type = input("> ").strip()
    if register_type == RegisterType.NONCOMMERCIAL.value:
        print(
            "Please enter your name and email address to register this copy for noncommercial use."
        )
        result = register_by_license_type(RegisterType.NONCOMMERCIAL.value)

    elif register_type == RegisterType.TRIAL.value:
        print(
            "Please enter your name, email address, and company to register and begin your trial."
        )
        result = register_by_license_type(RegisterType.TRIAL.value)
    else:
        result = {}
        print(
            f"Error: Invalid registration type {register_type}! "
            f"Please run cobalt.register_license() again."
        )
        config["config"]["license_type"] = LicenseType.UNREGISTERED.value

    if result.get("message"):
        data = result["message"]["data"]["attributes"]
        license_key = data["key"]
        config["license_key"] = license_key
        info_to_update = {
            "name": data["metadata"].get("name"),
            "email": data["metadata"].get("email"),
            "company": data["metadata"].get("company"),
            "license_type": data["metadata"].get("licenseType"),
        }
        config["config"].update(info_to_update)
    if result.get("error"):
        config["config"]["license_type"] = LicenseType.UNREGISTERED.value
    save_config(config)
    check_license()


def register_by_license_type(license_type):
    payload = {}
    result = {}
    error_email_msg = (
        "Invalid email address. Please run cobalt.register_license() again."
    )

    if license_type == RegisterType.NONCOMMERCIAL.value:
        name = input("Name: ")
        email = input("Email: ")
        if not is_valid_email(email):
            print(error_email_msg)
            result["error"] = error_email_msg
            return result

        payload["license_type"] = LicenseType.NONCOMMERCIAL.value
        payload["name"] = name
        payload["email"] = email
        success_msg = "Your noncommercial license is now registered."
    else:
        name = input("Name: ")
        email = input("Email: ")
        if not is_valid_email(email):
            print(error_email_msg)
            result["error"] = error_email_msg
            return result
        company = input("Company: ")
        payload["license_type"] = LicenseType.TRIAL.value
        payload["name"] = name
        payload["email"] = email
        payload["company"] = company
        success_msg = "Your trial is now registered."

    result = register_cobalt(payload)
    if result["error"]:
        print(f"Registration failed due to an error: {result['error']}")
    else:
        print(success_msg)
    return result


def is_valid_email(email):
    email_pattern = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
    return re.match(email_pattern, email) is not None


def setup_license():
    """Prompts for a license key and sets it in the configuration file.

    The license key will be saved in ~/.config/cobalt/cobalt.json.
    """
    print("Welcome to the Cobalt License Setup Wizard!")
    if is_vscode():
        print(
            "Please enter your license key to the input field at the top of the window \u2191"
        )
    # Online license setup
    valid = False
    while not valid:
        license_key = getpass.getpass("Please enter your license key: ").strip()
        if not license_key:
            print("Error: License key can not be empty.")
        elif license_key == DEFAULT_KEY:
            print("Error: The license key cannot be the same as the trial key.")
        else:
            valid = True

    setup_key(license_key)
    print("License key successfully set up!")
    print("validating license key...")
    check_license()


def setup_license_offline():
    license_key = getpass.getpass("Please enter your license key: ").strip()
    # Offline license setup via license file
    license_file = input(
        "Please enter the path to your license file (leave empty for the current directory): "
    ).strip()
    if not license_file:
        license_file = os.path.join(os.getcwd(), "license.lic")
        print("No path provided. Using default license file: license.lic")

    # Check if the file exists and validate
    if os.path.isfile(license_file):
        if license_file_not_empty(license_file):
            try:
                setup_key(license_key)

                config_dir = os.path.dirname(CONFIG_FILE_PATH)
                shutil.copy(license_file, config_dir)
                print("validating license file...")
                check_license()
            except Exception as e:
                print(f"Failed to set up offline license: {e}")
            print("License successfully set up using the license file!")
        else:
            print("Invalid license file content")
    else:
        print(
            f"License file not found at '{license_file}'. Please check the path and try again."
        )


def license_file_not_empty(license_file_path):
    try:
        with open(license_file_path) as file:
            license_data = file.read().strip()
            return bool(license_data)
    except Exception as e:
        print(f"Error reading license file: {e}")
        return False


def setup_key(license_key):
    config = load_config()
    config["license_key"] = license_key
    updated_config, _ = check_key(config, setup=True)
    save_config(updated_config)


class LicenseType(Enum):
    UNREGISTERED = "unregistered"
    TRIAL = "trial"
    NONCOMMERCIAL = "noncommercial"


def register_cobalt(payload):
    payload["event_type"] = EventType.VERIFY.value
    result = send_request(payload)
    if result["error"]:
        return result

    print(
        f"Thank you. We've sent a verification code to your email: {payload['email']}. "
        f"Please check your email and input your verification code here:"
    )
    code = input("> ").strip()
    payload["code"] = code
    payload["event_type"] = EventType.REGISTER.value
    result = send_request(payload)
    return result


def send_request(payload):
    result = {"error": None, "message": None}
    headers = {
        "x-cobalt-license-request": "true",
        "x-cobalt-app-version": str(LAMBDA_VERSION),
    }
    response = requests.post(
        f"{REGISTER_API_URL}/create-license", data=payload, headers=headers
    )

    if response.status_code != 200:
        result["error"] = response.json()
    else:
        message = response.json()
        if message.get("error"):
            result["error"] = message["error"]
        else:
            result["message"] = message
    return result


def check_license_type():
    config = load_config()

    if os.getenv("COBALT_LICENSE_KEY"):
        show_message = False
        updated_config = config

    elif (
        not config["license_key"]
        or not config["config"]
        or config["config"]["license_type"] == LicenseType.UNREGISTERED.value
    ):
        show_message = True
        updated_config = config
    else:
        updated_config, show_message = check_key(config)

    if show_message:
        updated_config["config"]["license_type"] = "unregistered"
        save_config(updated_config)
        if not is_colab_environment():
            print(
                "Thank you for using BluelightAI Cobalt!\n"
                "This version is licensed for noncommercial and trial use.\n"
                "Please register by running cobalt.register_license()"
            )
    else:
        save_config(updated_config)
