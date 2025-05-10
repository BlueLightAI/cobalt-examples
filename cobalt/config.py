# Copyright (C) 2023-2024 BlueLightAI, Inc. All Rights Reserved.
#
# Any use, distribution, or modification of this software is subject to the
# terms of the BlueLightAI Software License Agreement.

import inspect
import json
import logging
import os
import sys
from dataclasses import dataclass
from datetime import date, datetime
from enum import Enum
from typing import Callable

import ipynbname
import mapper
from dotenv import load_dotenv
from IPython.display import Javascript, clear_output, display

load_dotenv()

current_folder = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.split(current_folder)[0]  # path to the root folder of the project

# Path to logs folder.
# Can be set as environment variable or in `.env` file.
# If path is relative - `logs` folder will be created in the current directory.
# If path is absolute - `logs` folder will be created on absolute path.
# If path is not set - `logs` folder will be created in the default location `~/.config/cobalt/logs`
COBALT_LOGS_FOLDER_PATH = os.getenv("COBALT_LOGS_FOLDER_PATH")

if COBALT_LOGS_FOLDER_PATH is None:
    LOGS_FOLDER = os.path.join(os.path.expanduser("~"), ".config", "cobalt", "logs")
else:
    LOGS_FOLDER = os.path.join(os.path.abspath(COBALT_LOGS_FOLDER_PATH), "logs")

if not os.path.isdir(LOGS_FOLDER):
    os.makedirs(LOGS_FOLDER)

try:
    LOG_FILENAME = os.path.join(LOGS_FOLDER, f"{ipynbname.name()}_{date.today()}.out")
except (FileNotFoundError, IndexError):
    # during tests, ipynbname.name() raises FileNotFoundError because no notebook is running
    # in nbconvert, it raises IndexError because it does string processing on an ill-formed kernelid
    LOG_FILENAME = os.path.join(
        LOGS_FOLDER, f"cobalt_{datetime.now().isoformat(timespec='microseconds')}.out"
    )


def is_vscode():
    # Check for VS Code environment variables
    if "VSCODE_PID" in os.environ or "VSCODE_CWD" in os.environ:
        return True

    # Check if the Jupyter front-end is VS Code by inspecting IPython configuration
    try:
        ip = get_ipython()  # noqa: F821
        if ip is not None:
            config = ip.config
            # Look for VS Code-specific configurations in the IPython configuration
            if "vscode" in str(config).lower():
                return True
    except Exception:
        pass

    # Check for loaded VS Code-specific modules in sys.modules
    vscode_modules = ["vscode", "vscode_notebooks"]
    for module in vscode_modules:
        if module in sys.modules:
            return True

    # Check if the Jupyter kernel's metadata contains any VS Code-specific information
    try:
        kernel_file = os.path.join(
            os.path.expanduser("~"),
            ".local",
            "share",
            "jupyter",
            "runtime",
            "kernel-*.json",
        )
        with open(kernel_file) as f:
            kernel_data = json.load(f)
            if "vscode" in str(kernel_data).lower():
                return True
    except Exception:
        pass

    return False


def is_colab_environment() -> bool:
    """Check if the package is running in Google Colab."""
    try:
        module = get_ipython().__module__
        is_colab = module == "google.colab._shell"
    except NameError:
        is_colab = False
    return is_colab


def setup_colab_widget_environment():
    print(
        "Colab environment detected. Enabling custom ipywidgets. "
        "This is required to use the Cobalt UI."
    )
    print("To disable, run:")
    print()
    print("    from google.colab import output")
    print("    output.disable_custom_widget_manager()")
    from google.colab import output

    output.enable_custom_widget_manager()


def setup_debug_logger(name="cobalt_debug"):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    file_handler = logging.FileHandler(os.path.join(LOGS_FOLDER, f"{name}.log"), "w")
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    )
    file_handler.setLevel(logging.DEBUG)
    logger.addHandler(file_handler)
    return logger


def get_logger():
    logger = logging.getLogger("cobalt")
    if not logger.handlers:
        logger.propagate = 0
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        )
        logger.handlers.clear()
        logger.addHandler(handler)
        logging.addLevelName(
            logging.ERROR, f"\033[1;41m{logging.getLevelName(logging.ERROR)}\033[1;0m"
        )
        logger.setLevel(logging.INFO)
    return logger


def get_file_logger():
    logger = logging.getLogger("cobalt_file")
    if not logger.handlers:
        logger.propagate = 0
        handler = logging.FileHandler(LOG_FILENAME)
        handler.setFormatter(
            logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        )
        logger.handlers.clear()
        logger.addHandler(handler)
        logger.setLevel(logging.ERROR)
    return logger


class Notification(str, Enum):
    INFO = "info"
    ERROR = "error"
    WARNING = "warning"
    DEBUG = "debug"


def notify(msg, notification_type=Notification.ERROR):
    """Notifies the user about an error/warning/info without stopping the app.

    Produces log messages, which are stored in a separate log file.
    """
    logger = get_logger()
    map_log_level = {
        "error": logger.error,
        "info": logger.info,
        "warning": logger.warning,
        "debug": logger.debug,
    }

    map_log_level[notification_type](msg)


def notify_and_stop(msg):
    """Notify the user about a custom error and stop the app."""
    notify(msg)
    sys.exit(1)


def notify_callback_error(cb_name):
    msg = (
        f"Something went wrong while executing <{cb_name}> callback. "
        f"Please check the logs for more details: {LOG_FILENAME}"
    )
    logger = get_file_logger()
    logger.exception(f"Something went wrong while executing <{cb_name}> callback.")
    notify(msg)


def inspect_function(func: Callable, args, kwargs):
    """Get fuction params by type and log them."""
    try:
        function_path = f"{func.__module__}.{func.__name__}"

        message = f"Calling {function_path} with "
        if args:
            args_names = list(inspect.signature(func).parameters.keys())
            if "args" in args_names:
                args_names.remove("args")
            if "kwargs" in args_names:
                args_names.remove("kwargs")
            args_info = []
            for i, arg in enumerate(args):
                if i < len(args_names):
                    args_info.append(f"{args_names[i]}: {type(arg).__name__}")
                else:
                    args_info.append(f"{type(arg).__name__}")

            args_str = ", ".join(args_info)

            message = message + "args: " + args_str + " "
        if kwargs:
            kwargs_str = ", ".join(
                [f"{key}: {type(value)}" for key, value in kwargs.items()]
            )
            message = message + "kwargs: " + kwargs_str + " "

        message += "."
        debug_logger.info(message)
    except Exception as e:
        notify(f"Cannot parse function arguments {func.__name__} {e}")


class StopExecution(Exception):
    def _render_traceback_(self):
        return []


def _handle_cb_exceptions(func):
    """Decorator to handle exceptions in callbacks.

    When an exception is raised, shows a generic error message to the user and
    writes a traceback to a log file.
    """

    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            if isinstance(e, mapper.LicenseError):
                message = e.args[0] if e.args else "Unknown License Error"
                show_license_exception(message)
                raise StopExecution from None
            else:
                notify_callback_error(func.__name__)

    return wrapper


def _handle_cb_exceptions_with_debugger(func):
    """Decorator to handle exceptions in callbacks.

    When an exception is raised, shows a generic error message to the user and
    writes a traceback to a log file.
    """

    def wrapper(*args, **kwargs):
        try:
            inspect_function(func, args, kwargs)
            result = func(*args, **kwargs)
            debug_logger.info(f"{func.__name__} completed successfully.")
            return result
        except Exception:
            notify_callback_error(func.__name__)

    return wrapper


def show_license_exception(msg):
    import ipyvuetify as v
    import ipywidgets as widgets

    popup = widgets.Output()

    def show_message(message):
        with popup:
            # Clear previous content
            clear_output()
            # Display the message
            if is_colab_environment():
                from ipyvue import Html

                display(Html(tag="div", style_="display: none"))
            close_button = v.Btn(
                children=["Close"],
                density="compact",
                text=True,
                class_="mx-1",
            )
            close_button.on_event("click", close_message)
            display(
                v.Card(
                    children=[
                        v.CardTitle(children=["License key error"]),
                        v.CardText(children=[message]),
                        close_button,
                    ],
                    width="500",
                )
            )

    def close_message(*_):
        with popup:
            clear_output()

    show_message(msg)
    display(popup)


def check_license():
    """Check the configured license key and print the result."""
    mapper.check_license_manually()


def trivial_wrapper(func):
    return func


if os.getenv("DEBUG", "0") == "1":
    handle_cb_exceptions = _handle_cb_exceptions_with_debugger
elif os.getenv("RAISE_CB_EXC", "0") == "1":
    handle_cb_exceptions = trivial_wrapper
else:
    handle_cb_exceptions = _handle_cb_exceptions

USE_EXPERIMENTAL = os.getenv("USE_EXPERIMENTAL", "0") == "1"


CONFIG_WIDTH = 300
MAX_DRIFT_BINS = 1000

debug_logger = setup_debug_logger()
CONFIG_FILE_PATH = os.path.join(
    os.path.expanduser("~"), ".config", "cobalt", "cobalt.json"
)

REGISTER_API_URL = os.getenv(
    "COBALT_REGISTER_API_URL", "https://api.cobalt.bluelightai.com"
)


def load_config():
    """Load the configuration from the JSON file, or.

    return a default structure if file does not exist.
    """
    if not os.path.exists(CONFIG_FILE_PATH):
        return {"api_keys": {}, "license_key": "", "config": {}}

    with open(CONFIG_FILE_PATH) as config_file:
        return json.load(config_file)


def save_config(config):
    """Save the configuration back to the JSON file or.

    create a new one.
    """
    directory = os.path.dirname(CONFIG_FILE_PATH)
    os.makedirs(directory, exist_ok=True)

    with open(CONFIG_FILE_PATH, "w") as config_file:
        json.dump(config, config_file, indent=4)


def console_log(message: str):
    """Log a Python message to the browser's console."""
    js_code = f'console.log("{message}");'
    display(Javascript(js_code))


@dataclass
class CobaltSettings:
    """Overridable settings that affect global behavior."""

    # don't base64 encode more than 5MB worth of images in colab
    MAX_BASE64_TOTAL_SIZE: int = 5_000_000 if is_colab_environment() else 20_000_000
    """The maximum amount of image data to base64 encode in a table display."""

    USE_RICH_NODE_LABELS: bool = False
    """Default node hover label format for graphs."""

    USE_EXPERIMENTAL: bool = USE_EXPERIMENTAL
    """Enable experimental features."""


settings = CobaltSettings()

LAMBDA_VERSION = 3
