import getpass
import os

import requests

from cobalt.config import is_vscode, load_config, save_config


class AnyAPIWrapper:
    def __init__(self, api_key=None):
        self.api_key = api_key
        self.base_url = ""
        if not self.api_key:
            raise ValueError("API key is required")

    def _get_headers(self):
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

    def get(self, endpoint, params=None):
        url = f"{self.base_url}/{endpoint}"
        headers = self._get_headers()
        response = requests.get(url, headers=headers, params=params)
        return self.handle_response(response)

    def post(self, endpoint, data=None):
        url = f"{self.base_url}/{endpoint}"
        headers = self._get_headers()
        response = requests.post(url, headers=headers, json=data)
        return self.handle_response(response)

    def put(self, endpoint, data=None):
        url = f"{self.base_url}/{endpoint}"
        headers = self._get_headers()
        response = requests.put(url, headers=headers, json=data)
        return self.handle_response(response)

    def delete(self, endpoint):
        url = f"{self.base_url}/{endpoint}"
        headers = self._get_headers()
        response = requests.delete(url, headers=headers)
        return self.handle_response(response)

    @staticmethod
    def handle_response(response):
        if 200 <= response.status_code < 300:
            try:
                return response.json()
            except ValueError:
                return response.text
        else:
            response.raise_for_status()


class OpenAIWrapper(AnyAPIWrapper):
    def __init__(self, api_key=None):
        self.api_key = api_key
        super().__init__(api_key)

        self.base_url = "https://api.openai.com/v1"
        self.raw_client = self._get_raw_client()

    @property
    def name(self):
        return "OpenAI APi Wrapper"

    def __repr__(self):
        return self.name

    def _get_raw_client(self):
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("openai package is not installed.") from None

        try:
            client = OpenAI(api_key=self.api_key)
        except Exception as e:
            raise ValueError(f"Error while initializing OpenAI client: {e}") from e
        else:
            return client

    def prompt(self, message, model="gpt-4"):
        """Creates a model response for the given chat conversation.

        Args:
            message (str): The message to send to the model.
            model (str, optional): The model to use. Defaults to "gpt-4".
                List of models: https://platform.openai.com/docs/models#model-endpoint-compatibility

        Returns: A string containing the conversation response.
        """
        messages = [
            {"role": "system", "content": message},
        ]
        completion = self.raw_client.chat.completions.create(
            model=model,
            messages=messages,
        )
        return completion.choices[0].message.content


API_CLIENT_MAP = {"openai": {"cls": OpenAIWrapper, "env_var_name": "OPENAI_API_KEY"}}


def check_openai_api_key(api_key):
    """Check if OpenAi token is valid."""
    from openai import AuthenticationError, OpenAI

    client = OpenAI(api_key=api_key)
    try:
        client.models.list()
    except AuthenticationError:
        return False
    else:
        return True


def setup_api_client():
    """Set up the API client by updating or adding the API key to the JSON config file."""
    config = load_config()
    if is_vscode():
        print(
            "Please enter your API configuration in the input field at the top of the window \u2191"
        )

    api_name = input(
        f"Choose API client from the list: {list(API_CLIENT_MAP.keys())}: "
    ).strip()
    api_client_config = API_CLIENT_MAP.get(api_name)

    if not api_client_config:
        raise ValueError(
            f"Unrecognized API client name. Valid names are: {list(API_CLIENT_MAP.keys())}"
        )

    valid = False
    api_client_env_name = api_client_config["env_var_name"]

    while not valid:
        api_key = getpass.getpass(f"Enter your valid API KEY for {api_name}: ").strip()
        api_key_checker = check_openai_api_key(api_key=api_key)

        # Check if the API key is Valid
        if not api_key_checker:
            print("API token is invalid")

        else:
            print("API token is valid")
            valid = True

    # Check if the API key already exists in the config
    if api_client_env_name in config["api_keys"]:
        overwrite = (
            input(
                f"API key for {api_name} already exists. Do you want to overwrite it? (yes/no): "
            )
            .strip()
            .lower()
        )
        if overwrite == "yes":
            config["api_keys"][api_client_env_name] = api_key
            print(f"API key for {api_name} was updated.")
        else:
            print(f"API key for {api_name} was not updated.")
    else:
        # Add new API key to the config
        config["api_keys"][api_client_env_name] = api_key
        print(f"API key for {api_name} saved.")

    # Save the updated config back to the JSON file
    save_config(config)


def get_api_client(api_name: str = "openai"):
    """Get the API client by loading the API key from the JSON config or environment variables."""
    config = load_config()  # Load existing config from JSON

    api_client_config = API_CLIENT_MAP.get(api_name)
    if not api_client_config:
        raise ValueError(
            f"Unrecognized API client name. Valid names are: {list(API_CLIENT_MAP.keys())}"
        )

    api_client_cls = api_client_config["cls"]
    api_client_env_name = api_client_config["env_var_name"]

    # Try to get the API key from the JSON config first
    api_key = config["api_keys"].get(api_client_env_name)

    # Check if the API key is set in the environment variables
    api_key_from_env = os.getenv(api_client_env_name)
    if not api_key and not api_key_from_env:
        raise ValueError(
            "You need to configure your API client first. "
            "Run Workspace.setup_api_client()"
        )

    # If the environment variable is set, it overrides the config file key
    if api_key_from_env:
        api_key = api_key_from_env
    return api_client_cls(api_key=api_key)
