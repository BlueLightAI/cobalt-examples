import os
import threading
from http.server import SimpleHTTPRequestHandler
from socketserver import TCPServer


class QuietHandler(SimpleHTTPRequestHandler):
    def log_message(self, fmt, *args):
        # Suppress DEBUG and INFO messages based on the status code
        if self.command in ["GET", "POST"]:
            status_code = int(args[1])
            if status_code < 400:
                return
        super().log_message(fmt, *args)


class ImageServer:

    def __init__(self, image_folder: str) -> None:
        self.image_folder = image_folder
        if not os.path.exists(image_folder):
            raise ValueError(f"Images Path {image_folder} does not exist.")

        self.img_host_url = None

    def start(self):
        """Starts the server, serving images at `image_folder`."""
        # This may be redundant code.

        server_thread = threading.Thread(
            target=self.run, args=(self.image_folder,), daemon=True
        )
        server_thread.start()

    def has_started(self) -> bool:
        """Returns whether server has started."""
        return self.img_host_url is not None

    def run(self, image_folder):
        """Run a simple HTTP server to serve local images."""

        def get_handler(*args, **kwargs):
            return QuietHandler(*args, directory=image_folder, **kwargs)

        handler = get_handler
        TCPServer.allow_reuse_address = True
        with TCPServer(("", 0), handler) as httpd:

            def stop_server():
                httpd.shutdown()
                httpd.server_close()

            try:
                abs_path = os.path.abspath(image_folder)
                handler.directory = abs_path
                _, port = httpd.server_address
                host = "localhost"
                self.img_host_url = f"http://{host}:{port}/"
                httpd.serve_forever()
            except Exception as e:
                print(f"Error: {e}")
            finally:
                stop_server()


class ServerRepository:
    def __init__(self):
        self.servers = {}

    def add_server(self, image_folder: str):
        if image_folder not in self.servers:
            server = ImageServer(image_folder)
            self.servers[image_folder] = server

    def run_servers(self):
        for _, server in self.servers.items():
            if not server.has_started():
                server.start()

    def get_servers(self):
        return self.servers.values()


# Global server registry shared across all workspaces
SERVER_REGISTRY = ServerRepository()
