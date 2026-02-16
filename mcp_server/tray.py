import pystray
from PIL import Image
import threading
import sys
import os
from pathlib import Path
import time
import webbrowser

import multiprocessing

if getattr(sys, 'frozen', False):
    BASE_PATH = Path(sys._MEIPASS)

else:
    BASE_PATH = Path(__file__).resolve().parent.parent


# --- Shared status file (same location used by server.py) ---
_STATUS_DIR = Path.home() / ".gdskg"
_STATUS_FILE = _STATUS_DIR / "server_status"


def _read_status() -> bool:
    """
    Read the current server status from the shared indicator file.

    Returns:
        bool: True if the server is marked as 'started', False otherwise.
    """
    try:
        if _STATUS_FILE.exists():
            return _STATUS_FILE.read_text().strip() == "started"
    except Exception:
        pass
    return False



def _write_status(started: bool) -> None:
    """
    Persist the server status to the shared indicator file.

    Args:
        started (bool): Whether the server should be marked as 'started'.
    """
    _STATUS_DIR.mkdir(parents=True, exist_ok=True)
    _STATUS_FILE.write_text("started" if started else "stopped")



class MCPTrayApp:
    """
    System tray application for controlling the GDSKG MCP server.
    
    Provides a visual indicator of the server's status and allows the user
    to start/stop the server and access documentation.
    """

    def __init__(self):
        """
        Initialize the system tray application and resolve asset paths.
        """

        self.icon = None
        self._server_loading = False

        if getattr(sys, 'frozen', False):

            self.icon_path = BASE_PATH / "mcp_server" / "icon.png"
        else:
            self.icon_path = Path(__file__).resolve().parent / "icon.png"

    @property
    def is_running(self) -> bool:
        """
        Check if the server is currently considered running.

        Returns:
            bool: True if the server is active.
        """
        return _read_status()


    def create_menu(self) -> pystray.Menu:
        """
        Construct the dynamic context menu for the system tray icon based on current status.

        Returns:
            pystray.Menu: The built menu object.
        """

        running = self.is_running

        if self._server_loading:
            status_text = "Status: Starting..." if not running else "Status: Stopping..."
        else:
            status_text = "Status: Running" if running else "Status: Stopped"

        return pystray.Menu(
            pystray.MenuItem(status_text, lambda: None, enabled=False),
            pystray.MenuItem("Start Server", self.start_server, enabled=not running and not self._server_loading),
            pystray.MenuItem("Stop Server", self.stop_server, enabled=running and not self._server_loading),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem("About", self.show_about),
            pystray.MenuItem("Exit", self.exit_app)
        )

    def start_server(self, icon: pystray.Icon = None, item: pystray.MenuItem = None) -> None:
        """
        Initiate the server startup process in a background thread.

        Args:
            icon (pystray.Icon, optional): The tray icon instance.
            item (pystray.MenuItem, optional): The menu item that triggered the action.
        """

        if self.is_running or self._server_loading:
            return

        def run():
            self._server_loading = True
            self.update_icon()
            try:
                _write_status(True)
                time.sleep(0.3)

            finally:
                self._server_loading = False
                self.update_icon()

        threading.Thread(target=run, daemon=True).start()

    def stop_server(self, icon: pystray.Icon = None, item: pystray.MenuItem = None) -> None:
        """
        Initiate the server shutdown process in a background thread.

        Args:
            icon (pystray.Icon, optional): The tray icon instance.
            item (pystray.MenuItem, optional): The menu item that triggered the action.
        """

        if not self.is_running or self._server_loading:
            return

        def run():
            self._server_loading = True
            self.update_icon()
            try:
                _write_status(False)
                time.sleep(0.3)
            finally:
                self._server_loading = False
                self.update_icon()

        threading.Thread(target=run, daemon=True).start()

    def update_icon(self) -> None:
        """
        Refresh the system tray icon's menu to reflect status changes.
        """

        if self.icon:
            self.icon.menu = self.create_menu()

    def show_about(self, icon: pystray.Icon, item: pystray.MenuItem) -> None:
        """
        Open the project's GitHub page in the default web browser.
        """

        webbrowser.open("https://github.com/gdskg/gdskg")

    def exit_app(self, icon: pystray.Icon, item: pystray.MenuItem) -> None:
        """
        Shut down the tray application.
        """

        icon.stop()

    def run(self) -> None:
        """
        Initialize and run the pystray main loop.
        """
        if not self.icon_path.exists():

            print(f"Error: Icon not found at {self.icon_path}")
            return

        _write_status(True)


        image = Image.open(self.icon_path)
        self.icon = pystray.Icon("GDSKG", image, "GDSKG MCP Server", self.create_menu())
        self.icon.run()

if __name__ == "__main__":
    multiprocessing.freeze_support()


    if "--server" in sys.argv:
        try:

            from mcp_server.server import mcp
            mcp.run()
        except ImportError:
            import sys

            sys.path.append(str(Path(__file__).resolve().parent))
            from server import mcp
            mcp.run()
    else:
        app = MCPTrayApp()
        app.run()
