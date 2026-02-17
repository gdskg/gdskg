import pystray
from PIL import Image
import threading
import sys
import os
from pathlib import Path
import time
import webbrowser

import multiprocessing

# Resolve the base directory
if getattr(sys, 'frozen', False):
    BASE_PATH = Path(sys._MEIPASS)
else:
    # Assuming we are in mcp_server/tray.py in dev mode
    BASE_PATH = Path(__file__).resolve().parent.parent

# --- Shared status file (same location used by server.py) ---
_STATUS_DIR = Path.home() / ".gdskg"
_STATUS_FILE = _STATUS_DIR / "server_status"


def _read_status() -> bool:
    """Read the server status from the shared file."""
    try:
        if _STATUS_FILE.exists():
            return _STATUS_FILE.read_text().strip() == "started"
    except Exception:
        pass
    return False


def _write_status(started: bool):
    """Write the server status to the shared file."""
    _STATUS_DIR.mkdir(parents=True, exist_ok=True)
    _STATUS_FILE.write_text("started" if started else "stopped")


class MCPTrayApp:
    def __init__(self):
        self.icon = None
        self._server_loading = False

        # Determine icon path
        if getattr(sys, 'frozen', False):
            self.icon_path = BASE_PATH / "mcp_server" / "icon.png"
        else:
            self.icon_path = Path(__file__).resolve().parent / "icon.png"

    @property
    def is_running(self) -> bool:
        """Read live status from the shared file."""
        return _read_status()

    def create_menu(self):
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

    def start_server(self, icon=None, item=None):
        if self.is_running or self._server_loading:
            return

        def run():
            self._server_loading = True
            self.update_icon()
            try:
                _write_status(True)
                # Brief pause so the user sees the loading indicator
                time.sleep(0.3)
            finally:
                self._server_loading = False
                self.update_icon()

        threading.Thread(target=run, daemon=True).start()

    def stop_server(self, icon=None, item=None):
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

    def update_icon(self):
        if self.icon:
            self.icon.menu = self.create_menu()

    def show_about(self, icon, item):
        webbrowser.open("https://github.com/gdskg/gdskg")

    def exit_app(self, icon, item):
        icon.stop()

    def run(self):
        if not self.icon_path.exists():
            print(f"Error: Icon not found at {self.icon_path}")
            return

        # Auto-start: mark server as started on app launch
        _write_status(True)

        image = Image.open(self.icon_path)
        self.icon = pystray.Icon("GDSKG", image, "GDSKG MCP Server", self.create_menu())
        self.icon.run()

if __name__ == "__main__":
    # Support for multiprocessing in frozen apps
    multiprocessing.freeze_support()

    if "--server" in sys.argv:
        # Avoid circular import by importing here
        try:
            from mcp_server.server import mcp
            mcp.run()
        except ImportError:
            # Fallback for dev mode where it might be in current dir or different path
            import sys
            sys.path.append(str(Path(__file__).resolve().parent))
            from server import mcp
            mcp.run()
    else:
        app = MCPTrayApp()
        app.run()
