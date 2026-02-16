#!/usr/bin/env python3
import os
import signal
import subprocess
import sys
import time

import gi
gi.require_version('Gtk', '3.0')
try:
    gi.require_version('AyatanaAppIndicator3', '0.1')
    from gi.repository import AyatanaAppIndicator3 as AppIndicator
except ImportError:
    try:
        gi.require_version('AppIndicator3', '0.1')
        from gi.repository import AppIndicator3 as AppIndicator
    except ImportError:
        print("Neither AyatanaAppIndicator3 nor AppIndicator3 found.")
        sys.exit(1)

from gi.repository import Gtk, GLib

APP_NAME = "Imgenie"
ICON_NAME = "utilities-terminal"  # Using a stock icon
SCRIPT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "imgenie.sh")

class ImgenieTray:
    def __init__(self):
        self.indicator = AppIndicator.Indicator.new(
            "imgenie-tray",
            ICON_NAME,
            AppIndicator.IndicatorCategory.APPLICATION_STATUS
        )
        self.indicator.set_status(AppIndicator.IndicatorStatus.ACTIVE)
        self.indicator.set_menu(self.build_menu())
        self.process = None
        
        self.start_server()

    def build_menu(self):
        menu = Gtk.Menu()
        
        # Status item (disabled)
        self.status_item = Gtk.MenuItem(label="Imgenie Running...")
        self.status_item.set_sensitive(False)
        menu.append(self.status_item)
        
        menu.append(Gtk.SeparatorMenuItem())
        
        # Stop item
        stop_item = Gtk.MenuItem(label="Stop & Exit")
        stop_item.connect("activate", self.quit)
        menu.append(stop_item)
        
        menu.show_all()
        return menu

    def check_container_exists(self):
        try:
            result = subprocess.run(
                ["docker", "ps", "-a", "--format", "{{.Names}}"], 
                capture_output=True, text=True
            )
            return "imgenie" in result.stdout.splitlines()
        except:
            return False

    def start_server(self):
        if not self.check_container_exists():
            self.show_error("Container 'imgenie' not found.\nPlease run imgenie.sh in a terminal first to set it up.")
            sys.exit(1)

        print(f"Starting {SCRIPT_PATH}...")
        log_file = os.path.expanduser("~/.imgenie/imgenie.log")
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        
        try:
            # Open log file for appending
            self.log_fd = open(log_file, "a")
            self.log_fd.write(f"\n--- Starting Imgenie at {time.ctime()} ---\n")
            self.log_fd.flush()

            # Run without shell=True to easily kill via PID logic if needed, 
            # or use shell=False with arguments list. imgenie.sh is executable.
            # Redirect stdout/stderr to log file
            self.process = subprocess.Popen(
                [SCRIPT_PATH], 
                stdout=self.log_fd,
                stderr=subprocess.STDOUT,
                preexec_fn=os.setsid
            )
            self.status_item.set_label(f"Imgenie (PID: {self.process.pid})")
            
            # Check if it crashes immediately (give it 1s)
            GLib.timeout_add(1000, self.check_process_status)
            
        except Exception as e:
            self.show_error(f"Failed to start server:\n{e}")
            sys.exit(1)

    def check_process_status(self):
        if self.process:
            ret = self.process.poll()
            if ret is not None:
                self.show_error(f"Imgenie server exited repeatedly with code {ret}.\nCheck ~/.imgenie/imgenie.log for details.")
                self.quit(None)
                return False
        return True

    def show_error(self, message):
        dialog = Gtk.MessageDialog(
            parent=None,
            flags=0,
            message_type=Gtk.MessageType.ERROR,
            buttons=Gtk.ButtonsType.OK,
            text="Imgenie Error"
        )
        dialog.format_secondary_text(message)
        dialog.run()
        dialog.destroy()

    def quit(self, _):
        print("Stopping server...")
        
        # 1. Kill the process inside the container first
        try:
            # We use pkill regarding the full command line (-f)
            subprocess.run(
                ["docker", "exec", "imgenie", "pkill", "-f", "imgenie_server.py"],
                stdout=subprocess.DEVNULL, 
                stderr=subprocess.DEVNULL
            )
        except Exception as e:
            print(f"Error stopping container process: {e}")

        # 2. Stop the local wrapper script
        if self.process:
            try:
                # Send SIGTERM to the process group
                os.killpg(os.getpgid(self.process.pid), signal.SIGTERM)
                self.process.wait(timeout=2)
            except Exception:
                pass
        
        Gtk.main_quit()

if __name__ == "__main__":
    # Handle Ctrl+C from terminal if run there
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    
    app = ImgenieTray()
    Gtk.main()
