import gi
import torch
gi.require_version('Gtk', '4.0')
gi.require_version('Adw', '1') # Use Adw for that modern Ubuntu look
from gi.repository import Gtk, Adw

class ImgenieApp(Adw.Application):
    def __init__(self):
        super().__init__(application_id='com.imgenie.dev')
        
    def do_activate(self):
        # The Window
        win = Adw.ApplicationWindow(application=self)
        win.set_title("Imgenie Dev Tool")
        win.set_default_size(400, 200)

        # Main Layout
        box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=12)
        box.set_margin_top(20)
        
        self.label = Gtk.Label(label="GTK 4 + PyTorch Ready")
        box.append(self.label)

        button = Gtk.Button(label="Check GPU Status")
        button.connect("clicked", self.on_check_gpu)
        box.append(button)

        # win.set_child(box)
        win.set_content(box)
        win.present()

    def on_check_gpu(self, button):
        if torch.cuda.is_available():
            # For AMD/ROCm, torch.cuda.is_available() still returns True
            device_name = torch.cuda.get_device_name(0)
            self.label.set_text(f"GPU Found: {device_name}")
        else:
            self.label.set_text("No GPU detected - Check /dev/dri")

if __name__ == "__main__":
    app = ImgenieApp()
    app.run(None)
