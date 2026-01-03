#!/usr/bin/env python3

import sys
import threading
import yaml
import requests

# GTK4 Imports
import gi
gi.require_version('Gtk', '4.0')
gi.require_version('Gdk', '4.0')
from gi.repository import Gtk, Gdk, GLib, Pango, GdkPixbuf

from PIL import Image

SERVER_URL = "http://localhost:8000"

class ImgenieApp(Gtk.Application):
    def __init__(self):
        super().__init__(application_id='com.imgenie.app',
                         flags=0)
        self.server = None
        self.config = {}

    def do_activate(self):
        win = ImgenieWindow(self)
        win.present()

    def do_startup(self):
        Gtk.Application.do_startup(self)
        # Load config
        try:
            with open('imgenie/config/imgenie.config.default.yaml', 'r') as f:
                self.config = yaml.safe_load(f)
        except Exception as e:
            print(f"Error loading config: {e}")        

class StepHeader(Gtk.Box):
    def __init__(self, step_index, title, on_click_callback):
        super().__init__(orientation=Gtk.Orientation.VERTICAL)
        self.set_size_request(60, -1)
        self.add_css_class("step-header")
        
        # Click gesture
        gesture = Gtk.GestureClick()
        gesture.connect("released", lambda x, n, a, b: on_click_callback(step_index))
        self.add_controller(gesture)

        # Label (Rotated)
        self.label = Gtk.Label(label=title)
        # self.label.set_angle(90) # Not supported in GTK4
        self.label.add_css_class("rotated-label")
        self.label.set_vexpand(True)
        self.label.add_css_class("step-header-label")
        self.append(self.label)
        
        # Number/Icon
        self.num_label = Gtk.Label(label=str(step_index + 1))
        self.num_label.set_css_classes(["title"])
        self.prepend(self.num_label)

    def set_active(self, active):
        if active:
            self.add_css_class("step-header-active")
        else:
            self.remove_css_class("step-header-active")

class StepBase(Gtk.Box):
    def __init__(self, app_window):
        super().__init__(orientation=Gtk.Orientation.VERTICAL)
        self.app_window = app_window
        self.add_css_class("step-content")
        self.set_hexpand(True)
        self.set_vexpand(True)
        self.setup_ui()
    
    def setup_ui(self):
        pass

class Step1LoadModels(StepBase):
    def setup_ui(self):
        title = Gtk.Label(label="Step 1: Load Models")
        title.add_css_class("title")
        title.set_halign(Gtk.Align.START)
        self.append(title)
        
        grid = Gtk.Grid()
        grid.set_column_spacing(20)
        grid.set_row_spacing(20)
        self.append(grid)

        # T2I Dropdown
        grid.attach(Gtk.Label(label="Text-to-Image Model:"), 0, 0, 1, 1)
        self.t2i_store = Gtk.StringList()
        start_opts = self.app_window.app.config.get('txt2img', {}).keys()
        for k in start_opts:
            self.t2i_store.append(k)
        self.t2i_dropdown = Gtk.DropDown(model=self.t2i_store)
        grid.attach(self.t2i_dropdown, 1, 0, 1, 1)

        # I2T Dropdown
        grid.attach(Gtk.Label(label="Image-to-Text Model:"), 0, 1, 1, 1)
        self.i2t_store = Gtk.StringList()
        img_opts = self.app_window.app.config.get('img2txt', {}).keys()
        for k in img_opts:
            self.i2t_store.append(k)
        self.i2t_dropdown = Gtk.DropDown(model=self.i2t_store)
        grid.attach(self.i2t_dropdown, 1, 1, 1, 1)

        # Load Button
        self.load_btn = Gtk.Button(label="Load Models")
        self.load_btn.add_css_class("suggested-action")
        self.load_btn.connect("clicked", self.on_load_clicked)
        self.load_btn.set_margin_top(20)
        self.append(self.load_btn)

        # Status
        self.status_label = Gtk.Label(label="Select models and click Load.")
        self.status_label.set_margin_top(10)
        self.append(self.status_label)

    def on_load_clicked(self, btn):
        t2i_idx = self.t2i_dropdown.get_selected()
        i2t_idx = self.i2t_dropdown.get_selected()
        
        t2i_model = self.t2i_store.get_string(t2i_idx)
        i2t_model = self.i2t_store.get_string(i2t_idx)

        self.status_label.set_label(f"Loading {t2i_model} and {i2t_model}...")
        self.load_btn.set_sensitive(False)

        # Run in thread
        thread = threading.Thread(target=self.load_models_thread, args=(t2i_model, i2t_model))
        thread.start()

    def load_models_thread(self, t2i, i2t):
        try:
            self.app_window.app.server._load_models({"t2i": t2i, "i2t": i2t})
            GLib.idle_add(self.on_load_success)
        except Exception as e:
            GLib.idle_add(self.on_load_error, str(e))

    def on_load_success(self):
        self.status_label.set_label("Models loaded successfully!")
        self.load_btn.set_sensitive(True)
        # Advance to step 2?
        self.app_window.go_to_step(1)

    def on_load_error(self, err):
        self.status_label.set_label(f"Error loading models: {err}")
        self.load_btn.set_sensitive(True)

class Step2SelectImage(StepBase):
    def setup_ui(self):
        title = Gtk.Label(label="Step 2: Select Reference Image")
        title.add_css_class("title")
        self.append(title)

        # Drop/Select Area
        self.image_area = Gtk.Button()
        self.image_area.set_vexpand(True)
        self.image_area.set_hexpand(True)
        self.image_area.add_css_class("flat") # Custom style for drop area?
        
        box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=10)
        box.set_halign(Gtk.Align.CENTER)
        box.set_valign(Gtk.Align.CENTER)
        self.preview_image = Gtk.Picture()
        self.preview_image.set_size_request(200, 200)
        self.default_icon = Gtk.Image.new_from_icon_name("document-open-symbolic")
        self.default_icon.set_pixel_size(64)
        
        box.append(self.default_icon)
        box.append(Gtk.Label(label="Click to select an image"))
        self.image_area.set_child(box)
        self.image_area.connect("clicked", self.on_select_image)
        
        # Right click to clear
        gesture = Gtk.GestureClick()
        gesture.set_button(3)
        gesture.connect("pressed", self.on_right_click)
        self.image_area.add_controller(gesture)
        
        self.append(self.image_area)

        # Describe Button / Navigation
        self.describe_btn = Gtk.Button(label="Describe Image")
        self.describe_btn.add_css_class("suggested-action")
        self.describe_btn.connect("clicked", self.on_describe_clicked)
        self.append(self.describe_btn)
        
        self.selected_file_path = None

    def on_right_click(self, gesture, n_press, x, y):
        self.clear_image()
        
    def clear_image(self):
        self.selected_file_path = None
        # Restore default look
        box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=10)
        box.set_halign(Gtk.Align.CENTER)
        box.set_valign(Gtk.Align.CENTER)
        icon = Gtk.Image.new_from_icon_name("document-open-symbolic")
        icon.set_pixel_size(64)
        box.append(icon)
        box.append(Gtk.Label(label="Click to select an image"))
        self.image_area.set_child(box)

    def on_select_image(self, btn):
        # File Chooser Dialog
        dialog = Gtk.FileChooserNative.new("Open Image", self.app_window, Gtk.FileChooserAction.OPEN, "_Open", "_Cancel")
        filter_img = Gtk.FileFilter()
        filter_img.set_name("Images")
        filter_img.add_mime_type("image/*")
        dialog.add_filter(filter_img)
        
        dialog.connect("response", self.on_file_response)
        dialog.show()

    def on_file_response(self, dialog, response):
        if response == Gtk.ResponseType.ACCEPT:
            f = dialog.get_file()
            self.selected_file_path = f.get_path()
            self.display_image(self.selected_file_path)

    def display_image(self, path):
        # Update preview
        texture = Gdk.Texture.new_from_filename(path)
        self.preview_image.set_paintable(texture)
        # Replace the icon with the picture
        self.image_area.set_child(self.preview_image)

    def on_describe_clicked(self, btn):
        if not self.selected_file_path:
            return
        
        btn.set_sensitive(False)
        btn.set_label("Describing...")
        
        thread = threading.Thread(target=self.describe_thread, args=(self.selected_file_path,))
        thread.start()

    def describe_thread(self, path):
        try:
            img = Image.open(path).convert("RGB")
            # Need to ensure backend is ready. Assuming step 1 passed.
            if self.app_window.app.server.i2t_model:
                res = self.app_window.app.server.i2t_model.describe(img)
                desc = res['description']
                GLib.idle_add(self.on_describe_success, desc)
            else:
                 GLib.idle_add(self.on_describe_error, "Model not loaded")
        except Exception as e:
            GLib.idle_add(self.on_describe_error, str(e))

    def on_describe_success(self, description):
        self.describe_btn.set_sensitive(True)
        self.describe_btn.set_label("Describe Image")
        # Update Step 3
        self.app_window.step3.set_description(description)
        self.app_window.go_to_step(2)

    def on_describe_error(self, err):
        self.describe_btn.set_sensitive(True)
        self.describe_btn.set_label("Describe Image")
        # Show error (alert or label)
        print(f"Error: {err}")

class Step3UpdateDescription(StepBase):
    def setup_ui(self):
        title = Gtk.Label(label="Step 3: Edit Description")
        title.add_css_class("title")
        self.append(title)

        self.textview = Gtk.TextView()
        self.textview.set_wrap_mode(Gtk.WrapMode.WORD)
        self.textview.set_vexpand(True)
        self.buffer = self.textview.get_buffer()
        self.buffer.set_text("Description will appear here...")
        
        scroller = Gtk.ScrolledWindow()
        scroller.set_child(self.textview)
        self.append(scroller)

        self.gen_btn_btn = Gtk.Button(label="Generate New Image")
        self.gen_btn_btn.add_css_class("suggested-action")
        self.gen_btn_btn.connect("clicked", self.on_generate_clicked)
        self.gen_btn_btn.set_margin_top(10)
        self.append(self.gen_btn_btn)

    def set_description(self, text):
        self.buffer.set_text(text)

    def on_generate_clicked(self, btn):
        start, end = self.buffer.get_bounds()
        text = self.buffer.get_text(start, end, True)
        self.app_window.step4.start_generation(text)
        self.app_window.go_to_step(3)

class Step4GenerateImage(StepBase):
    def setup_ui(self):
        title = Gtk.Label(label="Step 4: Result")
        title.add_css_class("title")
        self.append(title)

        self.image_display = Gtk.Picture()
        self.image_display.set_vexpand(True)
        self.image_display.set_hexpand(True)
        self.image_display.set_content_fit(Gtk.ContentFit.CONTAIN)
        
        # Right click to clear
        gesture = Gtk.GestureClick()
        gesture.set_button(3)
        gesture.connect("pressed", self.on_right_click)
        self.image_display.add_controller(gesture)

        self.append(self.image_display)

        self.status = Gtk.Label(label="Waiting for input...")
        self.append(self.status)

    def on_right_click(self, gesture, n_press, x, y):
        self.image_display.set_paintable(None)
        self.status.set_label("Cleared.")

    def start_generation(self, prompt):
        self.status.set_label("Generating image... Please wait.")
        # Load a spinner or GIF
        
        thread = threading.Thread(target=self.generate_thread, args=(prompt,))
        thread.start()

    def generate_thread(self, prompt):
        try:
            # Check model
            if self.app_window.app.server.t2i_model:
                path, _ = self.app_window.app.server.t2i_model.generate(prompt=prompt)
                GLib.idle_add(self.on_gen_success, path)
            else:
                 GLib.idle_add(self.on_gen_error, "Model not loaded")
        except Exception as e:
            GLib.idle_add(self.on_gen_error, str(e))

    def on_gen_success(self, path):
        self.status.set_label("Generation Complete!")
        texture = Gdk.Texture.new_from_filename(path)
        self.image_display.set_paintable(texture)

    def on_gen_error(self, err):
        self.status.set_label(f"Error: {err}")

class ImgenieWindow(Gtk.ApplicationWindow):
    def __init__(self, app):
        super().__init__(application=app)
        self.app = app
        self.set_title("Imgenie")
        self.set_default_size(900, 600)

        # CSS
        provider = Gtk.CssProvider()
        provider.load_from_path('imgenie/style.css')
        Gtk.StyleContext.add_provider_for_display(
            Gdk.Display.get_default(), 
            provider, 
            Gtk.STYLE_PROVIDER_PRIORITY_APPLICATION
        )

        # Main Container
        self.main_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL)
        self.main_box.add_css_class("step-container")
        self.set_child(self.main_box)

        # Steps
        self.steps = []
        self.step_headers = []
        
        self.step1 = Step1LoadModels(self)
        self.step2 = Step2SelectImage(self)
        self.step3 = Step3UpdateDescription(self)
        self.step4 = Step4GenerateImage(self)
        
        steps_objs = [
            ("Load Models", self.step1),
            ("Select Image", self.step2),
            ("Edit Description", self.step3),
            ("Generate Image", self.step4)
        ]

        for i, (title, widget) in enumerate(steps_objs):
            # Header
            header = StepHeader(i, title, self.go_to_step)
            self.step_headers.append(header)
            self.main_box.append(header)
            
            # Content (wrapped in a generic container if needed, but StepBase calls itself content)
            self.steps.append(widget)
            self.main_box.append(widget)
        
        # Settings (Rightmost)
        # self.settings_btn = ...
        
        self.current_step = 0
        self.update_ui_state()

    def go_to_step(self, index):
        self.current_step = index
        self.update_ui_state()

    def update_ui_state(self):
        for i, widget in enumerate(self.steps):
            if i == self.current_step:
                widget.set_visible(True)
                self.step_headers[i].set_active(True)
            else:
                widget.set_visible(False)
                self.step_headers[i].set_active(False)

if __name__ == "__main__":
    app = ImgenieApp()
    app.run(sys.argv)
