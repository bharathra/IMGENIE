#!/usr/bin/env python3

"""
IMGENIE Web UI Server - Simple Flask wrapper
"""

import os
import sys
import base64
import io
import yaml
import time
import logging
from pathlib import Path
from typing import Optional, Dict, Union, List

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

from PIL import Image

from image_describer import ImageDescriber
from image_generator import ImageGenerator

# ========================
# CONFIG & STATE
# ========================

class ImgenieServer:
    """Manage model loading and generation"""
    
    def __init__(self, config_path: str):
        self.config_path = Path(config_path)
        self.config = self._load_config()
        
        self.output_folder = Path(self.config.get('output_path', '/root/.imgenie/output'))
        self.input_folder = Path(self.config.get('input_path', '/root/.imgenie/input'))
        self.lora_path = Path(self.config.get('lora_path', '/root/.imgenie/loras'))
        
        # Load model configs
        self.t2i_cfg = self.config.get('txt2img', {})
        self.i2t_cfg = self.config.get('img2txt', {})
        
        # Current loaded models
        self.t2i_model: Optional[ImageGenerator] = None
        self.i2t_model: Optional[ImageDescriber] = None
        self.current_t2i_id: Optional[str] = None
        self.current_i2t_id: Optional[str] = None

    def _load_config(self) -> dict:
        if not self.config_path.exists():
            print(f"Config file not found: {self.config_path}")
            return {}
        try:
            with open(self.config_path, 'r') as f:
                return yaml.safe_load(f) or {}
        except Exception as e:
            print(f"Error loading config: {e}")
            return {}

server: Optional['ImgenieServer'] = None

# ========================
# SETUP FLASK
# ========================

app = Flask(__name__, static_folder='ui', static_url_path='/ui')
CORS(app)

UI_DIR = Path(__file__).parent / 'ui'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'webp'}
UPLOAD_FOLDER = '/tmp/imgenie_uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# ========================
# UI ROUTES
# ========================

@app.route('/')
def index():
    return send_from_directory(UI_DIR, 'index.html')

@app.route('/app.js')
def serve_app_js():
    return send_from_directory(UI_DIR, 'app.js')

@app.route('/ui/<path:path>')
def send_ui(path):
    return send_from_directory(UI_DIR, path)

# ========================
# API ROUTES
# ========================

@app.route('/api/config', methods=['GET'])
def get_config():
    """Get UI configuration"""
    return jsonify({
        'theme_colors': {
            'dark': {'primary': '#0a0e27', 'accent': '#ffd700', 'success': '#66ff66', 'error': '#ff6b6b'},
            'light': {'primary': '#ffffff', 'accent': '#0052cc', 'success': '#00cc66', 'error': '#ff6b6b'}
        },
        'default_theme': 'dark'
    })


@app.route('/api/app-config', methods=['GET'])
def get_app_config():
    """Get application configuration"""
    if not server:
        return jsonify({})

    config = {
        'txt2img': server.t2i_cfg,
        'img2txt': server.i2t_cfg
    }
    return jsonify(config)


@app.route('/api/models', methods=['GET'])
def get_models():
    """Get available models for a task"""
    task = request.args.get('task', 'text-to-image')

    if not server:
        return jsonify([])

    models = []

    if task == 'text-to-image':
        for model_id, cfg in server.t2i_cfg.items():
            if isinstance(cfg, dict):
                models.append({
                    'id': model_id,
                    'name': cfg.get('name', model_id),
                    'description': cfg.get('description', ''),
                    'model_path': cfg.get('model_path', '')
                })

    elif task == 'image-to-text':
        for model_id, cfg in server.i2t_cfg.items():
            if isinstance(cfg, dict):
                models.append({
                    'id': model_id,
                    'name': cfg.get('name', model_id),
                    'description': cfg.get('description', ''),
                    'model_path': cfg.get('model_path', '')
                })

    return jsonify(models)


@app.route('/api/models/<model_id>/resolutions', methods=['GET'])
def get_model_resolutions(model_id):
    """Get available resolutions for a model"""
    task = request.args.get('task', 'text-to-image')
    resolutions = ['720x720', '1024x1024']  # Default

    if not server:
        return jsonify({'resolutions': resolutions})

    try:
        if task == 'text-to-image' and model_id in server.t2i_cfg:
            cfg = server.t2i_cfg[model_id]
            if isinstance(cfg, dict) and 'resolution_options' in cfg:
                resolutions = cfg['resolution_options']

        elif task == 'image-to-text' and model_id in server.i2t_cfg:
            cfg = server.i2t_cfg[model_id]
            if isinstance(cfg, dict) and 'resolution_options' in cfg:
                resolutions = cfg['resolution_options']
    except:
        pass

    return jsonify({'model_id': model_id, 'task': task, 'resolutions': resolutions})


@app.route('/api/loras', methods=['GET'])
def get_loras():
    """Get available LoRAs for the current T2I model"""
    if not server:
        return jsonify({})

    loras = {
        'characters': [],
        'concepts': []
    }
    
    # Use current loaded model's config, or fallback to first available
    model_id = server.current_t2i_id
    if not model_id and server.t2i_cfg:
        model_id = list(server.t2i_cfg.keys())[0]

    if not model_id or model_id not in server.t2i_cfg:
        return jsonify(loras)

    cfg = server.t2i_cfg[model_id]
    
    # Paths from config
    char_path_str = cfg.get('character_lora_path')
    concept_path_str = cfg.get('concept_lora_path')

    # Helper to scan dir
    def scan_dir(path_str):
        found = []
        if not path_str:
            return found
        p = Path(path_str)
        # If relative, prepend root if exists
        if not p.is_absolute() and server.config.get('root_dir'):
            p = Path(server.config.get('root_dir')) / p
            
        if p.exists() and p.is_dir():
            for f in p.glob('*.safetensors'):
                found.append(f.stem)
        return sorted(found)

    loras['characters'] = scan_dir(char_path_str)
    loras['concepts'] = scan_dir(concept_path_str)

    return jsonify(loras)


@app.route('/api/model/load', methods=['POST'])
def load_model():
    """Load a model"""
    try:
        data = request.get_json()
        model_id = data.get('model_id')
        task = data.get('task', 'text-to-image')

        if not model_id or not server:
            return jsonify({'success': False, 'error': 'Invalid request'}), 400

        # Get the actual model path from config
        if task == 'text-to-image':
            configs = server.t2i_cfg
            current_loaded = server.current_t2i_id
        else:
            configs = server.i2t_cfg
            current_loaded = server.current_i2t_id

        if model_id not in configs:
            available = list(configs.keys())
            return jsonify({
                'success': False,
                'error': f'Model {model_id} not found. Available: {available}'
            }), 400

        if current_loaded == model_id:
             return jsonify({
                'success': True,
                'message': f'Model {model_id} already loaded',
                'model': model_id,
                'task': task
            })

        model_config = configs[model_id]
        model_path = model_config.get('model_path', model_id)
        
        # Resolve model path relative to root if needed (optional logic, keeping simple for now)
        # Assuming model_path is relative to /root/.imgenie or absolute
        if not os.path.isabs(model_path):
             # Try appending to root_dir if available in config
             root_dir = server.config.get('root_dir')
             if root_dir:
                 potential_path = os.path.join(root_dir, model_path)
                 if os.path.exists(potential_path):
                     model_path = potential_path


        # Create and load model instance
        try:
            if task == 'text-to-image':
                # Unload previous logic not strictly needed if we just overwrite, but good practice
                if server.t2i_model:
                     try: server.t2i_model.unload_model()
                     except: pass
                
                # Get lora path for this specific model if defined, else global
                lora_path = model_config.get('lora_path', str(server.lora_path))
                if not os.path.isabs(lora_path) and server.config.get('root_dir'):
                     lora_path = os.path.join(server.config.get('root_dir'), lora_path)
                
                server.t2i_model = ImageGenerator(
                    model_path=model_path,
                    input_dir=str(server.input_folder),
                    output_dir=str(server.output_folder),
                    lora_path=lora_path
                )
                if server.t2i_model.load_model():
                    server.current_t2i_id = model_id
                    status = True
                else:
                    status = False
                    
            else:
                if server.i2t_model:
                     try: server.i2t_model.unload_model() 
                     except: pass

                server.i2t_model = ImageDescriber(
                    model_path=model_path,
                    input_dir=str(server.input_folder),
                    output_dir=str(server.output_folder)
                )
                if server.i2t_model.load_model():
                    server.current_i2t_id = model_id
                    status = True
                else:
                    status = False

        except Exception as load_err:
            print(f"Failed to load model: {load_err}")
            import traceback
            traceback.print_exc()
            return jsonify({
                'success': False,
                'error': f'Failed to load model: {str(load_err)}'
            }), 500

        if not status:
             return jsonify({'success': False, 'error': 'Model load returned false'}), 500

        return jsonify({
            'success': True,
            'message': f'Model {model_id} loaded',
            'model': model_id,
            'task': task
        })

    except Exception as e:
        print(f"Error loading model: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/model/unload', methods=['POST'])
def unload_model():
    """Unload a model"""
    try:
        data = request.get_json() or {}
        task = data.get('task', 'text-to-image')

        if task == 'text-to-image':
            if server.t2i_model:
                server.t2i_model.unload_model()
                server.t2i_model = None
                server.current_t2i_id = None
        else:
            if server.i2t_model:
                server.i2t_model.unload_model()
                server.i2t_model = None
                server.current_i2t_id = None

        return jsonify({'success': True, 'message': 'Model unloaded'})

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


import torch

# Global progress tracking
generation_progress = {
    'status': 'idle',
    'progress': 0,
    'step': 0,
    'total_steps': 0,
    'message': ''
}

def progress_callback(step: int, timestep: int, latents: torch.FloatTensor):
    """Callback for diffusion pipeline"""
    global generation_progress
    generation_progress['status'] = 'generating'
    generation_progress['step'] = step
    # exact progress calculation depends on total steps which we might need to pass or infer
    # For now, just store the step.
                                                                                               
@app.route('/api/progress', methods=['GET'])
def get_progress():
    """Get generation progress"""
    return jsonify(generation_progress)


@app.route('/api/model/status', methods=['GET'])
def model_status():
    """Get model status and GPU memory usage"""
    status = {
        't2i': server.current_t2i_id is not None,
        'i2t': server.current_i2t_id is not None,
        't2i_model': server.current_t2i_id,
        'i2t_model': server.current_i2t_id,
        'gpu_memory': {
            'allocated': 0,
            'reserved': 0,
            'max': 0
        }
    }
    
    try:
        if torch.cuda.is_available():
            status['gpu_memory'] = {
                'allocated': torch.cuda.memory_allocated() / (1024**3), # GB
                'reserved': torch.cuda.memory_reserved() / (1024**3),   # GB
                'max': torch.cuda.get_device_properties(0).total_memory / (1024**3) # GB
            }
    except Exception as e:
        print(f"Error getting GPU status: {e}")
        
    return jsonify(status)


@app.route('/api/generate', methods=['POST'])
def generate():
    """Generate image or describe image"""
    global generation_progress
    try:
        if not server:
            return jsonify({'success': False, 'error': 'Server not initialized'}), 500

        if request.is_json:
            data = request.json
        else:
            data = request.form
        task = data.get('task', 'text-to-image')
        
        # Handle FormData for image upload - check if it is really image-to-text or img2img
        if request.files and 'task' not in request.form:
             # Fallback if task not explicit in form
             task = request.form.get('task', 'image-to-text')

        if task == 'text-to-image':
            # Text-to-image generation (or Img2Img)
            if not server.t2i_model:
                return jsonify({'success': False, 'error': 'T2I model not loaded'}), 400

            prompt = data.get('prompt', '')
            steps = int(data.get('steps', 30))
            guidance_scale = float(data.get('guidance_scale', 7.5))
            resolution = data.get('resolution', '720x720')
            seed = data.get('seed', '')
            strength = float(data.get('strength', 0.8)) # Default strength
            
            ref_image_path = None
            if 'image' in request.files:
                file = request.files['image']
                if file and allowed_file(file.filename):
                    filename = f"ref_{int(time.time())}_{file.filename}"
                    ref_image_path = os.path.join(UPLOAD_FOLDER, filename)
                    file.save(ref_image_path)
            
            # Reset progress
            generation_progress = {
                'status': 'starting',
                'progress': 0,
                'step': 0,
                'total_steps': steps,
                'message': 'Starting...'
            }

            def update_progress(pipe, step, timestep, callback_kwargs):
                global generation_progress
                generation_progress['status'] = 'generating'
                generation_progress['step'] = step
                generation_progress['progress'] = int((step / steps) * 100)
                generation_progress['message'] = f'Step {step}/{steps}'
                return callback_kwargs

            try:
                seed = int(seed) if seed else -1
            except:
                seed = -1

            if not prompt:
                return jsonify({'success': False, 'error': 'Prompt required'}), 400

            # Parse resolution
            try:
                if isinstance(resolution, str) and 'x' in resolution:
                    w, h = resolution.split('x')
                    width, height = int(w), int(h)
                elif isinstance(resolution, list):
                    width, height = int(resolution[0]), int(resolution[1])
                else:
                     width, height = 720, 720
            except:
                width, height = 720, 720

            # Handle LoRAs
            loras = data.get('loras', [])
            if server.t2i_model:
                try:
                    # Always reset loras if none provided? 
                    # image_generator.load_loras handles empty list by unloading.
                    # So we should always call it if we want to support "no lora" when user clears selection.
                    
                    lora_paths = []
                    lora_weights = []
                    
                    # Get base paths from config
                    # We need the model config for paths. 
                    # server.current_t2i_id should be set if model is loaded.
                    model_cfg = server.t2i_cfg.get(server.current_t2i_id, {})
                    char_base = model_cfg.get('character_lora_path')
                    concept_base = model_cfg.get('concept_lora_path')

                    for lora in loras:
                        # Expected format: {'type': 'character'|'concept', 'name': 'filename', 'weight': 1.0}
                        l_type = lora.get('type')
                        l_name = lora.get('name')
                        l_weight = float(lora.get('weight', 1.0))
                        
                        if not l_name: continue
                        
                        full_path = None
                        if l_type == 'character' and char_base:
                            full_path = os.path.join(char_base, f"{l_name}.safetensors")
                        elif l_type == 'concept' and concept_base:
                            full_path = os.path.join(concept_base, f"{l_name}.safetensors")
                            
                        if full_path and os.path.exists(full_path):
                            lora_paths.append(full_path)
                            lora_weights.append(l_weight)
                    
                    # Call load_loras even if empty to clear previous
                    print(f"Loading LoRAs: {lora_paths}")
                    server.t2i_model.load_loras(lora_paths, lora_weights)

                except Exception as e:
                    print(f"Error loading LoRAs: {e}")
                    import traceback
                    traceback.print_exc()


            # Call generate
            try:
                output_image = server.t2i_model.generate(
                    prompt=prompt,
                    num_inference_steps=steps,
                    guidance_scale=guidance_scale,
                    height=height,
                    width=width,
                    seed=seed if seed >= 0 else None,
                    callback=update_progress,
                    ref_image_path=ref_image_path,
                    strength=strength,
                )

                # Mark as done
                generation_progress['status'] = 'completed'
                generation_progress['progress'] = 100
                generation_progress['message'] = 'Completed'
                
                if output_image:
                    # Save local copy to TEMP folder
                    timestamp = time.strftime('%Y%m%d_%H%M%S')
                    safe_prompt = "".join([c if c.isalnum() else "_" for c in prompt])[:20]
                    filename = f"gen_{timestamp}_{safe_prompt}.png"
                    temp_path = os.path.join('/tmp', filename)
                    output_image.save(temp_path)
                    print(f"DEBUG: Saved generated image to temp: {temp_path}")

                    # Encode for response
                    buffered = io.BytesIO()
                    output_image.save(buffered, format="PNG")
                    img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

                    return jsonify({
                        'success': True,
                        'image': f'data:image/png;base64,{img_base64}',
                        'image_id': filename,
                        'params': {
                            'prompt': prompt, 
                            'steps': steps, 
                            'guidance_scale': guidance_scale, 
                            'resolution': f'{width}x{height}', 
                            'ref_image_strength': strength,
                            'seed': seed,
                            'ref_image_path': ref_image_path
                        }
                    })
                else:
                    return jsonify({'success': False, 'error': 'No image generated'}), 500

            except Exception as gen_err:
                 generation_progress['status'] = 'failed'
                 generation_progress['message'] = str(gen_err)
                 print(f"Generation error: {gen_err}")
                 import traceback
                 traceback.print_exc()
                 return jsonify({'success': False, 'error': str(gen_err)}), 500

        elif task == 'image-to-text':
            # Image description
            if not server.i2t_model:
                return jsonify({'success': False, 'error': 'I2T model not loaded'}), 400

            if 'image' not in request.files:
                return jsonify({'success': False, 'error': 'No image file'}), 400

            file = request.files['image']
            if not allowed_file(file.filename):
                return jsonify({'success': False, 'error': 'Invalid file type'}), 400

            img = Image.open(io.BytesIO(file.read())).convert('RGB')
            # Assuming describe returns a dict
            result = server.i2t_model.describe(img=img)

            if result and 'description' in result:
                return jsonify({
                    'success': True,
                    'description': result['description'],
                    'filename': str(file.filename)
                })
            else:
                return jsonify({'success': False, 'error': 'Failed to describe image'}), 500

        return jsonify({'success': False, 'error': 'Unknown task'}), 400

    except Exception as e:
        print(f"Error in generate: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/health', methods=['GET'])
def health():
    """Health check"""
    return jsonify({'status': 'healthy', 'server': server is not None})


# ========================
# MAIN
# ========================

def main():
    """Main entry point"""

    import flask.cli
    flask.cli.show_server_banner = lambda *args: None
    
    # Option A: Only show errors and warnings (Recommended)
    # This removes the "INFO" request logs but keeps critical errors
    logging.getLogger('werkzeug').setLevel(logging.ERROR)

    # Option B: Completely disable the werkzeug logger
    # This will stop ALL messages starting with INFO:werkzeug
    log = logging.getLogger('werkzeug')
    log.disabled = True

    global server

    import argparse

    parser = argparse.ArgumentParser(description='IMGENIE Web UI Server')
    parser.add_argument('--port', type=int, default=5000)
    parser.add_argument('--config', type=str, help='Path to config file')
    parser.add_argument('--host', type=str, default='127.0.0.1')
    parser.add_argument('--debug', action='store_true')

    args = parser.parse_args()

    # Initialize model server
    try:
        config_path = args.config or 'imgenie/config/imgenie.config.default.yaml'
        if not os.path.isabs(config_path):
             config_path = os.path.abspath(config_path)
             
        # print(f"\n📂 Initializing ImgenieServer with config: {config_path}")

        server = ImgenieServer(config_path=config_path)
        # print(f"✓ Server initialized")
        # print(f"  Output folder: {server.output_folder}")
        # print(f"  T2I models: {list(server.t2i_cfg.keys()) if server.t2i_cfg else 'none'}")
        # print(f"  I2T models: {list(server.i2t_cfg.keys()) if server.i2t_cfg else 'none'}")
    except Exception as e:
        print(f"✗ Failed to initialize server: {e}")
        import traceback
        traceback.print_exc()
        return

    # Print startup info
    print(f"\n" + "="*70)
    print("IMGENIE Web UI Server")
    print("="*70)
    print(f"  Host: {args.host}")
    print(f"  Port: {args.port}")
    print(f"  URL: http://{args.host}:{args.port}")
    print(f"  Config: {config_path}")
    print("="*70 + "\n")

    # Run server
    try:
        app.run(host=args.host, port=args.port, debug=args.debug, use_reloader=False, threaded=True)
    except KeyboardInterrupt:
        print("\n⏹️  Shutting down...")
        sys.exit(0)


@app.route('/api/save', methods=['POST'])
def save_image():
    """Save generated image and metadata to output folder"""
    if not server:
        return jsonify({'success': False, 'error': 'Server not initialized'}), 500
        
    data = request.json
    image_id = data.get('image_id')
    metadata = data.get('metadata', {})
    
    if not image_id:
        return jsonify({'success': False, 'error': 'Image ID required'}), 400
        
    temp_path = os.path.join('/tmp', image_id)
    if not os.path.exists(temp_path):
        return jsonify({'success': False, 'error': 'Image not found (expired?)'}), 404
        
    try:
        # Move image to output folder
        target_path = os.path.join(server.output_folder, image_id)
        
        # Copy instead of move so we don't break subsequent saves if user clicks multiple times
        import shutil
        shutil.copy2(temp_path, target_path)
        
        # Save metadata
        yaml_path = os.path.splitext(target_path)[0] + '.yaml'
        # Add timestamp and tool info if not present
        metadata['saved_at'] = time.strftime('%Y-%m-%d %H:%M:%S')
        metadata['tool'] = 'Imgenie Web UI'
        
        with open(yaml_path, 'w') as f:
            yaml.dump(metadata, f, default_flow_style=False)
            
        return jsonify({'success': True, 'path': target_path})

    except Exception as e:
        print(f"Error saving image: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

if __name__ == '__main__':
    main()
