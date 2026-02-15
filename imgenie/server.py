#!/usr/bin/env python3

"""
IMGENIE Web UI Server - Simple Flask wrapper
Uses ModelServer config and creates model instances as needed
"""

import os
import sys
import base64
import io
from pathlib import Path
from typing import Optional, Dict, Union

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

from PIL import Image

from image_describer import ImageDescriber
from image_generator import ImageGenerator
from imgenie.server_old import ModelServer, ModelType

# ========================
# SETUP
# ========================

app = Flask(__name__, static_folder='.', static_url_path='')
CORS(app)

UI_DIR = Path(__file__).parent
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'webp'}
UPLOAD_FOLDER = '/tmp/imgenie_uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

server = None
current_models: Dict[str, Optional[Union[ImageGenerator, ImageDescriber]]] = {
    't2i': None, 'i2t': None}  # Store currently loaded models


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# ========================
# UI ROUTES
# ========================

@app.route('/')
def index():
    return send_from_directory(UI_DIR, 'index.html')


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

    config = {}

    # Extract txt2img models
    if server.t2i_cfg and isinstance(server.t2i_cfg, dict):
        config['txt2img'] = {}
        for name, cfg in server.t2i_cfg.items():
            if isinstance(cfg, dict):
                config['txt2img'][name] = cfg

    # Extract img2txt models
    if server.i2t_cfg and isinstance(server.i2t_cfg, dict):
        config['img2txt'] = {}
        for name, cfg in server.i2t_cfg.items():
            if isinstance(cfg, dict):
                config['img2txt'][name] = cfg

    return jsonify(config)


@app.route('/api/models', methods=['GET'])
def get_models():
    """Get available models for a task"""
    task = request.args.get('task', 'text-to-image')

    if not server:
        return jsonify([])

    models = []

    if task == 'text-to-image' and isinstance(server.t2i_cfg, dict):
        for model_id, cfg in server.t2i_cfg.items():
            if isinstance(cfg, dict):
                models.append({
                    'id': model_id,
                    'name': cfg.get('name', model_id),
                    'description': cfg.get('description', ''),
                    'model_path': cfg.get('model_path', '')
                })

    elif task == 'image-to-text' and isinstance(server.i2t_cfg, dict):
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
            configs = server.t2i_cfg or {}
            model_key = 't2i'
        else:
            configs = server.i2t_cfg or {}
            model_key = 'i2t'

        if model_id not in configs:
            available = list(configs.keys())
            return jsonify({
                'success': False,
                'error': f'Model {model_id} not found. Available: {available}'
            }), 400

        model_config = configs[model_id]
        model_path = model_config.get('model_path', model_id)

        # Create and load model instance
        try:
            if model_key == 't2i':
                current_models['t2i'] = ImageGenerator(
                    model_path=model_path,
                    input_dir=str(server.input_folder),
                    output_dir=str(server.output_folder),
                    lora_path=str(server.lora_path)
                )
                current_models['t2i'].load_model()
                status = True
            else:
                current_models['i2t'] = ImageDescriber(
                    model_path=model_path,
                    input_dir=str(server.input_folder),
                    output_dir=str(server.output_folder)
                )
                current_models['i2t'].load_model()
                status = True

        except Exception as load_err:
            print(f"Failed to load model: {load_err}")
            import traceback
            traceback.print_exc()
            status = False
            return jsonify({
                'success': False,
                'error': f'Failed to load model: {str(load_err)}'
            }), 500

        return jsonify({
            'success': status,
            'message': f'Model {model_id} loaded' if status else f'Failed to load {model_id}',
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

        model_key = 't2i' if task == 'text-to-image' else 'i2t'

        if current_models[model_key]:
            try:
                current_models[model_key].unload_model()
            except:
                pass
            current_models[model_key] = None

        return jsonify({'success': True, 'message': 'Model unloaded'})

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/model/status', methods=['GET'])
def model_status():
    """Get model status"""
    return jsonify({
        't2i': current_models['t2i'] is not None,
        'i2t': current_models['i2t'] is not None
    })


@app.route('/api/generate', methods=['POST'])
def generate():
    """Generate image or describe image"""
    try:
        if not server:
            return jsonify({'success': False, 'error': 'Server not initialized'}), 500

        data = request.json or {}
        task = data.get('task', 'text-to-image')

        if task == 'text-to-image':
            # Text-to-image generation
            if not current_models['t2i']:
                return jsonify({'success': False, 'error': 'T2I model not loaded'}), 400

            prompt = data.get('prompt', '')
            steps = data.get('steps', 30)
            guidance_scale = data.get('guidance_scale', 7.5)
            resolution = data.get('resolution', ['720', '720'])
            seed = data.get('seed', -1)

            if not prompt:
                return jsonify({'success': False, 'error': 'Prompt required'}), 400

            # Convert resolution to integers
            try:
                if isinstance(resolution, list) and len(resolution) >= 2:
                    if isinstance(resolution[0], str):
                        height, width = int(resolution[0]), int(resolution[1])
                    else:
                        height, width = resolution[0], resolution[1]
                else:
                    height, width = 720, 720
            except (ValueError, TypeError):
                height, width = 720, 720

            # Get list of files before generation
            output_dir = server.output_folder
            output_dir.mkdir(parents=True, exist_ok=True)
            files_before = set(os.listdir(str(output_dir)))

            # Call backend generate
            try:
                current_models['t2i'].generate(
                    prompt=prompt,
                    num_inference_steps=steps,
                    guidance_scale=guidance_scale,
                    height=height,
                    width=width,
                    seed=seed if seed >= 0 else None
                )
            except TypeError:
                # Fallback - try without resolution parameters
                current_models['t2i'].generate(
                    prompt=prompt,
                    num_inference_steps=steps,
                    guidance_scale=guidance_scale
                )

            # Find the newly generated image
            files_after = set(os.listdir(str(output_dir)))
            new_files = files_after - files_before

            # Filter for image files
            image_files = [f for f in new_files if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))]

            if image_files:
                # Get the most recent
                latest = max([str(output_dir / f) for f in image_files], key=os.path.getctime)
                with open(latest, 'rb') as f:
                    img_base64 = base64.b64encode(f.read()).decode('utf-8')

                return jsonify({
                    'success': True,
                    'image': f'data:image/png;base64,{img_base64}',
                    'params': {'prompt': prompt, 'steps': steps, 'guidance_scale': guidance_scale, 'resolution': f'{height}x{width}'}
                })
            else:
                return jsonify({'success': False, 'error': 'No image generated'}), 500

        elif task == 'image-to-text':
            # Image description
            if not current_models['i2t']:
                return jsonify({'success': False, 'error': 'I2T model not loaded'}), 400

            if 'image' not in request.files:
                return jsonify({'success': False, 'error': 'No image file'}), 400

            file = request.files['image']
            if not allowed_file(file.filename):
                return jsonify({'success': False, 'error': 'Invalid file type'}), 400

            img = Image.open(io.BytesIO(file.read())).convert('RGB')
            result = current_models['i2t'].describe(img=img)

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


@app.errorhandler(404)
def not_found(error):
    if not request.path.startswith('/api/'):
        return send_from_directory(UI_DIR, 'ui/index.html')
    return jsonify({'error': 'Not found'}), 404


# ========================
# MAIN
# ========================

def main():
    """Main entry point"""
    global server

    import argparse

    parser = argparse.ArgumentParser(description='IMGENIE Web UI Server')
    parser.add_argument('--port', type=int, default=5000)
    parser.add_argument('--config', type=str, help='Path to config file')
    parser.add_argument('--host', type=str, default='127.0.0.1')
    parser.add_argument('--debug', action='store_true')

    args = parser.parse_args()

    if not ModelServer:
        print("Error: ModelServer not available")
        return

    # Initialize model server
    try:
        config_path = args.config or 'imgenie/config/imgenie.config.default.yaml'
        print(f"\n📂 Initializing ModelServer with config: {config_path}")

        server = ModelServer(config_path=config_path)
        print(f"✓ Server initialized")
        print(f"  Output folder: {server.output_folder}")
        print(f"  T2I models: {list(server.t2i_cfg.keys()) if server.t2i_cfg else 'none'}")
        print(f"  I2T models: {list(server.i2t_cfg.keys()) if server.i2t_cfg else 'none'}")
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


if __name__ == '__main__':
    main()
