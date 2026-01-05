"""
Flask API application for Face Recognition System
"""

import os
import sys
import logging

# Save current directory and flask module path
current_dir = os.path.dirname(__file__)
parent_dir = os.path.join(current_dir, '..')

# Temporarily remove flask directory from path to import Flask package
if current_dir in sys.path:
    sys.path.remove(current_dir)

# Import Flask package (should work now that __init__.py is removed)
from flask import Flask, g, request

# Now add parent directory for our modules
sys.path.insert(0, parent_dir)

from src.utils import load_config, setup_logging

# Import from flask directory (current directory)
import importlib.util
middleware_path = os.path.join(current_dir, 'middleware.py')
spec = importlib.util.spec_from_file_location("middleware", middleware_path)
middleware = importlib.util.module_from_spec(spec)
spec.loader.exec_module(middleware)

routes_path = os.path.join(current_dir, 'routes', '__init__.py')
spec = importlib.util.spec_from_file_location("routes", routes_path)
routes = importlib.util.module_from_spec(spec)
spec.loader.exec_module(routes)

setup_cors = middleware.setup_cors
setup_rate_limiting = middleware.setup_rate_limiting
error_handler = middleware.error_handler
register_routes = routes.register_routes

# Setup logging
log_level = os.getenv('LOG_LEVEL', 'INFO')
log_file = os.getenv('LOG_FILE', 'logs/face_recognition.log')
logger = setup_logging(log_level, log_file)

# Load configuration
config = load_config()

# Create Flask app
app = Flask(__name__)
app.config.update({
    'CORS': config.get('api', {}).get('cors', {}),
    'RATE_LIMIT': config.get('api', {}).get('rate_limit', {}),
    'MAX_CONTENT_LENGTH': 16 * 1024 * 1024  # 16MB max file size
})

# Store config in app context
@app.before_request
def before_request():
    g.config = config

# Setup middleware
setup_cors(app)
limiter = setup_rate_limiting(app)
error_handler(app)

# Register routes
register_routes(app)


@app.route('/')
def index():
    """API index"""
    return {
        'service': 'Face Recognition API',
        'version': '1.0.0',
        'endpoints': {
            'face': '/api/v1/face/*',
            'liveness': '/api/v1/liveness/*',
            'surveillance': '/api/v1/surveillance/*',
            'admin': '/api/v1/*'
        }
    }


if __name__ == '__main__':
    api_config = config.get('api', {})
    host = api_config.get('host', '0.0.0.0')
    port = api_config.get('port', 8000)
    debug = api_config.get('debug', False)
    
    logger.info(f"Starting Flask API on {host}:{port}")
    app.run(host=host, port=port, debug=debug)

