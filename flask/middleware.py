"""
Flask middleware for authentication, CORS, and error handling
"""

import logging
from functools import wraps
from flask import request, jsonify, g
import os

logger = logging.getLogger(__name__)


def setup_cors(app):
    """Setup CORS for Flask app"""
    from flask_cors import CORS
    
    config = app.config.get('CORS', {})
    if config.get('enabled', True):
        origins = config.get('origins', ['*'])
        CORS(app, origins=origins)
        logger.info("CORS enabled")


def setup_rate_limiting(app):
    """Setup rate limiting for Flask app"""
    from flask_limiter import Limiter
    from flask_limiter.util import get_remote_address
    
    config = app.config.get('RATE_LIMIT', {})
    if config.get('enabled', True):
        limiter = Limiter(
            app=app,
            key_func=get_remote_address,
            default_limits=[f"{config.get('per_minute', 60)} per minute"]
        )
        logger.info("Rate limiting enabled")
        return limiter
    return None


def require_api_key(f):
    """Decorator to require API key authentication"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        auth_config = g.config.get('api', {}).get('auth', {})
        
        if not auth_config.get('enabled', False):
            return f(*args, **kwargs)
        
        api_key = request.headers.get('X-API-Key') or request.args.get('api_key')
        expected_key = auth_config.get('api_key')
        
        if not api_key or api_key != expected_key:
            return jsonify({'error': 'Invalid or missing API key'}), 401
        
        return f(*args, **kwargs)
    
    return decorated_function


def error_handler(app):
    """Register error handlers"""
    
    @app.errorhandler(400)
    def bad_request(error):
        return jsonify({'error': 'Bad request', 'message': str(error)}), 400
    
    @app.errorhandler(401)
    def unauthorized(error):
        return jsonify({'error': 'Unauthorized', 'message': str(error)}), 401
    
    @app.errorhandler(404)
    def not_found(error):
        return jsonify({'error': 'Not found', 'message': str(error)}), 404
    
    @app.errorhandler(500)
    def internal_error(error):
        logger.error(f"Internal server error: {error}")
        return jsonify({'error': 'Internal server error'}), 500
    
    @app.errorhandler(Exception)
    def handle_exception(e):
        logger.exception("Unhandled exception")
        return jsonify({'error': 'Internal server error', 'message': str(e)}), 500

