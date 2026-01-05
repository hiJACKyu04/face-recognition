"""
Admin and management API routes
"""

from flask import Blueprint, jsonify, g, request
import logging
import sys
import os
import psutil

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from src.database import FaceDatabase
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from middleware import require_api_key

logger = logging.getLogger(__name__)

admin_bp = Blueprint('admin', __name__)


def get_database():
    """Get database instance"""
    if 'database' not in g:
        db_config = g.config.get('database', {})
        db_path = db_config.get('path', 'data/face_recognition.db')
        g.database = FaceDatabase(db_path)
    return g.database


@admin_bp.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'face-recognition-api'
    })


@admin_bp.route('/stats', methods=['GET'])
@require_api_key
def get_stats():
    """Get system statistics"""
    try:
        database = get_database()
        
        # Get database stats
        persons = database.list_persons()
        events = database.get_events(limit=1000)
        
        # System stats
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        
        return jsonify({
            'success': True,
            'database': {
                'persons_count': len(persons),
                'events_count': len(events)
            },
            'system': {
                'cpu_percent': cpu_percent,
                'memory_percent': memory.percent,
                'memory_available_mb': memory.available / (1024 * 1024)
            }
        })
    
    except Exception as e:
        logger.error(f"Error in get_stats: {e}")
        return jsonify({'error': str(e)}), 500


@admin_bp.route('/events', methods=['GET'])
@require_api_key
def get_events():
    """Get event history"""
    try:
        database = get_database()
        
        limit = int(request.args.get('limit', 100))
        event_type = request.args.get('type')
        
        events = database.get_events(limit=limit, event_type=event_type)
        
        return jsonify({
            'success': True,
            'events': events,
            'count': len(events)
        })
    
    except Exception as e:
        logger.error(f"Error in get_events: {e}")
        return jsonify({'error': str(e)}), 500

