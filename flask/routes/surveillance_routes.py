"""
Video surveillance API routes
"""

from flask import Blueprint, jsonify, g
import logging
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

# Import middleware from parent flask directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from middleware import require_api_key

logger = logging.getLogger(__name__)

surveillance_bp = Blueprint('surveillance', __name__)


@surveillance_bp.route('/surveillance/status', methods=['GET'])
@require_api_key
def surveillance_status():
    """Get surveillance system status"""
    try:
        # TODO: Implement surveillance status
        return jsonify({
            'success': True,
            'status': 'active',
            'cameras': []
        })
    
    except Exception as e:
        logger.error(f"Error in surveillance_status: {e}")
        return jsonify({'error': str(e)}), 500

