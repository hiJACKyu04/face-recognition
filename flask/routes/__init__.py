"""
Flask API routes
"""

import sys
import os
from pathlib import Path

# Add routes directory to path
routes_dir = Path(__file__).parent
if str(routes_dir) not in sys.path:
    sys.path.insert(0, str(routes_dir))

def register_routes(app):
    """Register all routes"""
    import face_routes
    import liveness_routes
    import surveillance_routes
    import admin_routes
    
    app.register_blueprint(face_routes.face_bp, url_prefix='/api/v1')
    app.register_blueprint(liveness_routes.liveness_bp, url_prefix='/api/v1')
    app.register_blueprint(surveillance_routes.surveillance_bp, url_prefix='/api/v1')
    app.register_blueprint(admin_routes.admin_bp, url_prefix='/api/v1')

