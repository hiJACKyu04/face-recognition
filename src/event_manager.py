"""
Event management and alerting system
"""

import logging
import json
from typing import Dict, List, Optional, Callable
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class EventType(Enum):
    """Event types"""
    FACE_DETECTED = "face_detected"
    FACE_MATCHED = "face_matched"
    UNKNOWN_FACE = "unknown_face"
    LIVENESS_PASSED = "liveness_passed"
    LIVENESS_FAILED = "liveness_failed"
    CAMERA_CONNECTED = "camera_connected"
    CAMERA_DISCONNECTED = "camera_disconnected"
    ERROR = "error"


class EventManager:
    """Manage events and alerts"""
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize event manager
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.event_handlers: List[Callable] = []
        self.alert_enabled = self.config.get('events', {}).get('alerts', {}).get('enabled', True)
        
        # Notification handlers
        self.notification_handlers = {
            'email': None,
            'webhook': None,
            'mqtt': None
        }
        
        logger.info("Event manager initialized")
    
    def register_handler(self, handler: Callable):
        """Register an event handler"""
        self.event_handlers.append(handler)
        logger.debug(f"Registered event handler: {handler.__name__}")
    
    def emit_event(self, event_type: EventType, data: Dict, 
                   person_id: Optional[int] = None, face_id: Optional[int] = None,
                   camera_id: Optional[str] = None, similarity: Optional[float] = None):
        """
        Emit an event
        
        Args:
            event_type: Type of event
            data: Event data
            person_id: Associated person ID
            face_id: Associated face ID
            camera_id: Associated camera ID
            similarity: Similarity score (for matches)
        """
        event = {
            'type': event_type.value,
            'timestamp': datetime.now().isoformat(),
            'person_id': person_id,
            'face_id': face_id,
            'camera_id': camera_id,
            'similarity': similarity,
            'data': data
        }
        
        logger.info(f"Event emitted: {event_type.value}")
        
        # Call registered handlers
        for handler in self.event_handlers:
            try:
                handler(event)
            except Exception as e:
                logger.error(f"Error in event handler {handler.__name__}: {e}")
        
        # Check if alert should be sent
        if self.alert_enabled:
            self._check_alert(event)
    
    def _check_alert(self, event: Dict):
        """Check if alert should be sent for event"""
        event_type = event['type']
        alerts_config = self.config.get('events', {}).get('alerts', {})
        
        should_alert = False
        
        if event_type == EventType.UNKNOWN_FACE.value:
            should_alert = alerts_config.get('unknown_face', False)
        elif event_type == EventType.FACE_MATCHED.value:
            should_alert = alerts_config.get('match_found', False)
        elif event_type == EventType.LIVENESS_FAILED.value:
            should_alert = alerts_config.get('liveness_failed', False)
        
        if should_alert:
            self._send_alert(event)
    
    def _send_alert(self, event: Dict):
        """Send alert notification"""
        notifications = self.config.get('events', {}).get('notifications', {})
        
        # Email notification
        if notifications.get('email', {}).get('enabled', False):
            self._send_email(event, notifications['email'])
        
        # Webhook notification
        if notifications.get('webhook', {}).get('enabled', False):
            self._send_webhook(event, notifications['webhook'])
        
        # MQTT notification
        if notifications.get('mqtt', {}).get('enabled', False):
            self._send_mqtt(event, notifications['mqtt'])
    
    def _send_email(self, event: Dict, config: Dict):
        """Send email notification"""
        try:
            import smtplib
            from email.mime.text import MIMEText
            from email.mime.multipart import MIMEMultipart
            
            smtp_server = config.get('smtp_server')
            smtp_port = config.get('smtp_port', 587)
            username = config.get('username')
            password = config.get('password')
            recipients = config.get('recipients', [])
            
            if not all([smtp_server, username, password, recipients]):
                logger.warning("Email configuration incomplete")
                return
            
            msg = MIMEMultipart()
            msg['From'] = username
            msg['To'] = ', '.join(recipients)
            msg['Subject'] = f"Face Recognition Alert: {event['type']}"
            
            body = f"""
            Event Type: {event['type']}
            Timestamp: {event['timestamp']}
            Camera ID: {event.get('camera_id', 'N/A')}
            Person ID: {event.get('person_id', 'N/A')}
            
            Details:
            {json.dumps(event['data'], indent=2)}
            """
            
            msg.attach(MIMEText(body, 'plain'))
            
            server = smtplib.SMTP(smtp_server, smtp_port)
            server.starttls()
            server.login(username, password)
            server.send_message(msg)
            server.quit()
            
            logger.info(f"Email alert sent for event: {event['type']}")
            
        except Exception as e:
            logger.error(f"Error sending email alert: {e}")
    
    def _send_webhook(self, event: Dict, config: Dict):
        """Send webhook notification"""
        try:
            import requests
            
            url = config.get('url')
            if not url:
                logger.warning("Webhook URL not configured")
                return
            
            response = requests.post(url, json=event, timeout=5)
            response.raise_for_status()
            
            logger.info(f"Webhook alert sent for event: {event['type']}")
            
        except Exception as e:
            logger.error(f"Error sending webhook alert: {e}")
    
    def _send_mqtt(self, event: Dict, config: Dict):
        """Send MQTT notification"""
        try:
            import paho.mqtt.client as mqtt
            
            broker = config.get('broker')
            port = config.get('port', 1883)
            topic = config.get('topic', 'face_recognition/events')
            
            if not broker:
                logger.warning("MQTT broker not configured")
                return
            
            client = mqtt.Client()
            client.connect(broker, port, 60)
            client.publish(topic, json.dumps(event))
            client.disconnect()
            
            logger.info(f"MQTT alert sent for event: {event['type']}")
            
        except ImportError:
            logger.warning("paho-mqtt not installed, skipping MQTT notification")
        except Exception as e:
            logger.error(f"Error sending MQTT alert: {e}")

