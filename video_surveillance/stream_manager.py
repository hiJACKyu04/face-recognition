"""
Stream manager for video surveillance
"""

import logging
from typing import Dict, List, Optional
from src.stream_processor import StreamProcessor

logger = logging.getLogger(__name__)


class StreamManager:
    """Manage multiple video streams"""
    
    def __init__(self):
        """Initialize stream manager"""
        self.streams: Dict[str, StreamProcessor] = {}
        logger.info("Stream manager initialized")
    
    def add_stream(self, stream_id: str, source: str, stream_type: str = "auto",
                   frame_skip: int = 1) -> bool:
        """
        Add a new stream
        
        Args:
            stream_id: Unique stream identifier
            source: Stream source (URL, file path, or camera index)
            stream_type: Type of stream
            frame_skip: Process every Nth frame
            
        Returns:
            True if added successfully
        """
        if stream_id in self.streams:
            logger.warning(f"Stream {stream_id} already exists")
            return False
        
        try:
            processor = StreamProcessor(source, stream_type, frame_skip)
            self.streams[stream_id] = processor
            logger.info(f"Added stream: {stream_id}")
            return True
        except Exception as e:
            logger.error(f"Error adding stream {stream_id}: {e}")
            return False
    
    def remove_stream(self, stream_id: str):
        """Remove a stream"""
        if stream_id in self.streams:
            self.streams[stream_id].stop()
            del self.streams[stream_id]
            logger.info(f"Removed stream: {stream_id}")
    
    def start_stream(self, stream_id: str, callback=None):
        """Start a stream"""
        if stream_id in self.streams:
            self.streams[stream_id].start(callback)
            logger.info(f"Started stream: {stream_id}")
        else:
            logger.warning(f"Stream {stream_id} not found")
    
    def stop_stream(self, stream_id: str):
        """Stop a stream"""
        if stream_id in self.streams:
            self.streams[stream_id].stop()
            logger.info(f"Stopped stream: {stream_id}")
    
    def get_frame(self, stream_id: str, timeout: float = 1.0):
        """Get frame from stream"""
        if stream_id in self.streams:
            return self.streams[stream_id].get_frame(timeout)
        return None
    
    def get_all_streams(self) -> List[str]:
        """Get list of all stream IDs"""
        return list(self.streams.keys())
    
    def stop_all(self):
        """Stop all streams"""
        for stream_id in list(self.streams.keys()):
            self.remove_stream(stream_id)
        logger.info("All streams stopped")

