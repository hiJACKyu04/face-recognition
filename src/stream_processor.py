"""
Video stream processing module
Handles RTSP, webcam, and video file inputs
"""

import cv2
import numpy as np
import threading
import queue
import logging
from typing import Optional, Callable, Dict, Any
from pathlib import Path

logger = logging.getLogger(__name__)


class StreamProcessor:
    """Process video streams from various sources"""
    
    def __init__(self, source: str, stream_type: str = "auto", 
                 frame_skip: int = 1, max_queue_size: int = 10):
        """
        Initialize stream processor
        
        Args:
            source: Stream source (RTSP URL, webcam index, or file path)
            stream_type: Type of stream ('rtsp', 'webcam', 'file', 'auto')
            frame_skip: Process every Nth frame
            max_queue_size: Maximum frame queue size
        """
        self.source = source
        self.frame_skip = frame_skip
        self.max_queue_size = max_queue_size
        self.frame_queue = queue.Queue(maxsize=max_queue_size)
        
        # Auto-detect stream type
        if stream_type == "auto":
            self.stream_type = self._detect_stream_type(source)
        else:
            self.stream_type = stream_type
        
        self.cap: Optional[cv2.VideoCapture] = None
        self.is_running = False
        self.thread: Optional[threading.Thread] = None
        self.frame_count = 0
        
        logger.info(f"Stream processor initialized: {source} ({self.stream_type})")
    
    def _detect_stream_type(self, source: str) -> str:
        """Auto-detect stream type from source"""
        source_lower = source.lower()
        
        if source_lower.startswith(('rtsp://', 'http://', 'https://')):
            return 'rtsp'
        elif source_lower.endswith(('.mp4', '.avi', '.mov', '.mkv', '.flv')):
            return 'file'
        elif source.isdigit() or source == '0':
            return 'webcam'
        else:
            # Try as file path
            if Path(source).exists():
                return 'file'
            return 'webcam'  # Default
    
    def start(self, callback: Optional[Callable] = None):
        """
        Start processing stream
        
        Args:
            callback: Optional callback function for each frame
        """
        if self.is_running:
            logger.warning("Stream processor already running")
            return
        
        # Open stream
        self.cap = self._open_stream()
        if self.cap is None or not self.cap.isOpened():
            logger.error(f"Failed to open stream: {self.source}")
            return
        
        self.is_running = True
        self.thread = threading.Thread(target=self._process_stream, args=(callback,), daemon=True)
        self.thread.start()
        logger.info("Stream processor started")
    
    def _open_stream(self) -> Optional[cv2.VideoCapture]:
        """Open video stream based on type"""
        try:
            if self.stream_type == 'rtsp':
                # RTSP streams may need special handling
                cap = cv2.VideoCapture(self.source, cv2.CAP_FFMPEG)
                # Set buffer size to reduce latency
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            elif self.stream_type == 'webcam':
                index = int(self.source) if self.source.isdigit() else 0
                cap = cv2.VideoCapture(index)
            elif self.stream_type == 'file':
                cap = cv2.VideoCapture(self.source)
            else:
                logger.error(f"Unknown stream type: {self.stream_type}")
                return None
            
            if cap.isOpened():
                logger.info(f"Stream opened successfully: {self.source}")
                return cap
            else:
                logger.error(f"Failed to open stream: {self.source}")
                return None
                
        except Exception as e:
            logger.error(f"Error opening stream: {e}")
            return None
    
    def _process_stream(self, callback: Optional[Callable]):
        """Process stream in background thread"""
        while self.is_running and self.cap is not None:
            ret, frame = self.cap.read()
            
            if not ret:
                logger.warning("Failed to read frame")
                # Try to reopen stream if it's RTSP
                if self.stream_type == 'rtsp':
                    self.cap.release()
                    self.cap = self._open_stream()
                    if self.cap is None:
                        break
                else:
                    break
            
            self.frame_count += 1
            
            # Frame skipping
            if self.frame_count % self.frame_skip != 0:
                continue
            
            # Add to queue (non-blocking)
            try:
                self.frame_queue.put_nowait(frame)
            except queue.Full:
                # Remove oldest frame
                try:
                    self.frame_queue.get_nowait()
                    self.frame_queue.put_nowait(frame)
                except queue.Empty:
                    pass
            
            # Call callback if provided
            if callback:
                try:
                    callback(frame, self.frame_count)
                except Exception as e:
                    logger.error(f"Error in frame callback: {e}")
        
        logger.info("Stream processing stopped")
    
    def get_frame(self, timeout: float = 1.0) -> Optional[np.ndarray]:
        """
        Get next frame from queue
        
        Args:
            timeout: Timeout in seconds
            
        Returns:
            Frame as numpy array or None
        """
        try:
            frame = self.frame_queue.get(timeout=timeout)
            return frame
        except queue.Empty:
            return None
    
    def stop(self):
        """Stop stream processing"""
        self.is_running = False
        
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2.0)
        
        if self.cap:
            self.cap.release()
            self.cap = None
        
        # Clear queue
        while not self.frame_queue.empty():
            try:
                self.frame_queue.get_nowait()
            except queue.Empty:
                break
        
        logger.info("Stream processor stopped")
    
    def get_info(self) -> Dict[str, Any]:
        """Get stream information"""
        if self.cap is None:
            return {}
        
        info = {
            'width': int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            'fps': self.cap.get(cv2.CAP_PROP_FPS),
            'frame_count': self.frame_count,
            'is_running': self.is_running,
            'stream_type': self.stream_type
        }
        
        return info
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.stop()

