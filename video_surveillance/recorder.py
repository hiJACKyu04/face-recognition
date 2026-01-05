"""
Video recording module
"""

import cv2
import numpy as np
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)


class VideoRecorder:
    """Record video on events"""
    
    def __init__(self, output_dir: str = "recordings", duration: int = 30, fps: int = 30):
        """
        Initialize video recorder
        
        Args:
            output_dir: Output directory for recordings
            duration: Recording duration in seconds
            fps: Frames per second
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.duration = duration
        self.fps = fps
        self.frames = []
        self.is_recording = False
        self.frame_count = 0
        self.max_frames = duration * fps
        
        logger.info(f"Video recorder initialized: {output_dir}")
    
    def start_recording(self):
        """Start recording"""
        self.frames = []
        self.is_recording = True
        self.frame_count = 0
        logger.info("Recording started")
    
    def add_frame(self, frame: np.ndarray):
        """Add frame to recording"""
        if not self.is_recording:
            return
        
        self.frames.append(frame.copy())
        self.frame_count += 1
        
        if self.frame_count >= self.max_frames:
            self.stop_recording()
    
    def stop_recording(self) -> Optional[str]:
        """
        Stop recording and save video
        
        Returns:
            Path to saved video file or None
        """
        if not self.is_recording or not self.frames:
            return None
        
        self.is_recording = False
        
        # Generate filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"recording_{timestamp}.mp4"
        output_path = self.output_dir / filename
        
        # Get frame dimensions
        if not self.frames:
            return None
        
        height, width = self.frames[0].shape[:2]
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path), fourcc, self.fps, (width, height))
        
        # Write frames
        for frame in self.frames:
            out.write(frame)
        
        out.release()
        
        logger.info(f"Recording saved: {output_path}")
        return str(output_path)
    
    def clear(self):
        """Clear recorded frames"""
        self.frames = []
        self.frame_count = 0

