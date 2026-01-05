"""
Database management for face recognition system
"""

import sqlite3
import json
import numpy as np
from typing import List, Dict, Optional, Any, Tuple
from datetime import datetime
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class FaceDatabase:
    """Database manager for face embeddings and metadata"""
    
    def __init__(self, db_path: str = "data/face_recognition.db"):
        """
        Initialize database connection
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._create_tables()
        
        logger.info(f"Database initialized: {db_path}")
    
    def _create_tables(self):
        """Create database tables if they don't exist"""
        cursor = self.conn.cursor()
        
        # Persons table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS persons (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                metadata TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Faces table (embeddings)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS faces (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                person_id INTEGER,
                embedding BLOB NOT NULL,
                image_path TEXT,
                bbox TEXT,
                age INTEGER,
                gender INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (person_id) REFERENCES persons(id) ON DELETE CASCADE
            )
        """)
        
        # Events table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_type TEXT NOT NULL,
                person_id INTEGER,
                face_id INTEGER,
                camera_id TEXT,
                similarity REAL,
                metadata TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (person_id) REFERENCES persons(id) ON DELETE SET NULL,
                FOREIGN KEY (face_id) REFERENCES faces(id) ON DELETE SET NULL
            )
        """)
        
        # Cameras table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS cameras (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                stream_url TEXT,
                stream_type TEXT,
                enabled BOOLEAN DEFAULT 1,
                metadata TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create indexes
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_faces_person_id ON faces(person_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_events_created_at ON events(created_at)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_events_event_type ON events(event_type)")
        
        self.conn.commit()
        logger.debug("Database tables created/verified")
    
    def add_person(self, name: str, metadata: Optional[Dict] = None) -> int:
        """
        Add a new person to the database
        
        Args:
            name: Person name
            metadata: Optional metadata dictionary
            
        Returns:
            Person ID
        """
        cursor = self.conn.cursor()
        metadata_json = json.dumps(metadata) if metadata else None
        
        cursor.execute("""
            INSERT INTO persons (name, metadata)
            VALUES (?, ?)
        """, (name, metadata_json))
        
        person_id = cursor.lastrowid
        self.conn.commit()
        logger.info(f"Added person: {name} (ID: {person_id})")
        return person_id
    
    def add_face(self, person_id: int, embedding: np.ndarray, 
                 image_path: Optional[str] = None, bbox: Optional[List[float]] = None,
                 age: Optional[int] = None, gender: Optional[int] = None) -> int:
        """
        Add a face embedding to the database
        
        Args:
            person_id: Person ID
            embedding: Face embedding (numpy array)
            image_path: Path to face image
            bbox: Bounding box [x1, y1, x2, y2]
            age: Age estimate
            gender: Gender (0=Female, 1=Male)
            
        Returns:
            Face ID
        """
        cursor = self.conn.cursor()
        
        # Convert embedding to bytes
        embedding_bytes = embedding.tobytes()
        bbox_json = json.dumps(bbox) if bbox else None
        
        cursor.execute("""
            INSERT INTO faces (person_id, embedding, image_path, bbox, age, gender)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (person_id, embedding_bytes, image_path, bbox_json, age, gender))
        
        face_id = cursor.lastrowid
        self.conn.commit()
        logger.debug(f"Added face for person {person_id} (Face ID: {face_id})")
        return face_id
    
    def get_person(self, person_id: int) -> Optional[Dict]:
        """Get person by ID"""
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM persons WHERE id = ?", (person_id,))
        row = cursor.fetchone()
        
        if row:
            return dict(row)
        return None
    
    def get_person_faces(self, person_id: int) -> List[Dict]:
        """Get all faces for a person"""
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM faces WHERE person_id = ?", (person_id,))
        rows = cursor.fetchall()
        
        faces = []
        for row in rows:
            face_dict = dict(row)
            # Convert embedding bytes back to numpy array
            if face_dict['embedding']:
                face_dict['embedding'] = np.frombuffer(face_dict['embedding'], dtype=np.float32)
            # Parse bbox JSON
            if face_dict['bbox']:
                face_dict['bbox'] = json.loads(face_dict['bbox'])
            faces.append(face_dict)
        
        return faces
    
    def get_all_embeddings(self) -> List[Tuple[int, int, np.ndarray]]:
        """
        Get all face embeddings from database
        
        Returns:
            List of (face_id, person_id, embedding) tuples
        """
        cursor = self.conn.cursor()
        cursor.execute("SELECT id, person_id, embedding FROM faces")
        rows = cursor.fetchall()
        
        embeddings = []
        for row in rows:
            face_id, person_id, embedding_bytes = row
            embedding = np.frombuffer(embedding_bytes, dtype=np.float32)
            embeddings.append((face_id, person_id, embedding))
        
        return embeddings
    
    def search_faces(self, query_embedding: np.ndarray, threshold: float = 0.6, 
                    max_results: int = 10) -> List[Dict]:
        """
        Search for similar faces in database
        
        Args:
            query_embedding: Query face embedding
            threshold: Similarity threshold
            max_results: Maximum results to return
            
        Returns:
            List of matching faces with similarity scores
        """
        all_embeddings = self.get_all_embeddings()
        
        if not all_embeddings:
            return []
        
        # Calculate similarities
        similarities = []
        query_norm = query_embedding / np.linalg.norm(query_embedding)
        
        for face_id, person_id, db_embedding in all_embeddings:
            db_norm = db_embedding / np.linalg.norm(db_embedding)
            similarity = float(np.dot(query_norm, db_norm))
            
            if similarity >= threshold:
                similarities.append({
                    'face_id': face_id,
                    'person_id': person_id,
                    'similarity': similarity
                })
        
        # Sort by similarity
        similarities.sort(key=lambda x: x['similarity'], reverse=True)
        
        # Get full face and person info
        results = []
        for match in similarities[:max_results]:
            face_info = self.get_face(match['face_id'])
            person_info = self.get_person(match['person_id'])
            
            if face_info and person_info:
                results.append({
                    'face_id': match['face_id'],
                    'person_id': match['person_id'],
                    'person_name': person_info['name'],
                    'similarity': match['similarity'],
                    'age': face_info.get('age'),
                    'gender': face_info.get('gender'),
                    'image_path': face_info.get('image_path')
                })
        
        return results
    
    def get_face(self, face_id: int) -> Optional[Dict]:
        """Get face by ID"""
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM faces WHERE id = ?", (face_id,))
        row = cursor.fetchone()
        
        if row:
            face_dict = dict(row)
            if face_dict['embedding']:
                face_dict['embedding'] = np.frombuffer(face_dict['embedding'], dtype=np.float32)
            if face_dict['bbox']:
                face_dict['bbox'] = json.loads(face_dict['bbox'])
            return face_dict
        return None
    
    def delete_person(self, person_id: int) -> bool:
        """Delete a person and all associated faces"""
        cursor = self.conn.cursor()
        cursor.execute("DELETE FROM persons WHERE id = ?", (person_id,))
        deleted = cursor.rowcount > 0
        self.conn.commit()
        
        if deleted:
            logger.info(f"Deleted person {person_id}")
        return deleted
    
    def delete_face(self, face_id: int) -> bool:
        """Delete a face"""
        cursor = self.conn.cursor()
        cursor.execute("DELETE FROM faces WHERE id = ?", (face_id,))
        deleted = cursor.rowcount > 0
        self.conn.commit()
        
        if deleted:
            logger.debug(f"Deleted face {face_id}")
        return deleted
    
    def list_persons(self) -> List[Dict]:
        """List all persons"""
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM persons ORDER BY created_at DESC")
        rows = cursor.fetchall()
        return [dict(row) for row in rows]
    
    def add_event(self, event_type: str, person_id: Optional[int] = None,
                  face_id: Optional[int] = None, camera_id: Optional[str] = None,
                  similarity: Optional[float] = None, metadata: Optional[Dict] = None):
        """Add an event to the database"""
        cursor = self.conn.cursor()
        metadata_json = json.dumps(metadata) if metadata else None
        
        cursor.execute("""
            INSERT INTO events (event_type, person_id, face_id, camera_id, similarity, metadata)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (event_type, person_id, face_id, camera_id, similarity, metadata_json))
        
        self.conn.commit()
        logger.debug(f"Added event: {event_type}")
    
    def get_events(self, limit: int = 100, event_type: Optional[str] = None) -> List[Dict]:
        """Get recent events"""
        cursor = self.conn.cursor()
        
        if event_type:
            cursor.execute("""
                SELECT * FROM events 
                WHERE event_type = ?
                ORDER BY created_at DESC 
                LIMIT ?
            """, (event_type, limit))
        else:
            cursor.execute("""
                SELECT * FROM events 
                ORDER BY created_at DESC 
                LIMIT ?
            """, (limit,))
        
        rows = cursor.fetchall()
        events = []
        for row in rows:
            event_dict = dict(row)
            if event_dict['metadata']:
                event_dict['metadata'] = json.loads(event_dict['metadata'])
            events.append(event_dict)
        
        return events
    
    def add_camera(self, camera_id: str, name: str, stream_url: Optional[str] = None,
                   stream_type: Optional[str] = None, metadata: Optional[Dict] = None):
        """Add or update a camera"""
        cursor = self.conn.cursor()
        metadata_json = json.dumps(metadata) if metadata else None
        
        cursor.execute("""
            INSERT OR REPLACE INTO cameras (id, name, stream_url, stream_type, metadata, updated_at)
            VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
        """, (camera_id, name, stream_url, stream_type, metadata_json))
        
        self.conn.commit()
        logger.info(f"Added/updated camera: {name} ({camera_id})")
    
    def get_cameras(self) -> List[Dict]:
        """Get all cameras"""
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM cameras WHERE enabled = 1")
        rows = cursor.fetchall()
        return [dict(row) for row in rows]
    
    def close(self):
        """Close database connection"""
        self.conn.close()
        logger.info("Database connection closed")

