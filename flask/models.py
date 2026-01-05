"""
Database models for Flask API
"""

from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Boolean, Text, ForeignKey, BLOB
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from datetime import datetime

Base = declarative_base()


class Person(Base):
    """Person model"""
    __tablename__ = 'persons'
    
    id = Column(Integer, primary_key=True)
    name = Column(String(255), nullable=False)
    metadata = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    faces = relationship("Face", back_populates="person", cascade="all, delete-orphan")


class Face(Base):
    """Face model"""
    __tablename__ = 'faces'
    
    id = Column(Integer, primary_key=True)
    person_id = Column(Integer, ForeignKey('persons.id'), nullable=True)
    embedding = Column(BLOB, nullable=False)
    image_path = Column(String(500))
    bbox = Column(Text)
    age = Column(Integer)
    gender = Column(Integer)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    person = relationship("Person", back_populates="faces")


class Event(Base):
    """Event model"""
    __tablename__ = 'events'
    
    id = Column(Integer, primary_key=True)
    event_type = Column(String(50), nullable=False)
    person_id = Column(Integer, ForeignKey('persons.id'), nullable=True)
    face_id = Column(Integer, ForeignKey('faces.id'), nullable=True)
    camera_id = Column(String(100))
    similarity = Column(Float)
    metadata = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)


class Camera(Base):
    """Camera model"""
    __tablename__ = 'cameras'
    
    id = Column(String(100), primary_key=True)
    name = Column(String(255), nullable=False)
    stream_url = Column(String(500))
    stream_type = Column(String(50))
    enabled = Column(Boolean, default=True)
    metadata = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

