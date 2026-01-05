"""
Web Dashboard for Face Recognition System
"""

import streamlit as st
import sys
import os
import logging
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.utils import load_config, setup_logging
from src.database import FaceDatabase
import pandas as pd
from datetime import datetime

# Setup logging
logger = setup_logging()

# Load configuration
config = load_config()

# Page config
st.set_page_config(
    page_title="Face Recognition Dashboard",
    page_icon="👤",
    layout="wide"
)

# Initialize database
db_config = config.get('database', {})
database = FaceDatabase(db_path=db_config.get('path', 'data/face_recognition.db'))


def main():
    """Main dashboard"""
    st.title("👤 Face Recognition System Dashboard")
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a page",
        ["Overview", "Face Database", "Events", "Cameras", "Statistics"]
    )
    
    if page == "Overview":
        show_overview()
    elif page == "Face Database":
        show_face_database()
    elif page == "Events":
        show_events()
    elif page == "Cameras":
        show_cameras()
    elif page == "Statistics":
        show_statistics()


def show_overview():
    """Show overview page"""
    st.header("System Overview")
    
    # Get statistics
    persons = database.list_persons()
    events = database.get_events(limit=100)
    cameras = database.get_cameras()
    
    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Registered Persons", len(persons))
    
    with col2:
        total_faces = sum(len(database.get_person_faces(p['id'])) for p in persons)
        st.metric("Total Faces", total_faces)
    
    with col3:
        st.metric("Recent Events", len(events))
    
    with col4:
        st.metric("Active Cameras", len(cameras))
    
    # Recent events
    st.subheader("Recent Events")
    if events:
        events_df = pd.DataFrame(events)
        st.dataframe(events_df[['event_type', 'person_id', 'camera_id', 'created_at']], use_container_width=True)
    else:
        st.info("No events yet")


def show_face_database():
    """Show face database page"""
    st.header("Face Database")
    
    persons = database.list_persons()
    
    if persons:
        st.subheader(f"Registered Persons ({len(persons)})")
        
        for person in persons:
            with st.expander(f"Person ID: {person['id']} - {person['name']}"):
                faces = database.get_person_faces(person['id'])
                st.write(f"**Name:** {person['name']}")
                st.write(f"**Faces:** {len(faces)}")
                st.write(f"**Created:** {person['created_at']}")
                
                if st.button(f"Delete Person {person['id']}", key=f"delete_{person['id']}"):
                    database.delete_person(person['id'])
                    st.success(f"Person {person['id']} deleted")
                    st.rerun()
    else:
        st.info("No registered faces")


def show_events():
    """Show events page"""
    st.header("Event History")
    
    # Filters
    col1, col2 = st.columns(2)
    with col1:
        event_type_filter = st.selectbox(
            "Filter by type",
            ["All", "face_detected", "face_matched", "unknown_face", "liveness_passed", "liveness_failed"]
        )
    with col2:
        limit = st.slider("Number of events", 10, 500, 100)
    
    # Get events
    event_type = None if event_type_filter == "All" else event_type_filter
    events = database.get_events(limit=limit, event_type=event_type)
    
    if events:
        events_df = pd.DataFrame(events)
        st.dataframe(events_df, use_container_width=True)
    else:
        st.info("No events found")


def show_cameras():
    """Show cameras page"""
    st.header("Camera Management")
    
    cameras = database.get_cameras()
    
    if cameras:
        st.subheader(f"Active Cameras ({len(cameras)})")
        for camera in cameras:
            st.write(f"**{camera['name']}** ({camera['id']})")
            st.write(f"Stream: {camera.get('stream_url', 'N/A')}")
            st.write(f"Type: {camera.get('stream_type', 'N/A')}")
            st.divider()
    else:
        st.info("No cameras configured")
    
    # Add camera form
    st.subheader("Add Camera")
    with st.form("add_camera"):
        camera_id = st.text_input("Camera ID")
        camera_name = st.text_input("Camera Name")
        stream_url = st.text_input("Stream URL")
        stream_type = st.selectbox("Stream Type", ["rtsp", "webcam", "file"])
        
        if st.form_submit_button("Add Camera"):
            if camera_id and camera_name:
                database.add_camera(camera_id, camera_name, stream_url, stream_type)
                st.success(f"Camera {camera_name} added")
                st.rerun()


def show_statistics():
    """Show statistics page"""
    st.header("Statistics")
    
    persons = database.list_persons()
    events = database.get_events(limit=1000)
    
    # Event type distribution
    if events:
        event_types = [e['event_type'] for e in events]
        event_counts = pd.Series(event_types).value_counts()
        
        st.subheader("Event Type Distribution")
        st.bar_chart(event_counts)
    
    # Timeline
    if events:
        st.subheader("Events Timeline")
        events_df = pd.DataFrame(events)
        events_df['created_at'] = pd.to_datetime(events_df['created_at'])
        events_df = events_df.set_index('created_at')
        st.line_chart(events_df.groupby(events_df.index.date).size())


if __name__ == "__main__":
    main()

