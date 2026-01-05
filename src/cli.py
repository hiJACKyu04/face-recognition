"""
CLI interface for Face Recognition System
"""

import click
import sys
import os
import logging
import subprocess
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.utils import load_config, setup_logging
from src.database import FaceDatabase
from src.face_engine import FaceEngine
import cv2
import numpy as np

# Setup logging
logger = setup_logging()

# Load configuration
config = load_config()


@click.group()
@click.version_option(version='1.0.0')
def cli():
    """Face Recognition System CLI"""
    pass


@cli.command()
def start():
    """Start all services"""
    click.echo("Starting Face Recognition services...")
    
    # Start Flask API
    click.echo("Starting Flask API...")
    flask_process = subprocess.Popen(
        [sys.executable, "flask/app.py"],
        cwd=Path(__file__).parent.parent
    )
    
    # Start Gradio UI
    click.echo("Starting Gradio UI...")
    gradio_process = subprocess.Popen(
        [sys.executable, "gradio/app.py"],
        cwd=Path(__file__).parent.parent
    )
    
    click.echo("Services started!")
    click.echo(f"Flask API: http://localhost:{config.get('api', {}).get('port', 8000)}")
    click.echo("Gradio UI: http://localhost:7860")
    
    try:
        flask_process.wait()
        gradio_process.wait()
    except KeyboardInterrupt:
        click.echo("\nStopping services...")
        flask_process.terminate()
        gradio_process.terminate()


@cli.command()
def stop():
    """Stop all services"""
    click.echo("Stopping services...")
    # TODO: Implement service stopping
    click.echo("Services stopped")


@cli.command()
@click.option('--name', required=True, help='Person name')
@click.option('--image', required=True, type=click.Path(exists=True), help='Path to face image')
def register_face(name, image):
    """Register a new face to database"""
    try:
        click.echo(f"Registering face for: {name}")
        
        # Load image
        img = cv2.imread(image)
        if img is None:
            click.echo(f"Error: Could not load image: {image}", err=True)
            return
        
        # Initialize components
        face_engine = FaceEngine(
            model_name=config.get('model', {}).get('name', 'buffalo_l'),
            det_size=tuple(config.get('model', {}).get('det_size', [640, 640])),
            det_thresh=config.get('model', {}).get('detection_threshold', 0.5)
        )
        
        db_config = config.get('database', {})
        database = FaceDatabase(db_path=db_config.get('path', 'data/face_recognition.db'))
        
        # Detect face
        faces = face_engine.detect_faces(img)
        if not faces:
            click.echo("Error: No face detected in image", err=True)
            return
        
        face = faces[0]
        embedding = np.array(face['embedding'])
        
        # Add to database
        person_id = database.add_person(name)
        face_id = database.add_face(
            person_id=person_id,
            embedding=embedding,
            bbox=face['bbox'],
            age=face.get('age'),
            gender=face.get('gender')
        )
        
        click.echo(f"Successfully registered face!")
        click.echo(f"Person ID: {person_id}")
        click.echo(f"Face ID: {face_id}")
        
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        logger.exception("Error registering face")


@cli.command()
def list_faces():
    """List all registered faces"""
    try:
        db_config = config.get('database', {})
        database = FaceDatabase(db_path=db_config.get('path', 'data/face_recognition.db'))
        
        persons = database.list_persons()
        
        if not persons:
            click.echo("No registered faces found")
            return
        
        click.echo(f"\nRegistered Faces ({len(persons)}):")
        click.echo("-" * 50)
        
        for person in persons:
            faces = database.get_person_faces(person['id'])
            click.echo(f"ID: {person['id']} | Name: {person['name']} | Faces: {len(faces)}")
        
    except Exception as e:
        click.echo(f"Error: {e}", err=True)


@cli.command()
@click.option('--person-id', required=True, type=int, help='Person ID to delete')
def delete_face(person_id):
    """Delete a person and all associated faces"""
    try:
        db_config = config.get('database', {})
        database = FaceDatabase(db_path=db_config.get('path', 'data/face_recognition.db'))
        
        if click.confirm(f'Delete person {person_id}?'):
            success = database.delete_person(person_id)
            if success:
                click.echo(f"Person {person_id} deleted")
            else:
                click.echo(f"Person {person_id} not found", err=True)
    
    except Exception as e:
        click.echo(f"Error: {e}", err=True)


@cli.command()
def status():
    """Show system status"""
    try:
        import psutil
        
        click.echo("System Status:")
        click.echo("-" * 50)
        click.echo(f"CPU Usage: {psutil.cpu_percent(interval=1)}%")
        
        memory = psutil.virtual_memory()
        click.echo(f"Memory Usage: {memory.percent}%")
        click.echo(f"Memory Available: {memory.available / (1024**3):.2f} GB")
        
        # Database status
        db_config = config.get('database', {})
        database = FaceDatabase(db_path=db_config.get('path', 'data/face_recognition.db'))
        persons = database.list_persons()
        events = database.get_events(limit=1)
        
        click.echo(f"\nDatabase:")
        click.echo(f"  Registered Persons: {len(persons)}")
        click.echo(f"  Total Events: {len(events)}")
        
    except Exception as e:
        click.echo(f"Error: {e}", err=True)


@cli.command()
@click.option('--key', help='Configuration key')
@click.option('--value', help='Configuration value')
def config_cmd(key, value):
    """View or modify configuration"""
    try:
        # Load config
        from src.utils import load_config
        app_config = load_config()
        
        if key and value:
            click.echo(f"Setting {key} = {value}")
            click.echo("Note: Config modification not yet implemented. Edit config/config.yaml directly.")
        else:
            click.echo("Current Configuration:")
            click.echo("-" * 50)
            import yaml
            click.echo(yaml.dump(app_config, default_flow_style=False))
    except Exception as e:
        click.echo(f"Error: {e}", err=True)


def main():
    """CLI entry point"""
    cli()


if __name__ == '__main__':
    main()

