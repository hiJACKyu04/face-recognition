# System Architecture Documentation

## Table of Contents
1. [Overview](#overview)
2. [System Architecture](#system-architecture)
3. [Component Details](#component-details)
4. [Data Flow](#data-flow)
5. [Database Schema](#database-schema)
6. [API Architecture](#api-architecture)
7. [Deployment Architecture](#deployment-architecture)
8. [Security Architecture](#security-architecture)
9. [Performance Considerations](#performance-considerations)

---

## Overview

The Face Recognition System is a comprehensive, cross-platform solution designed for face detection, recognition, liveness detection, and video surveillance. It provides multiple interfaces (REST API, Web UI, CLI) and supports various deployment scenarios from single-machine setups to distributed systems.

### Key Design Principles

- **Modularity**: Each component is independently deployable and testable
- **Scalability**: Supports horizontal scaling through stateless API design
- **Extensibility**: Plugin-based architecture for custom integrations
- **Performance**: Optimized for real-time processing with GPU support
- **Security**: Built-in authentication, rate limiting, and data encryption

---

## System Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Client Layer                              │
├─────────────────────────────────────────────────────────────────┤
│  Web Browser  │  Mobile App  │  CLI  │  Third-party Services  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Interface Layer                            │
├─────────────────────────────────────────────────────────────────┤
│  Flask API  │  Gradio UI  │  CLI  │  Web Dashboard  │  Video   │
│  (REST)     │  (Web UI)   │       │  (Monitoring)   │  Stream  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Application Layer                          │
├─────────────────────────────────────────────────────────────────┤
│  Route Handlers  │  Middleware  │  Authentication  │  Rate Limit│
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                       Core Engine Layer                          │
├─────────────────────────────────────────────────────────────────┤
│  FaceEngine  │  LivenessDetector  │  FaceAnalyzer  │  Tracker   │
│  Database    │  EventManager     │  StreamProcessor│  Annotator│
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Model & Data Layer                         │
├─────────────────────────────────────────────────────────────────┤
│  InsightFace  │  RetinaFace  │  SQLite/PostgreSQL  │  File System│
│  (buffalo_l)  │  (Detection) │  (Face DB)          │  (Storage)  │
└─────────────────────────────────────────────────────────────────┘
```

### Component Interaction Diagram

```
┌──────────────┐
│   Client     │
└──────┬───────┘
       │ HTTP/WebSocket
       ▼
┌─────────────────────────────────────┐
│         Flask API Server            │
│  ┌──────────────┐  ┌─────────────┐ │
│  │  Middleware  │  │   Routes    │ │
│  │  - Auth      │  │  - Face     │ │
│  │  - CORS      │  │  - Liveness │ │
│  │  - RateLimit │  │  - Admin   │ │
│  └──────┬───────┘  └──────┬──────┘ │
└─────────┼──────────────────┼────────┘
          │                  │
          ▼                  ▼
┌─────────────────┐  ┌──────────────────┐
│   FaceEngine     │  │  LivenessDetector│
│  - Detection     │  │  - Eye Blink     │
│  - Recognition   │  │  - Head Movement │
│  - Embedding    │  │  - Texture       │
└────────┬─────────┘  └────────┬─────────┘
         │                      │
         └──────────┬────────────┘
                    ▼
         ┌──────────────────────┐
         │    FaceDatabase      │
         │  - Persons           │
         │  - Faces             │
         │  - Events            │
         └──────────────────────┘
```

---

## Component Details

### 1. FaceEngine (`src/face_engine.py`)

**Purpose**: Core face detection and recognition engine

**Responsibilities**:
- Face detection using RetinaFace
- Face embedding extraction (512-dimensional vectors)
- Face comparison and similarity calculation
- Age and gender estimation

**Key Methods**:
- `detect_faces(image)`: Detect all faces in an image
- `extract_embedding(face)`: Extract face embedding vector
- `compare_faces(embedding1, embedding2)`: Calculate similarity
- `get_face_attributes(face)`: Extract age, gender

**Dependencies**:
- InsightFace library
- ONNX Runtime (CPU/CUDA/CoreML)
- NumPy, OpenCV

**Configuration**:
```yaml
model:
  name: buffalo_l
  detection_threshold: 0.5
  recognition_threshold: 0.6
  det_size: [640, 640]
```

### 2. LivenessDetector (`src/liveness_detector.py`)

**Purpose**: Detect if a face is live (not a photo/video spoof)

**Methods**:
- Eye blink detection
- Head movement analysis
- Texture analysis (detect print/display artifacts)

**Algorithm**:
1. Track facial landmarks across frames
2. Analyze eye aspect ratio (EAR) for blinks
3. Calculate head pose changes
4. Analyze texture patterns for spoofing

**Configuration**:
```yaml
liveness:
  enabled: true
  methods: [eye_blink, head_movement, texture_analysis]
  threshold: 0.7
  min_blinks: 1
```

### 3. FaceAnalyzer (`src/face_analyzer.py`)

**Purpose**: Extract detailed face attributes

**Capabilities**:
- Age estimation
- Gender classification
- Emotion detection
- Mask detection
- Face quality assessment

### 4. FaceDatabase (`src/database.py`)

**Purpose**: Store and manage face data

**Schema**:
- **persons**: Person information (id, name, created_at)
- **faces**: Face embeddings (id, person_id, embedding, metadata)
- **events**: Event log (id, type, person_id, timestamp, data)
- **cameras**: Camera configuration

**Operations**:
- Register faces
- Search faces (1:N identification)
- Delete persons/faces
- Query events

**Supported Databases**:
- SQLite (default, embedded)
- PostgreSQL (production, scalable)

### 5. FaceTracker (`src/face_tracker.py`)

**Purpose**: Track faces across video frames

**Algorithm**:
- IoU (Intersection over Union) matching for bounding boxes
- Embedding similarity matching for identity
- Track management with age and hit counting
- Appearance segment tracking

**Features**:
- Multi-person tracking
- Occlusion handling
- Track lifecycle management
- Statistics generation

### 6. VideoAnnotator (`src/video_annotator.py`)

**Purpose**: Generate annotated videos with detection results

**Features**:
- Draw bounding boxes
- Display person names and similarity scores
- Show tracking IDs
- Draw movement trajectories
- Add timestamps

### 7. StreamProcessor (`src/stream_processor.py`)

**Purpose**: Process video streams from various sources

**Supported Sources**:
- Webcam (USB cameras)
- RTSP streams (IP cameras)
- Video files (MP4, AVI, MOV, etc.)
- HTTP streams

**Features**:
- Frame skipping for performance
- Queue-based processing
- Multi-threaded processing
- Automatic reconnection

### 8. EventManager (`src/event_manager.py`)

**Purpose**: Handle events and notifications

**Event Types**:
- `FACE_DETECTED`: Face detected in stream
- `FACE_IDENTIFIED`: Known person identified
- `UNKNOWN_FACE`: Unknown face detected
- `LIVENESS_FAILED`: Spoofing attempt detected

**Notification Channels**:
- Email
- Webhook (HTTP POST)
- MQTT
- File logging

---

## Data Flow

### Face Registration Flow

```
1. User uploads image
   ↓
2. Flask API receives request
   ↓
3. FaceEngine detects face
   ↓
4. Extract face embedding (512-dim vector)
   ↓
5. Store in database:
   - Create person record
   - Store face embedding
   - Link face to person
   ↓
6. Return success response
```

### Face Identification Flow

```
1. User uploads image/video
   ↓
2. FaceEngine detects faces
   ↓
3. Extract embeddings for each face
   ↓
4. Database search:
   - Compare with all stored embeddings
   - Calculate cosine similarity
   - Filter by threshold
   - Sort by similarity
   ↓
5. Return matches with similarity scores
```

### Video Processing Flow

```
1. Video stream/file input
   ↓
2. StreamProcessor reads frames
   ↓
3. For each frame (with frame_skip):
   a. FaceEngine detects faces
   b. FaceTracker updates tracks
   c. Database search (if identify=True)
   d. VideoAnnotator draws annotations
   ↓
4. Generate statistics:
   - Unique people detected
   - Total appearances
   - Time on screen
   ↓
5. Return results + annotated video
```

### Liveness Detection Flow

```
1. Receive image sequence
   ↓
2. Track facial landmarks
   ↓
3. Analyze:
   - Eye aspect ratio (EAR) → Blink detection
   - Head pose changes → Movement
   - Texture patterns → Spoofing
   ↓
4. Combine scores
   ↓
5. Return liveness result
```

---

## Database Schema

### Persons Table

```sql
CREATE TABLE persons (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    metadata TEXT  -- JSON metadata
);
```

### Faces Table

```sql
CREATE TABLE faces (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    person_id INTEGER NOT NULL,
    embedding BLOB NOT NULL,  -- 512-dim float32 array
    image_path TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    metadata TEXT,  -- JSON metadata
    FOREIGN KEY (person_id) REFERENCES persons(id) ON DELETE CASCADE
);

CREATE INDEX idx_faces_person_id ON faces(person_id);
```

### Events Table

```sql
CREATE TABLE events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    type TEXT NOT NULL,
    person_id INTEGER,
    camera_id TEXT,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    data TEXT,  -- JSON event data
    FOREIGN KEY (person_id) REFERENCES persons(id) ON DELETE SET NULL
);

CREATE INDEX idx_events_type ON events(type);
CREATE INDEX idx_events_timestamp ON events(timestamp);
CREATE INDEX idx_events_person_id ON events(person_id);
```

### Cameras Table

```sql
CREATE TABLE cameras (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    source TEXT NOT NULL,  -- RTSP URL, webcam index, etc.
    type TEXT NOT NULL,  -- rtsp, webcam, file
    enabled BOOLEAN DEFAULT 1,
    config TEXT  -- JSON configuration
);
```

---

## API Architecture

### RESTful Design

**Base URL**: `http://localhost:8000/api/v1`

**Versioning**: URL-based (`/api/v1/`)

**Authentication**:
- API Key in header: `X-API-Key: <key>`
- API Key in query: `?api_key=<key>`
- Configurable per endpoint

**Rate Limiting**:
- Per-IP or per-API-key
- Configurable limits (default: 60 requests/minute)
- In-memory or Redis backend

**CORS**:
- Configurable origins
- Preflight support
- Credentials support

### Endpoint Categories

1. **Face Operations** (`/face/*`)
   - Detection
   - Comparison (1:1)
   - Identification (1:N)
   - Registration
   - Analysis
   - Video processing

2. **Liveness** (`/liveness/*`)
   - Detection

3. **Surveillance** (`/surveillance/*`)
   - Status
   - Stream management

4. **Admin** (`/admin/*`)
   - Statistics
   - Events
   - Configuration

### Request/Response Format

**Request**:
- Content-Type: `multipart/form-data` (for images)
- JSON body (for parameters)

**Response**:
```json
{
  "success": true,
  "data": {...},
  "error": null
}
```

**Error Response**:
```json
{
  "success": false,
  "error": "Error message",
  "code": 400
}
```

---

## Deployment Architecture

### Single-Machine Deployment

```
┌─────────────────────────────────────┐
│         Single Server               │
│  ┌──────────┐  ┌──────────┐        │
│  │  Flask   │  │  Gradio  │        │
│  │  API     │  │  UI      │        │
│  └────┬─────┘  └────┬─────┘        │
│       │             │               │
│       └──────┬──────┘               │
│              ▼                      │
│       ┌──────────────┐              │
│       │  SQLite DB  │              │
│       └──────────────┘              │
└─────────────────────────────────────┘
```

### Docker Deployment

```
┌─────────────────────────────────────────┐
│         Docker Compose                  │
│  ┌──────────┐  ┌──────────┐            │
│  │  Flask   │  │  Gradio  │            │
│  │  API     │  │  UI      │            │
│  └────┬─────┘  └────┬─────┘            │
│       │             │                   │
│       └──────┬──────┘                   │
│              ▼                          │
│  ┌──────────────────────┐              │
│  │   PostgreSQL         │              │
│  │   (Container)        │              │
│  └──────────────────────┘              │
└─────────────────────────────────────────┘
```

### Production Deployment

```
┌─────────────┐     ┌─────────────┐
│   Nginx     │────▶│  Flask API  │
│  (Reverse   │     │  (Gunicorn) │
│   Proxy)    │     └──────┬──────┘
└─────────────┘            │
                           ▼
                    ┌──────────────┐
                    │  PostgreSQL  │
                    │  (Primary)   │
                    └──────────────┘
                           │
                           ▼
                    ┌──────────────┐
                    │  PostgreSQL  │
                    │  (Replica)   │
                    └──────────────┘
```

---

## Security Architecture

### Authentication & Authorization

- **API Key Authentication**: Per-request API key validation
- **Rate Limiting**: Prevent abuse and DoS attacks
- **CORS**: Control cross-origin access
- **Input Validation**: Sanitize all inputs

### Data Protection

- **Face Embeddings**: Stored as binary blobs
- **Database Encryption**: Optional encryption at rest
- **Secure Transmission**: HTTPS/TLS recommended
- **Access Control**: Database-level permissions

### Privacy Considerations

- **Data Retention**: Configurable retention policies
- **Anonymization**: Optional face blurring
- **Access Logging**: Audit trail for all operations
- **GDPR Compliance**: Data deletion capabilities

---

## Performance Considerations

### Optimization Strategies

1. **Model Optimization**:
   - ONNX Runtime with optimized providers
   - Batch processing where possible
   - Frame skipping for video

2. **Database Optimization**:
   - Indexed face embeddings
   - Connection pooling
   - Query optimization

3. **Caching**:
   - Model caching (loaded once)
   - Database connection pooling
   - Response caching for static data

4. **Parallel Processing**:
   - Multi-threaded frame processing
   - Async API endpoints
   - Background task queues

### Performance Metrics

- **Face Detection**: ~50-100ms per image (CPU)
- **Face Recognition**: ~10-20ms per comparison
- **Video Processing**: ~1-2 seconds per 100 frames
- **API Latency**: <100ms (p95)

### Scalability

- **Horizontal Scaling**: Stateless API design
- **Load Balancing**: Multiple API instances
- **Database Scaling**: Read replicas for queries
- **GPU Acceleration**: CUDA/CoreML support

---

## Technology Stack

### Core Technologies

- **Python 3.11+**: Main programming language
- **InsightFace**: Face recognition library
- **RetinaFace**: Face detection model
- **ONNX Runtime**: Model inference engine
- **OpenCV**: Image/video processing
- **Flask**: Web framework
- **Gradio**: Interactive UI
- **SQLAlchemy**: ORM
- **SQLite/PostgreSQL**: Database

### Dependencies

- **NumPy**: Numerical computing
- **Pillow**: Image processing
- **Requests**: HTTP client
- **Click**: CLI framework

---

## Future Enhancements

1. **Distributed Processing**: Multi-node face recognition
2. **Real-time Streaming**: WebSocket support
3. **Mobile SDK**: iOS/Android integration
4. **Cloud Integration**: AWS/Azure/GCP deployment
5. **Advanced Analytics**: ML-based insights
6. **Federated Learning**: Privacy-preserving training
