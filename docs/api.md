# API Documentation

## Base URL

```
http://localhost:8000/api/v1
```

## Authentication

Some endpoints require API key authentication. Include the API key in the request header:

```
X-API-Key: your-api-key
```

Or as a query parameter:

```
?api_key=your-api-key
```

## Endpoints

### Health Check

**GET** `/health`

Check API health status.

**Response:**
```json
{
  "status": "healthy",
  "service": "face-recognition-api"
}
```

### Face Detection

**POST** `/face/detect`

Detect faces in an image.

**Request:**
- `image` (file): Image file

**Response:**
```json
{
  "success": true,
  "faces_detected": 1,
  "faces": [
    {
      "bbox": [x1, y1, x2, y2],
      "kps": [[x, y], ...],
      "det_score": 0.95,
      "embedding": [...],
      "age": 30,
      "gender": 1
    }
  ]
}
```

### Face Comparison

**POST** `/face/compare`

Compare two faces (1:1 verification).

**Request:**
- `image1` (file): First image
- `image2` (file): Second image

**Response:**
```json
{
  "success": true,
  "similarity": 0.85,
  "is_match": true,
  "threshold": 0.6
}
```

### Face Analysis

**POST** `/face/analyze`

Analyze face attributes.

**Request:**
- `image` (file): Image file

**Response:**
```json
{
  "success": true,
  "analysis": {
    "age": 30,
    "gender": "Male",
    "emotion": {...},
    "mask": {...},
    "quality": {...}
  }
}
```

### Face Identification

**POST** `/face/identify`

Identify face from database (1:N search).

**Request:**
- `image` (file): Image file
- `threshold` (optional): Similarity threshold (default: 0.6)
- `max_results` (optional): Maximum results (default: 10)

**Response:**
```json
{
  "success": true,
  "matches_found": 1,
  "matches": [
    {
      "person_id": 1,
      "person_name": "John Doe",
      "similarity": 0.85
    }
  ]
}
```

### Register Face

**POST** `/face/register`

Register a new face to database.

**Request:**
- `image` (file): Image file
- `name` (form): Person name

**Response:**
```json
{
  "success": true,
  "person_id": 1,
  "face_id": 1,
  "name": "John Doe"
}
```

### List Faces

**GET** `/face/list`

List all registered faces.

**Response:**
```json
{
  "success": true,
  "persons": [
    {
      "person_id": 1,
      "name": "John Doe",
      "faces_count": 1
    }
  ],
  "total": 1
}
```

### Delete Face

**DELETE** `/face/{person_id}`

Delete a person and all associated faces.

**Response:**
```json
{
  "success": true,
  "message": "Person 1 deleted"
}
```

### Liveness Detection

**POST** `/liveness/detect`

Detect face liveness.

**Request:**
- `image` (file): Image file

**Response:**
```json
{
  "success": true,
  "is_live": true,
  "confidence": 0.85,
  "methods": {
    "eye_blink": 0.9,
    "head_movement": 0.8,
    "texture_analysis": 0.85
  }
}
```

### Statistics

**GET** `/stats`

Get system statistics.

**Response:**
```json
{
  "success": true,
  "database": {
    "persons_count": 10,
    "events_count": 100
  },
  "system": {
    "cpu_percent": 25.5,
    "memory_percent": 45.2
  }
}
```

### Events

**GET** `/events`

Get event history.

**Query Parameters:**
- `limit` (optional): Number of events (default: 100)
- `type` (optional): Event type filter

**Response:**
```json
{
  "success": true,
  "events": [...],
  "count": 100
}
```

