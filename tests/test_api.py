"""
Tests for Flask API
"""

import pytest
from flask import Flask
from flask.testing import FlaskClient


def test_health_endpoint(client: FlaskClient):
    """Test health check endpoint"""
    response = client.get('/api/v1/health')
    assert response.status_code == 200
    assert 'status' in response.json


if __name__ == "__main__":
    pytest.main([__file__])

