"""Unit tests for src/app.py.

Covers merge_stacked_boxes (pure logic) directly, and the API routes via
FastAPI's TestClient with run_detection mocked out -- so these tests run in
CI without needing real model weights or a GPU.
"""
import io
from unittest.mock import patch

from fastapi.testclient import TestClient
from PIL import Image

from src.app import app, merge_stacked_boxes

client = TestClient(app)


class TestMergeStackedBoxes:
    def test_no_boxes_returns_empty(self) -> None:
        assert merge_stacked_boxes([], []) == []

    def test_single_box_unchanged(self) -> None:
        boxes = [[10.0, 10.0, 50.0, 50.0]]
        confs = [0.8]
        result = merge_stacked_boxes(boxes, confs)
        assert len(result) == 1
        assert result[0][0] == boxes[0]
        assert result[0][1] == 0.8

    def test_two_distant_boxes_stay_separate(self) -> None:
        boxes = [[0.0, 0.0, 20.0, 20.0], [500.0, 500.0, 520.0, 520.0]]
        confs = [0.7, 0.6]
        result = merge_stacked_boxes(boxes, confs)
        assert len(result) == 2

    def test_stacked_boxes_merge_into_one(self) -> None:
        # Simulates blades (top) + tower (bottom) of the same turbine: aligned
        # on the x-axis, touching vertically -- should merge into one box.
        blades = [40.0, 0.0, 80.0, 50.0]
        tower = [50.0, 50.0, 70.0, 150.0]
        result = merge_stacked_boxes([blades, tower], [0.9, 0.6])
        assert len(result) == 1
        merged_box, merged_conf = result[0]
        assert merged_box == [40.0, 0.0, 80.0, 150.0]
        assert merged_conf == 0.9  # keeps the higher confidence

    def test_side_by_side_boxes_stay_separate(self) -> None:
        # Two turbines standing side by side should NOT merge, even if close.
        left = [0.0, 0.0, 20.0, 100.0]
        right = [200.0, 0.0, 220.0, 100.0]
        result = merge_stacked_boxes([left, right], [0.8, 0.7])
        assert len(result) == 2


class TestRoutes:
    def test_health_check(self) -> None:
        response = client.get("/")
        assert response.status_code == 200
        assert response.json() == {"status": "Wind Turbine Detection API is running"}

    def test_frontend_serves_html(self) -> None:
        response = client.get("/app")
        assert response.status_code == 200
        assert "Wind Turbine Detection" in response.text

    @patch("src.app.run_detection")
    def test_detect_single_image_no_turbines(self, mock_run_detection) -> None:
        mock_run_detection.return_value = []
        image = Image.new("RGB", (100, 100), color="blue")
        buf = io.BytesIO()
        image.save(buf, format="JPEG")
        buf.seek(0)

        response = client.post("/detect", files={"file": ("test.jpg", buf, "image/jpeg")})
        assert response.status_code == 200
        data = response.json()
        assert data["turbine_count"] == 0
        assert data["message"] == "No turbines detected"

    @patch("src.app.run_detection")
    def test_detect_single_image_flags_low_confidence(self, mock_run_detection) -> None:
        mock_run_detection.return_value = [([10.0, 10.0, 50.0, 50.0], 0.20)]
        image = Image.new("RGB", (100, 100), color="blue")
        buf = io.BytesIO()
        image.save(buf, format="JPEG")
        buf.seek(0)

        response = client.post("/detect", files={"file": ("test.jpg", buf, "image/jpeg")})
        assert response.status_code == 200
        data = response.json()
        assert data["turbine_count"] == 1
        assert data["detections"][0]["needs_review"] is True

    def test_export_csv_empty_results(self) -> None:
        response = client.get("/detect-bulk/export-csv")
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/csv")