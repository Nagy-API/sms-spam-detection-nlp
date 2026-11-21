import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from textwrap import dedent
from unittest.mock import patch

from fastapi.testclient import TestClient

from app import main


HAM = "See you at lunch tomorrow"
SPAM = "Congratulations! You won a free prize. Call now to claim."


class ApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = cls.enterClassContext(TestClient(main.app))

    def test_health(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "ok"})

    def test_predictions_match_the_existing_model(self):
        for text, label, probability in (
            (HAM, "ham", 0.0005),
            (SPAM, "spam", 1.0),
            ("12345", "ham", 0.1262),
        ):
            with self.subTest(text=text):
                response = self.client.post("/predict", json={"text": text})
                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.json(), {"label": label, "spam_probability": probability})

    def test_frontend_can_reach_prediction_at_the_same_origin(self):
        page = self.client.get("/")
        self.assertEqual(page.status_code, 200)
        self.assertIn("text/html", page.headers["content-type"])
        self.assertIn('src="script.js"', page.text)
        self.assertEqual(self.client.get("/style.css").status_code, 200)
        script = self.client.get("/script.js")
        self.assertEqual(script.status_code, 200)
        endpoint = re.search(r'const API_URL = "([^"]+)"', script.text)
        self.assertIsNotNone(endpoint)
        self.assertRegex(endpoint[1], r"^/[^/]")
        response = self.client.post(endpoint[1], json={"text": HAM})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["label"], "ham")


class StartupTests(unittest.TestCase):
    def test_models_load_from_outside_the_project_directory(self):
        project = str(Path(main.__file__).resolve().parents[1])
        program = dedent("""\
            import sys
            sys.path.insert(0, sys.argv[1])
            from app.main import app
            from fastapi.testclient import TestClient

            with TestClient(app) as client:
                response = client.post('/predict', json={'text': 'See you at lunch tomorrow'})
                assert response.status_code == 200
                assert response.json() == {'label': 'ham', 'spam_probability': 0.0005}
            """)
        with tempfile.TemporaryDirectory() as directory:
            check = subprocess.run(
                [sys.executable, "-c", program, project],
                cwd=directory,
                capture_output=True,
                text=True,
                timeout=30,
            )
        self.assertEqual(check.returncode, 0, check.stderr)

    def test_startup_fails_if_either_artifact_is_missing(self):
        with tempfile.TemporaryDirectory() as directory:
            for artifact in ("MODEL_PATH", "VECTORIZER_PATH"):
                missing = Path(directory) / "missing.pkl"
                with self.subTest(artifact=artifact), patch.object(main, artifact, missing):
                    with self.assertRaisesRegex(FileNotFoundError, "missing.pkl"):
                        with TestClient(main.app):
                            pass
