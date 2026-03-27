import tempfile
import unittest

from app import app as flask_app


class HealthRouteTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._temp_dir = tempfile.TemporaryDirectory()
        flask_app.config.update(
            TESTING=True,
            BENCHMARK_EXPORT_DIR=cls._temp_dir.name,
        )
        cls.client = flask_app.test_client()

    @classmethod
    def tearDownClass(cls):
        cls._temp_dir.cleanup()

    def test_live_health_endpoint(self):
        response = self.client.get("/api/v1/health/live")

        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertEqual(payload["status"], "success")
        self.assertEqual(payload["data"]["check"], "live")

    def test_ready_health_endpoint_degrades_cleanly(self):
        response = self.client.get("/api/v1/health/ready")

        self.assertIn(response.status_code, {200, 503})
        payload = response.get_json()
        self.assertEqual(payload["status"], "success")
        self.assertEqual(payload["data"]["check"], "ready")


if __name__ == "__main__":
    unittest.main()
