"""
Integration Tests for Flask API Endpoints
"""
import unittest
import json
from backend.app import app

class TestAPI(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()

    def test_health_endpoint(self):
        res = self.client.get("/api/health")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data["status"], "ok")
        self.assertIn("embedding_model", data)

    def test_intent_analyze_endpoint(self):
        payload = {
            "profile": {"profession": "Developer"},
            "goal": "Prepare for DevOps placement interview",
            "preferredTopics": ["Docker", "Kubernetes"],
            "unwantedTopics": ["Politics"]
        }
        res = self.client.post("/api/intent/analyze", data=json.dumps(payload), content_type="application/json")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertIn("derivedTopics", data)
        self.assertIn("goalKeywords", data)

    def test_video_batch_analyze(self):
        intent_payload = {
            "profile": {"profession": "Developer"},
            "goal": "Learn Kubernetes",
            "preferredTopics": ["Kubernetes"],
            "unwantedTopics": ["Politics"]
        }
        res_intent = self.client.post("/api/intent/analyze", data=json.dumps(intent_payload), content_type="application/json")
        intent = res_intent.get_json()

        videos_payload = {
            "intent": intent,
            "videos": [
                {"video_id": "v1", "title": "Complete Kubernetes Tutorial for Beginners", "channel": "KubeDev"},
                {"video_id": "v2", "title": "Breaking Political Election News", "channel": "NewsLive"}
            ],
            "strictness": 0.65
        }
        res = self.client.post("/api/video/analyze_batch", data=json.dumps(videos_payload), content_type="application/json")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertIn("results", data)
        self.assertIn("feedAlignment", data)
        results = data["results"]
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]["decision"], "ALLOW")
        self.assertEqual(results[1]["decision"], "SUPPRESS")

    def test_feedback_endpoint(self):
        payload = {
            "intent_profile": {
                "current_intent": {"preferred_weights": {"Docker": 1.0}},
                "settings": {"feedback_learning_enabled": True}
            },
            "feedback": {
                "video_id": "v1",
                "feedback": "relevant",
                "topics": {"Docker": 0.9}
            }
        }
        res = self.client.post("/api/feedback", data=json.dumps(payload), content_type="application/json")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertIn("weight_changes", data)

if __name__ == "__main__":
    unittest.main()
