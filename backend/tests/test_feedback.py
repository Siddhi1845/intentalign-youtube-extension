"""
Unit Tests for Feedback Learning Service
"""
import unittest
from backend.services.feedback_service import FeedbackService

class TestFeedbackService(unittest.TestCase):
    def setUp(self):
        self.service = FeedbackService(default_learning_rate=0.05, min_weight=0.10, max_weight=2.0)

    def test_relevant_increases_weight(self):
        intent = {
            "current_intent": {
                "preferred_weights": {"Docker": 1.0, "Kubernetes": 1.0},
                "unwanted_weights": {"Politics": 1.0}
            },
            "settings": {"feedback_learning_enabled": True, "learning_rate": 0.10}
        }
        fb = {
            "video_id": "v123",
            "title": "Docker in 100 Seconds",
            "feedback": "relevant",
            "topics": {"Docker": 0.90}
        }
        res = self.service.record_feedback(fb, intent)
        new_w = res["intent_profile"]["current_intent"]["preferred_weights"]["Docker"]
        self.assertGreater(new_w, 1.0, "Relevant feedback must increase preference weight")
        self.assertLessEqual(new_w, 2.0, "Weight must respect max_weight bound")

    def test_not_relevant_increases_unwanted_weight(self):
        intent = {
            "current_intent": {
                "preferred_weights": {"Docker": 1.0},
                "unwanted_weights": {"Entertainment": 1.0}
            },
            "settings": {"feedback_learning_enabled": True, "learning_rate": 0.10}
        }
        fb = {
            "video_id": "v456",
            "title": "Celebrity Vlog",
            "feedback": "not_relevant",
            "topics": {"Entertainment": 0.85}
        }
        res = self.service.record_feedback(fb, intent)
        new_w = res["intent_profile"]["current_intent"]["unwanted_weights"]["Entertainment"]
        self.assertGreater(new_w, 1.0, "Not relevant feedback must increase unwanted weight penalty")

if __name__ == "__main__":
    unittest.main()
