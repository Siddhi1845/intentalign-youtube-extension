"""
Unit Tests for IntentService and Topic Normalization
"""
import unittest
from backend.services.embedding_service import EmbeddingService
from backend.services.keyword_service import KeywordService
from backend.services.intent_service import IntentService

class TestIntentService(unittest.TestCase):
    def setUp(self):
        self.emb = EmbeddingService()
        self.kwd = KeywordService(self.emb)
        self.service = IntentService(self.emb, self.kwd)

    def test_topic_normalization(self):
        self.assertEqual(self.service.normalize_topic("k8s"), "Kubernetes")
        self.assertEqual(self.service.normalize_topic("ml"), "Machine Learning")
        self.assertEqual(self.service.normalize_topic("ai"), "Artificial Intelligence")
        self.assertEqual(self.service.normalize_topic("ci/cd"), "CI/CD")
        self.assertEqual(self.service.normalize_topic("ci cd"), "CI/CD")
        self.assertEqual(self.service.normalize_topic("devops"), "DevOps")

    def test_duplicate_removal_and_filtering(self):
        raw = ["k8s", "Kubernetes", "kube", "docker", "Docker", "  ", None]
        cleaned = self.service.normalize_topic_list(raw)
        self.assertEqual(cleaned, ["Kubernetes", "Docker"])

    def test_derived_goal_topics(self):
        derived = self.service.derive_goal_concepts("Prepare for DevOps placement interview")
        topics = [d["topic"] for d in derived]
        self.assertIn("DevOps", topics)
        self.assertIn("Career", topics)
        self.assertTrue(len(derived) <= 12)

    def test_user_profile_separation(self):
        profile = self.service.build_intent_profile({
            "profile": {"profession": "IT Student", "interests": ["ai", "cloud"]},
            "goal": "Learn Kubernetes",
            "preferredTopics": ["k8s"],
            "unwantedTopics": ["politics"]
        })
        self.assertIn("user_profile", profile)
        self.assertIn("current_intent", profile)
        self.assertEqual(profile["user_profile"]["profession"], "IT Student")
        self.assertEqual(profile["current_intent"]["current_goal"], "Learn Kubernetes")

if __name__ == "__main__":
    unittest.main()
