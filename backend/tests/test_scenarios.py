"""
Automated Test Suite for Guide's Eight Mandatory Test Scenarios (Section 52)
Verifies semantic intent alignment, hard negative suppression, and context-sensitive soft negative mitigation.
"""
import unittest
from backend.services.embedding_service import EmbeddingService
from backend.services.keyword_service import KeywordService
from backend.services.intent_service import IntentService
from backend.services.content_service import ContentService
from backend.services.alignment_service import AlignmentService
from backend.services.divergence_service import DivergenceService
from backend.services.intervention_service import InterventionService

class TestGuideScenarios(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.emb = EmbeddingService()
        cls.kwd = KeywordService(cls.emb)
        cls.intent_svc = IntentService(cls.emb, cls.kwd)
        cls.content_svc = ContentService(cls.emb, cls.kwd)
        cls.align_svc = AlignmentService()
        cls.diverg_svc = DivergenceService()
        cls.interv_svc = InterventionService()

    def _evaluate(self, goal, preferred, unwanted, title, channel="TestChannel"):
        intent = self.intent_svc.build_intent_profile({
            "goal": goal,
            "preferredTopics": preferred,
            "unwantedTopics": unwanted,
            "strictness": 0.65
        })
        video = {"video_id": "test_v", "title": title, "channel": channel, "description": ""}
        content_res = self.content_svc.analyze_content(video, intent)
        align_res = self.align_svc.compute_alignment(content_res, intent, strictness=0.65)
        feed_eval = self.diverg_svc.evaluate_feed(intent, [])
        feed_interv = self.interv_svc.compute_feed_intervention(feed_eval, strictness=0.65)
        dec_res = self.interv_svc.decide_video(content_res, align_res, feed_interv, strictness=0.65)
        return dec_res["decision"], align_res, content_res

    def test_scenario_1_kubernetes_course(self):
        # Goal: Learn Kubernetes -> Complete Kubernetes Course -> ALLOW
        dec, align, content = self._evaluate(
            goal="Learn Kubernetes",
            preferred=["Kubernetes", "Docker", "DevOps"],
            unwanted=["Politics", "Entertainment"],
            title="Complete Kubernetes Course for Beginners"
        )
        self.assertEqual(dec, "ALLOW", f"Expected ALLOW, got {dec} (align={align['alignment']})")

    def test_scenario_2_political_news(self):
        # Goal: Learn Kubernetes -> Breaking Political News -> SUPPRESS
        dec, align, content = self._evaluate(
            goal="Learn Kubernetes",
            preferred=["Kubernetes", "Docker"],
            unwanted=["Politics", "News"],
            title="Breaking Political News Election Rally and Vote Results"
        )
        self.assertEqual(dec, "SUPPRESS", f"Expected SUPPRESS, got {dec} (align={align['alignment']})")

    def test_scenario_3_devops_interview_questions(self):
        # Goal: Prepare for DevOps interview -> Kubernetes Security Interview Questions -> ALLOW
        dec, align, content = self._evaluate(
            goal="Prepare for DevOps interview",
            preferred=["DevOps", "Kubernetes", "Linux", "Docker"],
            unwanted=["Politics", "Celebrity Gossip", "Entertainment"],
            title="Kubernetes Security Interview Questions and Answers"
        )
        self.assertEqual(dec, "ALLOW", f"Expected ALLOW, got {dec} (align={align['alignment']})")

    def test_scenario_4_celebrity_gossip(self):
        # Goal: Prepare for DevOps interview -> Celebrity Gossip Today -> SUPPRESS
        dec, align, content = self._evaluate(
            goal="Prepare for DevOps interview",
            preferred=["DevOps", "Kubernetes"],
            unwanted=["Politics", "Celebrity Gossip", "Entertainment"],
            title="Celebrity Gossip and Bollywood Drama Today"
        )
        self.assertEqual(dec, "SUPPRESS", f"Expected SUPPRESS, got {dec} (align={align['alignment']})")

    def test_scenario_5_python_ml_tutorial(self):
        # Goal: Learn Python -> Python Machine Learning Tutorial -> ALLOW
        dec, align, content = self._evaluate(
            goal="Learn Python",
            preferred=["Python", "Programming", "Machine Learning"],
            unwanted=["Politics", "Sports", "Gaming"],
            title="Python Machine Learning Tutorial Full Course"
        )
        self.assertEqual(dec, "ALLOW", f"Expected ALLOW, got {dec} (align={align['alignment']})")

    def test_scenario_6_football_highlights(self):
        # Goal: Learn Python -> Football Highlights -> SUPPRESS or DEPRIORITIZE
        dec, align, content = self._evaluate(
            goal="Learn Python",
            preferred=["Python", "Programming"],
            unwanted=["Sports", "Gaming", "Politics"],
            title="Football Highlights Match Goals FIFA World Cup"
        )
        self.assertIn(dec, ["SUPPRESS", "DEPRIORITIZE"], f"Expected SUPPRESS or DEPRIORITIZE, got {dec}")

    def test_scenario_7_cybersecurity_news_update(self):
        # Goal: Learn Cybersecurity -> Cybersecurity News Update -> ALLOW if cybersecurity relevance is sufficiently high
        dec, align, content = self._evaluate(
            goal="Learn Cybersecurity",
            preferred=["Cybersecurity", "Ethical Hacking", "Network Security"],
            unwanted=["Entertainment", "Sports", "News"],
            title="Cybersecurity Vulnerability and Zero Day Threat News Update"
        )
        self.assertIn(dec, ["ALLOW", "DEPRIORITIZE"], f"Expected ALLOW or DEPRIORITIZE, got {dec}")

    def test_scenario_8_k8s_security_news_context(self):
        # Goal: Prepare for Kubernetes Security Interview -> Kubernetes Security News -> ALLOW or DEPRIORITIZE
        # Soft negative News should NOT automatically trigger hard suppression!
        dec, align, content = self._evaluate(
            goal="Prepare for Kubernetes Security Interview",
            preferred=["Kubernetes", "Cybersecurity", "DevOps"],
            unwanted=["News", "Entertainment"],
            title="Kubernetes Security News and Vulnerability Patch Guide"
        )
        self.assertIn(dec, ["ALLOW", "DEPRIORITIZE"], f"Expected ALLOW or DEPRIORITIZE, got {dec} (was wrongly hard-suppressed)")

if __name__ == "__main__":
    unittest.main()
