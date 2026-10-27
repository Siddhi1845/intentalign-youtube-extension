"""
Test Suite for Incidental Topics & Semantic Contradictions (Section 17)
Covers the 7 mandatory scenarios:
1. Tom and Jerry - Computer Trouble vs. Python programming goal
2. Kubernetes Security News vs. Kubernetes goal
3. Oggy Sports Fans vs. Kubernetes goal
4. Cricket Match Analysis vs. Sports analysis goal
5. Oggy and the Cockroaches - Sports Fans vs. Sports analysis goal
6. Java Interview Questions and Answers vs. Java interview goal
7. Java coffee recipe vs. Java interview goal
"""
import unittest
from backend.services.embedding_service import EmbeddingService
from backend.services.keyword_service import KeywordService
from backend.services.intent_service import IntentService
from backend.services.content_service import ContentService
from backend.services.alignment_service import AlignmentService
from backend.services.divergence_service import DivergenceService
from backend.services.intervention_service import InterventionService

class TestIncidentalTopics(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.emb = EmbeddingService()
        cls.kwd = KeywordService(cls.emb)
        cls.intent_svc = IntentService(cls.emb, cls.kwd)
        cls.content_svc = ContentService(cls.emb, cls.kwd)
        cls.align_svc = AlignmentService()
        cls.diverg_svc = DivergenceService()
        cls.interv_svc = InterventionService()

    def _evaluate(self, goal, preferred, unwanted, title, channel="TestChannel", description=""):
        intent = self.intent_svc.build_intent_profile({
            "goal": goal,
            "preferredTopics": preferred,
            "unwantedTopics": unwanted,
            "strictness": 0.65
        })
        video = {"video_id": "test_incidental", "title": title, "channel": channel, "description": description}
        content_res = self.content_svc.analyze_content(video, intent)
        align_res = self.align_svc.compute_alignment(content_res, intent, strictness=0.65)
        feed_eval = self.diverg_svc.evaluate_feed(intent, [])
        feed_interv = self.interv_svc.compute_feed_intervention(feed_eval, strictness=0.65)
        dec_res = self.interv_svc.decide_video(content_res, align_res, feed_interv, strictness=0.65)
        return dec_res, align_res, content_res

    def test_scenario_1_tom_jerry_computer(self):
        # 1. Goal: Learn Python programming + Video: Tom and Jerry - Computer Trouble
        # -> LOW / DEPRIORITIZE (NOT high programming match)
        dec, align, content = self._evaluate(
            goal="Learn Python programming",
            preferred=["Python", "Programming"],
            unwanted=["Entertainment"],
            title="Tom and Jerry - Computer Trouble",
            channel="WB Kids"
        )
        self.assertIn(dec["decision"], ["DEPRIORITIZE", "SUPPRESS"])
        self.assertNotEqual(dec["label"], "High Intent Match")
        self.assertTrue(content["is_incidental"])
        self.assertEqual(content["content_type"], "Cartoon")
        self.assertLess(align["alignment"], 0.35)

    def test_scenario_2_kubernetes_security_news(self):
        # 2. Goal: Learn Kubernetes + Video: Kubernetes Security News -> HIGH / ALLOW
        dec, align, content = self._evaluate(
            goal="Learn Kubernetes",
            preferred=["Kubernetes", "DevOps"],
            unwanted=["Entertainment"],
            title="Kubernetes Security News and Vulnerability Patching Guide",
            channel="CloudNative Tech"
        )
        self.assertEqual(dec["decision"], "ALLOW")
        self.assertIn(dec["label"], ["High Intent Match", "Contextual Goal Match"])
        self.assertFalse(content["is_incidental"])
        self.assertGreaterEqual(align["alignment"], 0.50)

    def test_scenario_3_oggy_sports_vs_kubernetes(self):
        # 3. Goal: Learn Kubernetes + Video: Oggy Sports Fans -> SUPPRESS / LOW
        dec, align, content = self._evaluate(
            goal="Learn Kubernetes",
            preferred=["Kubernetes", "DevOps"],
            unwanted=["Entertainment"],
            title="Oggy Sports Fans",
            channel="Oggy Hindi"
        )
        self.assertEqual(dec["decision"], "SUPPRESS")
        self.assertEqual(dec["label"], "Low Intent Alignment")
        self.assertLess(align["alignment"], 0.15)

    def test_scenario_4_cricket_match_analysis(self):
        # 4. Goal: Sports analysis + Video: Cricket Match Analysis -> HIGH / ALLOW
        dec, align, content = self._evaluate(
            goal="Sports analysis",
            preferred=["Sports"],
            unwanted=["Politics"],
            title="Cricket Match Analysis and Team Tactics",
            channel="Cricket Insider"
        )
        self.assertEqual(dec["decision"], "ALLOW")
        self.assertEqual(dec["label"], "High Intent Match")
        self.assertEqual(content["content_type"], "Sports")
        self.assertGreaterEqual(align["alignment"], 0.65)

    def test_scenario_5_oggy_sports_vs_sports_goal(self):
        # 5. Goal: Sports analysis + Video: Oggy and the Cockroaches - Sports Fans
        # -> LOW / DEPRIORITIZE (NOT high match, flagged incidental)
        dec, align, content = self._evaluate(
            goal="Sports analysis",
            preferred=["Sports"],
            unwanted=["Politics"],
            title="Oggy and the Cockroaches - Sports Fans",
            channel="Oggy Cartoon"
        )
        self.assertIn(dec["decision"], ["DEPRIORITIZE", "SUPPRESS"])
        self.assertNotEqual(dec["label"], "High Intent Match")
        self.assertTrue(content["is_incidental"])
        self.assertEqual(content["content_type"], "Cartoon")
        self.assertIn("Cartoon", dec["reason"])

    def test_scenario_6_java_interview_qa(self):
        # 6. Goal: Prepare for Java interview + Video: Java Interview Questions and Answers -> HIGH / ALLOW
        dec, align, content = self._evaluate(
            goal="Prepare for Java interview",
            preferred=["Java", "Programming"],
            unwanted=["Entertainment"],
            title="Java Interview Questions and Answers Technical Round",
            channel="Java Brains"
        )
        self.assertEqual(dec["decision"], "ALLOW")
        self.assertEqual(dec["label"], "High Intent Match")
        self.assertGreaterEqual(align["alignment"], 0.70)
        self.assertFalse(content["is_incidental"])

    def test_scenario_7_java_coffee_recipe(self):
        # 7. Goal: Prepare for Java interview + Video: Java coffee recipe -> LOW / SUPPRESS
        dec, align, content = self._evaluate(
            goal="Prepare for Java interview",
            preferred=["Java", "Programming"],
            unwanted=["Entertainment"],
            title="Java coffee recipe and brewing guide for baristas",
            channel="Coffee Master"
        )
        self.assertIn(dec["decision"], ["SUPPRESS", "DEPRIORITIZE"])
        self.assertNotEqual(dec["label"], "High Intent Match")
        self.assertTrue(content["is_incidental"])
        self.assertEqual(content["content_type"], "Cooking")
        self.assertLess(align["alignment"], 0.25)

if __name__ == "__main__":
    unittest.main()
