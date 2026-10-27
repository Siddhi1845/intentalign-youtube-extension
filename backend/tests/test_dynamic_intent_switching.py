"""
End-to-End Dynamic Intent Switching and Session Isolation Test Suite (Section 36 and 52)
"""
import unittest
import json
from backend.app import app, SESSIONS

class TestDynamicIntentSwitching(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()
        SESSIONS.clear()

    def test_end_to_end_dynamic_intent_switching(self):
        # 1. Session 1 - Kubernetes & DevOps Goal
        s1_payload = {
            "session_id": "session_k8s_001",
            "profile": {
                "profession": "IT Student",
                "interests": ["Software Engineering", "Cloud Computing"],
                "hobbies": ["Cricket"]
            },
            "goal": "Prepare for Kubernetes and DevOps placement interview",
            "preferredTopics": ["Kubernetes", "Docker", "DevOps"],
            "unwantedTopics": ["Politics", "Gaming", "Bollywood"],
            "strictness": 0.65
        }
        res1 = self.client.post("/api/intent/analyze", data=json.dumps(s1_payload), content_type="application/json")
        self.assertEqual(res1.status_code, 200)
        s1_intent = res1.get_json()

        videos_s1 = [
            {"video_id": "v_k8s", "title": "Complete Kubernetes and Docker Tutorial for Beginners", "channel": "TechWorld"},
            {"video_id": "v_devops", "title": "DevOps CI/CD Automation Pipeline on Linux", "channel": "DevOpsPro"},
            {"video_id": "v_gta", "title": "GTA 5 Epic Car Stunts Gameplay Walkthrough", "channel": "GamerZone"},
            {"video_id": "v_bollywood", "title": "Latest Bollywood Hindi Movie Teaser and Songs", "channel": "FilmyHits"},
            {"video_id": "v_politics", "title": "Breaking Political Election News Debate", "channel": "NewsLive"},
            {"video_id": "v_cricket", "title": "India vs Australia Cricket Match Highlights", "channel": "SportsCenter"}
        ]

        batch_res1 = self.client.post("/api/video/analyze_batch", data=json.dumps({
            "session_id": "session_k8s_001",
            "intent": s1_intent,
            "videos": videos_s1,
            "strictness": 0.65
        }), content_type="application/json")
        self.assertEqual(batch_res1.status_code, 200)
        batch1_data = batch_res1.get_json()
        decisions_s1 = {r["video_id"]: r["decision"] for r in batch1_data["results"]}

        self.assertEqual(decisions_s1["v_k8s"], "ALLOW")
        self.assertEqual(decisions_s1["v_devops"], "ALLOW")
        self.assertEqual(decisions_s1["v_gta"], "SUPPRESS")
        self.assertEqual(decisions_s1["v_bollywood"], "SUPPRESS")
        self.assertEqual(decisions_s1["v_politics"], "SUPPRESS")
        self.assertIn(decisions_s1["v_cricket"], ["DEPRIORITIZE", "SUPPRESS"])

        # 2. Session 2 - Switch Goal to Python Data Science
        s2_payload = {
            "session_id": "session_ds_002",
            "profile": {
                "profession": "IT Student",
                "interests": ["Software Engineering", "Cloud Computing"],
                "hobbies": ["Cricket"]
            },
            "goal": "Learn Python for Data Science and Machine Learning",
            "preferredTopics": ["Python", "Data Science", "Pandas", "NumPy"],
            "unwantedTopics": ["Gaming", "Politics"],
            "strictness": 0.65
        }
        res2 = self.client.post("/api/intent/analyze", data=json.dumps(s2_payload), content_type="application/json")
        self.assertEqual(res2.status_code, 200)
        s2_intent = res2.get_json()

        videos_s2 = [
            {"video_id": "v_python", "title": "Python Pandas and NumPy Data Science Crash Course", "channel": "DataMaster"},
            {"video_id": "v_k8s_switch", "title": "Complete Kubernetes and Docker Tutorial for Beginners", "channel": "TechWorld"},
            {"video_id": "v_gta_switch", "title": "GTA 5 Epic Car Stunts Gameplay Walkthrough", "channel": "GamerZone"}
        ]

        batch_res2 = self.client.post("/api/video/analyze_batch", data=json.dumps({
            "session_id": "session_ds_002",
            "intent": s2_intent,
            "videos": videos_s2,
            "strictness": 0.65
        }), content_type="application/json")
        self.assertEqual(batch_res2.status_code, 200)
        batch2_data = batch_res2.get_json()
        decisions_s2 = {r["video_id"]: r["decision"] for r in batch2_data["results"]}

        self.assertEqual(decisions_s2["v_python"], "ALLOW")
        self.assertIn(decisions_s2["v_k8s_switch"], ["DEPRIORITIZE", "SUPPRESS"])
        self.assertEqual(decisions_s2["v_gta_switch"], "SUPPRESS")

        # 3. Session Isolation
        self.assertIn("session_k8s_001", SESSIONS)
        self.assertIn("session_ds_002", SESSIONS)
        self.assertEqual(len(SESSIONS["session_k8s_001"]["feed_history"]), 6)
        self.assertEqual(len(SESSIONS["session_ds_002"]["feed_history"]), 3)

        # Reset session 1
        reset_res = self.client.post("/api/session/reset", data=json.dumps({"session_id": "session_k8s_001"}), content_type="application/json")
        self.assertEqual(reset_res.status_code, 200)
        self.assertEqual(len(SESSIONS["session_k8s_001"]["feed_history"]), 0)
        self.assertEqual(len(SESSIONS["session_ds_002"]["feed_history"]), 3)

    def test_dynamic_novel_topic_space(self):
        web3_payload = {
            "session_id": "session_web3_001",
            "profile": {"profession": "Blockchain Engineer"},
            "goal": "Master Solidity Smart Contracts and Web3 Development",
            "preferredTopics": ["Solidity", "Smart Contracts", "Ethereum", "Web3"],
            "unwantedTopics": ["Gaming", "Cooking"],
            "strictness": 0.65
        }
        res = self.client.post("/api/intent/analyze", data=json.dumps(web3_payload), content_type="application/json")
        intent = res.get_json()

        videos = [
            {"video_id": "v_solidity", "title": "Solidity Smart Contract Security and Reentrancy Audit", "channel": "Web3Security"},
            {"video_id": "v_cooking", "title": "Italian Pizza and Pasta Cooking Recipe", "channel": "ChefKitchen"}
        ]

        batch_res = self.client.post("/api/video/analyze_batch", data=json.dumps({
            "session_id": "session_web3_001",
            "intent": intent,
            "videos": videos,
            "strictness": 0.65
        }), content_type="application/json")

        data = batch_res.get_json()
        decisions = {r["video_id"]: r["decision"] for r in data["results"]}
        self.assertEqual(decisions["v_solidity"], "ALLOW")
        self.assertEqual(decisions["v_cooking"], "SUPPRESS")

    def test_reversible_feedback_wrongly_hidden(self):
        intent_payload = {
            "session_id": "session_fb_001",
            "profile": {"profession": "Developer"},
            "goal": "Learn Kubernetes",
            "preferredTopics": ["Kubernetes"],
            "unwantedTopics": ["Politics"],
            "strictness": 0.65
        }
        res_intent = self.client.post("/api/intent/analyze", data=json.dumps(intent_payload), content_type="application/json")
        intent = res_intent.get_json()

        batch_res = self.client.post("/api/video/analyze_batch", data=json.dumps({
            "session_id": "session_fb_001",
            "intent": intent,
            "videos": [{"video_id": "v_borderline", "title": "Tech Industry Politics and Kubernetes Regulation", "channel": "TechPolicy"}],
            "strictness": 0.65
        }), content_type="application/json")
        self.assertEqual(batch_res.status_code, 200)

        fb_res = self.client.post("/api/feedback", data=json.dumps({
            "intent_profile": intent,
            "feedback": {
                "session_id": "session_fb_001",
                "video_id": "v_borderline",
                "feedback": "wrongly_hidden",
                "topics": {"Kubernetes": 0.8, "Politics": 0.2}
            }
        }), content_type="application/json")
        self.assertEqual(fb_res.status_code, 200)

        history = SESSIONS["session_fb_001"]["feed_history"]
        restored = [v for v in history if v["video_id"] == "v_borderline"][0]
        self.assertTrue(restored["controlled_visible"])
        self.assertEqual(restored["decision"], "RESTORED")

if __name__ == "__main__":
    unittest.main()
