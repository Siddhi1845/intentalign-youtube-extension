"""
20-Video End-to-End Acceptance Test Run (Section 21)
Evaluates a realistic 20-video YouTube feed against a student interview-prep intent.
Verifies complete 11-field schema and precise decision boundaries for every video card.
"""
import unittest
import json
from backend.app import app, SESSIONS

class TestAcceptance20Videos(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()
        SESSIONS.clear()

    def test_20_video_acceptance_feed(self):
        # 1. Configure user intent profile
        intent_payload = {
            "session_id": "session_acceptance_001",
            "profile": {
                "profession": "Computer Science Undergraduate",
                "interests": ["Software Engineering", "Cloud Computing"],
                "hobbies": ["Gaming"]
            },
            "goal": "Prepare for Kubernetes and DevOps Placement Interview",
            "preferredTopics": ["Kubernetes", "DevOps", "Docker", "CI/CD", "Linux"],
            "unwantedTopics": ["Politics", "Celebrity Gossip", "Kids", "Entertainment"],
            "strictness": 0.65
        }
        res_intent = self.client.post("/api/intent/analyze", data=json.dumps(intent_payload), content_type="application/json")
        self.assertEqual(res_intent.status_code, 200)
        intent = res_intent.get_json()

        # 2. Define 20 realistic feed videos
        raw_feed = [
            {"video_id": "v01", "title": "Kubernetes Cluster Architecture and Pod Networking Deep Dive", "channel": "TechWorld"},
            {"video_id": "v02", "title": "DevOps CI/CD Pipeline Automation with Docker and Jenkins", "channel": "DevOps Directive"},
            {"video_id": "v03", "title": "Kubernetes Security Best Practices & Container Hardening", "channel": "CloudNative Security"},
            {"video_id": "v04", "title": "Top 50 DevOps & Kubernetes Interview Questions and Answers", "channel": "Edureka"},
            {"video_id": "v05", "title": "Linux System Administration Crash Course for DevOps Engineers", "channel": "freeCodeCamp"},
            {"video_id": "v06", "title": "Docker Compose Multi-Container Deployment Tutorial", "channel": "Traversy Media"},
            {"video_id": "v07", "title": "Kubernetes Security Vulnerability News and CVE Patching", "channel": "K8s News"},
            {"video_id": "v08", "title": "Python Scripting for DevOps and Cloud Automation", "channel": "Tech with Tim"},
            {"video_id": "v09", "title": "System Design Primer: Scalable Microservices Architecture", "channel": "Gaurav Sen"},
            {"video_id": "v10", "title": "Tom and Jerry - Computer Trouble Cartoon Episode", "channel": "WB Kids"},
            {"video_id": "v11", "title": "Oggy and the Cockroaches - Sports Fans", "channel": "Oggy Hindi"},
            {"video_id": "v12", "title": "Docker Boat Harbor Cruise and Ship Docking at Marina", "channel": "TravelPort"},
            {"video_id": "v13", "title": "Breaking Political Election News Debate and Party Rallies", "channel": "News24"},
            {"video_id": "v14", "title": "Celebrity Gossip: Inside Hollywood Romance and Red Carpet Drama", "channel": "Entertainment Weekly"},
            {"video_id": "v15", "title": "GTA 5 Epic Car Stunts and Funny Moments Gameplay", "channel": "GamerZone"},
            {"video_id": "v16", "title": "Peppa Pig Visits the Science Museum Kids Cartoon", "channel": "Peppa Pig Official"},
            {"video_id": "v17", "title": "Traditional Italian Pizza Dough and Baking Recipe", "channel": "Chef Kitchen"},
            {"video_id": "v18", "title": "Lionel Messi Best Goals & Skills in Champions League", "channel": "Football World"},
            {"video_id": "v19", "title": "Vague Unclassified Short Clip Without Context", "channel": "RandomUser"},
            {"video_id": "v20", "title": "Kubernetes in 100 Seconds", "channel": "Fireship"}
        ]

        # 3. Analyze batch
        batch_res = self.client.post("/api/video/analyze_batch", data=json.dumps({
            "session_id": "session_acceptance_001",
            "intent": intent,
            "videos": raw_feed,
            "strictness": 0.65
        }), content_type="application/json")
        self.assertEqual(batch_res.status_code, 200)
        data = batch_res.get_json()

        results = data.get("results") or []
        self.assertEqual(len(results), 20, "Should analyze all 20 videos")

        # 4. Verify Schema Completeness for every video
        required_fields = [
            "video_id", "title", "content_type", "primary_topics", "secondary_topics",
            "goal_relevance", "positive_relevance", "negative_relevance",
            "alignment", "confidence", "decision", "label", "reason"
        ]

        by_id = {}
        for r in results:
            for field in required_fields:
                self.assertIn(field, r, f"Video {r.get('video_id')} missing {field}")
            by_id[r["video_id"]] = r

        # 5. Core Video Assertions
        # Highly aligned technical videos MUST be ALLOWed
        self.assertEqual(by_id["v01"]["decision"], "ALLOW", "K8s cluster architecture should be ALLOW")
        self.assertEqual(by_id["v02"]["decision"], "ALLOW", "DevOps CI/CD should be ALLOW")
        self.assertEqual(by_id["v03"]["decision"], "ALLOW", "K8s Security should be ALLOW")
        self.assertEqual(by_id["v04"]["decision"], "ALLOW", "Interview Q&A should be ALLOW")
        self.assertEqual(by_id["v05"]["decision"], "ALLOW", "Linux admin should be ALLOW")
        self.assertEqual(by_id["v06"]["decision"], "ALLOW", "Docker compose should be ALLOW")
        self.assertEqual(by_id["v07"]["decision"], "ALLOW", "K8s News mitigated should be ALLOW")
        self.assertEqual(by_id["v20"]["decision"], "ALLOW", "K8s in 100s should be ALLOW")

        # Hard negatives & irrelevant videos MUST be SUPPRESSed
        self.assertEqual(by_id["v13"]["decision"], "SUPPRESS", "Politics must be SUPPRESS")
        self.assertEqual(by_id["v14"]["decision"], "SUPPRESS", "Celebrity gossip must be SUPPRESS")
        self.assertEqual(by_id["v16"]["decision"], "SUPPRESS", "Kids cartoon must be SUPPRESS")
        self.assertEqual(by_id["v17"]["decision"], "SUPPRESS", "Pizza cooking must be SUPPRESS")
        self.assertEqual(by_id["v18"]["decision"], "SUPPRESS", "Football match must be SUPPRESS")

        # Incidental / Cartoon with tech keyword must NOT be ALLOW
        self.assertIn(by_id["v10"]["decision"], ["DEPRIORITIZE", "SUPPRESS"], "Tom & Jerry computer must not be ALLOW")
        self.assertNotEqual(by_id["v10"]["label"], "High Intent Match")
        self.assertEqual(by_id["v11"]["decision"], "SUPPRESS", "Oggy sports must be SUPPRESS for k8s goal")
        self.assertIn(by_id["v12"]["decision"], ["DEPRIORITIZE", "SUPPRESS"], "Docker boat cruise must not be ALLOW")

        # 6. Verify Dual Feed Divergence Metrics
        feed_metrics = data.get("feedAlignment") or data.get("feed_alignment") or {}
        self.assertIn("raw_jsd", feed_metrics)
        self.assertIn("controlled_jsd", feed_metrics)
        self.assertIn("raw_alignment", feed_metrics)
        self.assertIn("controlled_alignment", feed_metrics)

        # Controlled feed alignment MUST be strictly higher than raw alignment
        self.assertGreater(
            feed_metrics["controlled_alignment"],
            feed_metrics["raw_alignment"],
            "Controlled feed alignment must improve over raw feed"
        )

        # Controlled JSD MUST be strictly lower than raw JSD
        self.assertLess(
            feed_metrics["controlled_jsd"],
            feed_metrics["raw_jsd"],
            "Controlled JSD must reduce divergence from target intent"
        )

if __name__ == "__main__":
    unittest.main()
