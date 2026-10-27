"""
Comprehensive Semantic Adversarial Benchmark (Section 18)
Automates 32 paired tests for homonyms, polysemes, and incidental topic contamination.
Ensures technical videos are allowed while adversarial homonyms are suppressed or deprioritized.
"""
import unittest
from backend.services.embedding_service import EmbeddingService
from backend.services.keyword_service import KeywordService
from backend.services.intent_service import IntentService
from backend.services.content_service import ContentService
from backend.services.alignment_service import AlignmentService
from backend.services.divergence_service import DivergenceService
from backend.services.intervention_service import InterventionService

class TestSemanticAdversarial(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.emb = EmbeddingService()
        cls.kwd = KeywordService(cls.emb)
        cls.intent_svc = IntentService(cls.emb, cls.kwd)
        cls.content_svc = ContentService(cls.emb, cls.kwd)
        cls.align_svc = AlignmentService()
        cls.diverg_svc = DivergenceService()
        cls.interv_svc = InterventionService()

    def _eval_pair(self, goal, preferred, adv_title, adv_channel, tech_title, tech_channel):
        intent = self.intent_svc.build_intent_profile({
            "goal": goal,
            "preferredTopics": preferred,
            "unwantedTopics": ["Entertainment", "Cooking", "Sports", "Gaming"],
            "strictness": 0.65
        })
        feed_eval = self.diverg_svc.evaluate_feed(intent, [])
        feed_interv = self.interv_svc.compute_feed_intervention(feed_eval, strictness=0.65)

        # 1. Evaluate adversarial homonym video
        v_adv = {"video_id": "v_adv", "title": adv_title, "channel": adv_channel, "description": ""}
        c_adv = self.content_svc.analyze_content(v_adv, intent)
        a_adv = self.align_svc.compute_alignment(c_adv, intent, strictness=0.65)
        d_adv = self.interv_svc.decide_video(c_adv, a_adv, feed_interv, strictness=0.65)

        # 2. Evaluate authentic technical video
        v_tech = {"video_id": "v_tech", "title": tech_title, "channel": tech_channel, "description": ""}
        c_tech = self.content_svc.analyze_content(v_tech, intent)
        a_tech = self.align_svc.compute_alignment(c_tech, intent, strictness=0.65)
        d_tech = self.interv_svc.decide_video(c_tech, a_tech, feed_interv, strictness=0.65)

        return (d_adv, a_adv, c_adv), (d_tech, a_tech, c_tech)

    def test_01_python_snake_vs_coding(self):
        adv, tech = self._eval_pair(
            "Learn Python programming and data structures", ["Python", "Programming"],
            "Python Snake Wildlife Safari in Rainforest", "NatureGeo",
            "Complete Python Programming Course for Beginners", "TechCourse"
        )
        self.assertIn(adv[0]["decision"], ["DEPRIORITIZE", "SUPPRESS"])
        self.assertNotEqual(adv[0]["label"], "High Intent Match")
        self.assertEqual(tech[0]["decision"], "ALLOW")

    def test_02_java_coffee_vs_programming(self):
        adv, tech = self._eval_pair(
            "Learn Java backend development and Spring Boot", ["Java", "Programming"],
            "Java Coffee Brewing and Espresso Recipe Guide", "BaristaDaily",
            "Java Spring Boot Microservices Tutorial and Backend Architecture", "JavaMaster"
        )
        self.assertIn(adv[0]["decision"], ["DEPRIORITIZE", "SUPPRESS"])
        self.assertEqual(tech[0]["decision"], "ALLOW")

    def test_03_apple_pie_vs_ios(self):
        adv, tech = self._eval_pair(
            "Build iOS apps with Swift for Apple devices", ["Apple", "iOS", "Mobile Development"],
            "Apple Pie Baking Recipe and Cinnamon Dessert", "BakingChef",
            "Apple iOS Swift App Development Complete Guide", "iOSDev"
        )
        self.assertIn(adv[0]["decision"], ["DEPRIORITIZE", "SUPPRESS"])
        self.assertEqual(tech[0]["decision"], "ALLOW")

    def test_04_amazon_rainforest_vs_aws(self):
        adv, tech = self._eval_pair(
            "Learn Amazon AWS Cloud Infrastructure and DevOps", ["AWS", "Cloud"],
            "Amazon Rainforest Deforestation and River Tour", "EarthDoc",
            "AWS Cloud Architecture and Solutions Architect Guide", "CloudGuru"
        )
        self.assertIn(adv[0]["decision"], ["DEPRIORITIZE", "SUPPRESS"])
        self.assertEqual(tech[0]["decision"], "ALLOW")

    def test_05_cloud_weather_vs_cloud_computing(self):
        adv, tech = self._eval_pair(
            "Learn Cloud Computing and DevOps", ["Cloud", "DevOps"],
            "Storm Clouds Weather Forecast and Thunderstorm Atmosphere", "WeatherChannel",
            "Cloud Computing Kubernetes and Docker Deployment Architecture", "CloudPro"
        )
        self.assertIn(adv[0]["decision"], ["DEPRIORITIZE", "SUPPRESS"])
        self.assertEqual(tech[0]["decision"], "ALLOW")

    def test_06_react_reaction_vs_frontend(self):
        adv, tech = self._eval_pair(
            "Master React JS web development and frontend", ["React", "Programming"],
            "Reacting to Scary Ghost Videos Try Not To Laugh", "ReactionLounge",
            "React JS Frontend Hooks and State Management Tutorial", "ReactMaster"
        )
        self.assertIn(adv[0]["decision"], ["DEPRIORITIZE", "SUPPRESS"])
        self.assertEqual(tech[0]["decision"], "ALLOW")

    def test_07_docker_harbor_vs_containers(self):
        adv, tech = self._eval_pair(
            "Learn Docker and Container DevOps", ["Docker", "DevOps"],
            "Docker Boat Harbor Cruise and Ship Docking at Marina Seaport", "TravelPort",
            "Docker Containerization and Dockerfile Best Practices Tutorial", "DevOpsDaily"
        )
        self.assertIn(adv[0]["decision"], ["DEPRIORITIZE", "SUPPRESS"])
        self.assertEqual(tech[0]["decision"], "ALLOW")

    def test_08_rust_corrosion_vs_language(self):
        adv, tech = self._eval_pair(
            "Learn Rust systems programming language", ["Rust", "Programming"],
            "Rust Metal Corrosion Removal and Sandblasting Old Car", "AutoRestoration",
            "Rust Programming Language Memory Safety and Concurrency Tutorial", "RustDev"
        )
        self.assertIn(adv[0]["decision"], ["DEPRIORITIZE", "SUPPRESS"])
        self.assertEqual(tech[0]["decision"], "ALLOW")

    def test_09_go_boardgame_vs_golang(self):
        adv, tech = self._eval_pair(
            "Learn Go programming language and backend development", ["Go", "Programming"],
            "Game of Go Baduk Masterclass and Board Moves", "BoardGamesPro",
            "Golang Go Programming Concurrency and Goroutines Masterclass", "GoDev"
        )
        self.assertIn(adv[0]["decision"], ["DEPRIORITIZE", "SUPPRESS"])
        self.assertEqual(tech[0]["decision"], "ALLOW")

    def test_10_ai_parody_vs_deep_learning(self):
        adv, tech = self._eval_pair(
            "Study Artificial Intelligence and Deep Learning", ["AI", "Machine Learning"],
            "AI Voice Parody Meme and Funny Movie Trailer", "ComedyHub",
            "Deep Learning Neural Network Architecture and LLM Training Guide", "AIDeepDive"
        )
        self.assertIn(adv[0]["decision"], ["DEPRIORITIZE", "SUPPRESS"])
        self.assertEqual(tech[0]["decision"], "ALLOW")

    def test_11_ml_toy_vs_ml_engineering(self):
        adv, tech = self._eval_pair(
            "Master Machine Learning algorithms and data models", ["Machine Learning", "Data Science"],
            "Machine Learning Toy Building Blocks for Kids Play", "ToyReview",
            "Machine Learning Scikit-Learn Regression and Classification Full Course", "MLEngineer"
        )
        self.assertIn(adv[0]["decision"], ["DEPRIORITIZE", "SUPPRESS"])
        self.assertEqual(tech[0]["decision"], "ALLOW")

    def test_12_pandas_animal_vs_data_science(self):
        adv, tech = self._eval_pair(
            "Data Science with Python and Pandas", ["Pandas", "Data Science"],
            "Giant Panda Cubs Eating Bamboo at Zoo Sanctuary", "ZooAnimalTV",
            "Pandas DataFrame Data Analysis and CSV Cleaning Tutorial", "DataScienceLab"
        )
        self.assertIn(adv[0]["decision"], ["DEPRIORITIZE", "SUPPRESS"])
        self.assertEqual(tech[0]["decision"], "ALLOW")

    def test_13_spark_welding_vs_big_data(self):
        adv, tech = self._eval_pair(
            "Learn Apache Spark Big Data Engineering", ["Big Data", "Cloud"],
            "Welding Sparks and Electrical Fire Safety on Construction Site", "WorkshopSafety",
            "Apache Spark Big Data PySpark Distributed Computing Course", "BigDataPro"
        )
        self.assertIn(adv[0]["decision"], ["DEPRIORITIZE", "SUPPRESS"])
        self.assertEqual(tech[0]["decision"], "ALLOW")

    def test_14_chef_cooking_vs_devops(self):
        adv, tech = self._eval_pair(
            "Learn Chef DevOps infrastructure automation", ["DevOps", "Infrastructure"],
            "Master Chef Gordon Ramsay Steak Recipe and Dinner", "KitchenSecrets",
            "Chef DevOps Automation and Infrastructure Cookbooks Tutorial", "DevOpsHub"
        )
        self.assertIn(adv[0]["decision"], ["DEPRIORITIZE", "SUPPRESS"])
        self.assertEqual(tech[0]["decision"], "ALLOW")

    def test_15_puppet_show_vs_devops(self):
        adv, tech = self._eval_pair(
            "Learn Puppet configuration management and DevOps", ["DevOps", "Infrastructure"],
            "Puppet Show Marionette Theatre for Children", "KidsPuppets",
            "Puppet Infrastructure as Code DevOps Automation Tutorial", "DevOpsLab"
        )
        self.assertIn(adv[0]["decision"], ["DEPRIORITIZE", "SUPPRESS"])
        self.assertEqual(tech[0]["decision"], "ALLOW")

    def test_16_swift_popstar_vs_ios(self):
        adv, tech = self._eval_pair(
            "Learn Swift programming for iOS app development", ["Swift", "Mobile Development"],
            "Taylor Swift Eras Tour Concert Live Music and Song", "PopMusicDaily",
            "Swift UI Apple iOS Mobile App Development Tutorial", "SwiftDeveloper"
        )
        self.assertIn(adv[0]["decision"], ["DEPRIORITIZE", "SUPPRESS"])
        self.assertEqual(tech[0]["decision"], "ALLOW")

    def test_17_ruby_gem_vs_rails(self):
        adv, tech = self._eval_pair(
            "Learn Ruby on Rails backend web development", ["Ruby", "Web Development"],
            "Ruby Gemstone Mining and Jewelry Diamond Valuation", "GemExpert",
            "Ruby on Rails Web Development Full Tutorial for Beginners", "RailsCasts"
        )
        self.assertIn(adv[0]["decision"], ["DEPRIORITIZE", "SUPPRESS"])
        self.assertEqual(tech[0]["decision"], "ALLOW")

    def test_18_spring_season_vs_java(self):
        adv, tech = self._eval_pair(
            "Learn Spring Boot Java framework", ["Java", "Programming"],
            "Spring Flowers Blooming Cherry Blossom Nature Tour", "NatureWalks",
            "Spring Boot Java REST API Framework Tutorial", "JavaBrains"
        )
        self.assertIn(adv[0]["decision"], ["DEPRIORITIZE", "SUPPRESS"])
        self.assertEqual(tech[0]["decision"], "ALLOW")

    def test_19_angular_physics_vs_frontend(self):
        adv, tech = self._eval_pair(
            "Learn Angular web development and TypeScript", ["Web Development", "Programming"],
            "Angular Momentum Physics Mechanics and Rotation Velocity", "PhysicsLecture",
            "Angular TypeScript Google Frontend Framework Tutorial", "AngularPro"
        )
        self.assertIn(adv[0]["decision"], ["DEPRIORITIZE", "SUPPRESS"])
        self.assertEqual(tech[0]["decision"], "ALLOW")

    def test_20_bug_insects_vs_debugging(self):
        adv, tech = self._eval_pair(
            "Learn Software Debugging and Testing", ["Programming", "Software Engineering"],
            "Insects and Spider Bugs Entomology Microscope Crawling", "BioInsects",
            "Debugging Software Bugs and Fixing Memory Leaks in Code", "SoftwareTesting"
        )
        self.assertIn(adv[0]["decision"], ["DEPRIORITIZE", "SUPPRESS"])
        self.assertEqual(tech[0]["decision"], "ALLOW")

    def test_21_mouse_rodent_vs_hardware(self):
        adv, tech = self._eval_pair(
            "Computer Hardware and Gadgets review", ["Technology", "Gadgets"],
            "Pet Mouse Traps and Rodent Pest Removal in Garden", "PestControl",
            "Logitech Optical Mouse DPI Ergonomic Review for Productivity", "HardwareTech"
        )
        self.assertIn(adv[0]["decision"], ["DEPRIORITIZE", "SUPPRESS"])
        self.assertEqual(tech[0]["decision"], "ALLOW")

    def test_22_cookies_baking_vs_cybersecurity(self):
        adv, tech = self._eval_pair(
            "Learn Cybersecurity and Web Security", ["Cybersecurity", "Technology"],
            "Chocolate Chip Cookies Baking Recipe in Oven", "BakingTreats",
            "HTTP Cookies Web Security and Session Tokens Vulnerability", "SecurityNow"
        )
        self.assertIn(adv[0]["decision"], ["DEPRIORITIZE", "SUPPRESS"])
        self.assertEqual(tech[0]["decision"], "ALLOW")

    def test_23_cricket_insect_vs_sports(self):
        adv, tech = self._eval_pair(
            "Cricket Sports Analysis and match tactics", ["Sports"],
            "Cricket Insect Chirping Night Sounds Biology", "NatureSounds",
            "Cricket Match Highlights IPL T20 Final Bowling Analysis", "CricketHub"
        )
        self.assertIn(adv[0]["decision"], ["DEPRIORITIZE", "SUPPRESS"])
        self.assertEqual(tech[0]["decision"], "ALLOW")

    def test_24_matrix_movie_vs_algebra(self):
        adv, tech = self._eval_pair(
            "Study Linear Algebra and Mathematics for Machine Learning", ["Education", "Data Science"],
            "The Matrix Movie Sci Fi Action Scene Trailer", "MovieTrailers",
            "Matrix Linear Algebra Eigenvalues and Determinants Lecture", "MathAcademy"
        )
        self.assertIn(adv[0]["decision"], ["DEPRIORITIZE", "SUPPRESS"])
        self.assertEqual(tech[0]["decision"], "ALLOW")

    def test_25_subway_sandwich_vs_transit(self):
        adv, tech = self._eval_pair(
            "Urban Transportation and Travel systems", ["Travel"],
            "Subway Footlong Tuna Sandwich Fast Food Review", "FoodCritique",
            "Tokyo Subway Metro Underground Transit System Tour", "TravelJapan"
        )
        self.assertIn(adv[0]["decision"], ["DEPRIORITIZE", "SUPPRESS"])
        self.assertEqual(tech[0]["decision"], "ALLOW")

    def test_26_surf_waves_vs_web_browsing(self):
        adv, tech = self._eval_pair(
            "Learn Web Security and Browser Privacy", ["Cybersecurity", "Technology"],
            "Big Wave Surfing Hawaii Ocean Beach Sport", "ExtremeSurf",
            "Web Surfing Browser Security and Chrome Extension Safety", "CyberShield"
        )
        self.assertIn(adv[0]["decision"], ["DEPRIORITIZE", "SUPPRESS"])
        self.assertEqual(tech[0]["decision"], "ALLOW")

    def test_27_virus_biology_vs_malware(self):
        adv, tech = self._eval_pair(
            "Cybersecurity and Malware Analysis", ["Cybersecurity", "Technology"],
            "Microbiology Virus Infection and Vaccine Biology", "MedicalBio",
            "Computer Malware Virus Trojan Exploit Analysis", "MalwareAnalysis"
        )
        self.assertIn(adv[0]["decision"], ["DEPRIORITIZE", "SUPPRESS"])
        self.assertEqual(tech[0]["decision"], "ALLOW")

    def test_28_phishing_fishing_vs_cybersecurity(self):
        adv, tech = self._eval_pair(
            "Learn Cybersecurity Defense and Anti-Phishing", ["Cybersecurity", "Technology"],
            "Lake Trout Fishing Rod and Bass Bait Catch", "FishingOutdoors",
            "Phishing Scam Email Detection and Social Engineering Defense", "CyberSecurityPro"
        )
        self.assertIn(adv[0]["decision"], ["DEPRIORITIZE", "SUPPRESS"])
        self.assertEqual(tech[0]["decision"], "ALLOW")

    def test_29_worm_biology_vs_network_worm(self):
        adv, tech = self._eval_pair(
            "Cybersecurity and Network Defense", ["Cybersecurity", "DevOps"],
            "Earthworms Soil Garden Compost Creepy Crawly", "GardeningNature",
            "Network Worm Cybersecurity Malware Exploit and Propagation", "NetworkSecurity"
        )
        self.assertIn(adv[0]["decision"], ["DEPRIORITIZE", "SUPPRESS"])
        self.assertEqual(tech[0]["decision"], "ALLOW")

    def test_30_trojan_greek_myth_vs_malware(self):
        adv, tech = self._eval_pair(
            "Cybersecurity Threat Hunting and Malware", ["Cybersecurity", "Technology"],
            "Ancient Greek Trojan Horse Myth and Helen of Troy History", "HistoryDocumentary",
            "Trojan Horse Backdoor Malware Security Breach Investigation", "ThreatHunter"
        )
        self.assertIn(adv[0]["decision"], ["DEPRIORITIZE", "SUPPRESS"])
        self.assertEqual(tech[0]["decision"], "ALLOW")

    def test_31_firewall_masonry_vs_network_security(self):
        adv, tech = self._eval_pair(
            "Learn Network Security and Firewalls", ["Cybersecurity", "DevOps"],
            "Drywall Masonry Firewall Building Construction Safety", "BuildingConstruction",
            "Network Security Firewall iptables Router Traffic Rules Guide", "NetAdmin"
        )
        self.assertIn(adv[0]["decision"], ["DEPRIORITIZE", "SUPPRESS"])
        self.assertEqual(tech[0]["decision"], "ALLOW")

    def test_32_git_slang_vs_version_control(self):
        adv, tech = self._eval_pair(
            "Learn Git and GitHub DevOps version control", ["DevOps", "Programming"],
            "British Slang Silly Old Git Comedy Sketch", "BritishComedy",
            "Git Version Control Branching and Merge Conflicts Tutorial", "GitPro"
        )
        self.assertIn(adv[0]["decision"], ["DEPRIORITIZE", "SUPPRESS"])
        self.assertEqual(tech[0]["decision"], "ALLOW")

if __name__ == "__main__":
    unittest.main()
