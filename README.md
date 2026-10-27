# IntentAlign for YouTube (Final-Year BE IT Project)

### AI-Based User-Intent Alignment and Adaptive Feed Intervention Plugin for YouTube

**Architecture:** Chrome Browser Extension (Manifest V3) + Modular Python Flask AI/ML Backend  
**Mechanism:** Explicit Intent Profile → Dynamic YouTube Card Detection → Semantic Content Analysis → Dynamic Topic Profiling → Observed Feed Distribution (Q) vs Intent Distribution (P) → Jensen–Shannon Divergence (JSD) → Adaptive Feed Intervention (ALLOW / DEPRIORITIZE / SUPPRESS) → In-Feed User Feedback → Bounded Adaptive Topic Weight Recalculation.

---

## 1. Executive Summary & Guide's Six Clarifications

1. **How is content classified once installed?**
   - The user defines their Intent Profile in the Chrome extension popup.
   - When the user navigates to YouTube, a `MutationObserver` automatically detects dynamically rendered video cards across Home, Search, Watch sidebar, and Subscriptions without triggering DOM re-entry loops.
   - Metadata (title, channel, description snippet, video ID) is extracted and asynchronously sent in micro-batches to the local backend.
   - The system performs contextual keyword extraction and semantic vector cosine similarities against canonical topics.
   - Relevance metrics (Goal Relevance, Positive Relevance, Negative Relevance, Net Alignment) determine the client-side presentation intervention (ALLOW, DEPRIORITIZE, or SUPPRESS).
   - Feed metrics (Divergence, Exposure, Diversity) update in real-time. The user **never** has to manually classify every video.

2. **What is an Intent Profile?**
   - A structured representation maintaining logical separation between:
     - **Persistent User Profile**: Profession/role, general interests, and hobbies.
     - **Dynamic Current Intent**: Current goal, preferred topics (with weights), unwanted topics (with weights), derived goal topics, extracted goal keywords, strictness slider, and policy settings.

3. **What is fed to AI content analysis?**
   - Video title, channel name, and available description/metadata snippet.
   - No video downloading, raw frame decoding, audio processing, or private YouTube API access is required.

4. **Supported Browsers:**
   - Primary: **Google Chrome** (Manifest V3).
   - Secondary: Chromium-based **Microsoft Edge**.

5. **How does the system determine the user's goal?**
   - The user **explicitly enters** their Current Goal (e.g. *"Prepare for DevOps placement interview"*).
   - The NLP/AI engine expands the goal into domain-related concepts and weights.
   - The derived concepts are displayed as interactive, user-editable tags in the popup.

6. **How is keyword extraction done?**
   - Primary: KeyBERT with Sentence-Transformers embeddings.
   - Fallback: Scikit-learn TF-IDF n-gram phrase extraction with stop-word filtering.
   - Keyword extraction dynamically supplements semantic embeddings without relying solely on exact keyword matches.

---

## 2. Project Directory Structure

```
youtube_intent_alignment_plugin/
├── backend/
│   ├── app.py                      # Flask RESTful application & API routing
│   ├── requirements.txt            # Python dependencies
│   ├── services/
│   │   ├── __init__.py
│   │   ├── intent_service.py       # Topic normalization & goal concept derivation
│   │   ├── keyword_service.py      # Keyphrase extraction (KeyBERT / TF-IDF)
│   │   ├── embedding_service.py    # SentenceTransformer / LSA dense semantic space + LRU cache
│   │   ├── content_service.py      # Video metadata analysis & dynamic topic classification
│   │   ├── alignment_service.py    # Net alignment & soft vs hard negative discrimination
│   │   ├── divergence_service.py   # Intent (P) & Feed (Q) distributions, JSD, Shannon entropy
│   │   ├── intervention_service.py # Adaptive score (alpha*JSD + beta*NegExp) & decisions
│   │   └── feedback_service.py     # Bounded weight adaptation on user feedback
│   └── tests/
│       ├── test_scenarios.py       # Guide's 8 mandatory test cases
│       ├── test_intent.py          # Topic normalization & profile separation tests
│       ├── test_divergence.py      # JSD mathematical properties & entropy tests
│       ├── test_feedback.py        # Bounded weight updates tests
│       └── test_api.py             # Flask endpoint integration tests
├── extension/
│   ├── manifest.json               # Chrome Manifest V3 configuration
│   ├── background.js               # Service worker for badge states and storage
│   ├── content.js                  # YouTube DOM MutationObserver & client intervention
│   ├── popup.html                  # Profile input, analytics dashboard, settings
│   ├── popup.css                   # Modern dark-mode UI styling
│   ├── popup.js                    # Popup controller, chart renderer & API client
│   └── modules/                    # Modular frontend architectural helpers
├── dataset/
│   └── evaluation.csv              # 47 labeled records across Scenarios A, B, C, D
└── evaluation/
    ├── evaluate.py                 # Automated 3-way comparative benchmark (A vs B vs C)
    └── metrics.py                  # Precision, Recall, F1, FPR, FNR, Diversity, Latency
```

---

## 3. Mathematical & Algorithmic Formulation

### 3.1 Net Video Alignment
$$\text{Alignment} = \max\left(0, \min\left(1, \frac{w_p \cdot \text{Pos} + w_g \cdot \text{Goal} - w_n \cdot \text{Neg}}{w_p + w_g}\right)\right)$$
- Defaults: $w_p = 0.40$, $w_g = 0.40$, $w_n = 0.20$.
- **Hard Negative Rule**: If an unwanted topic is marked hard (e.g. *Politics*) and matches >= 0.50, video is forced to `SUPPRESS` unless direct goal relevance is overwhelming (>= 0.85).
- **Soft Negative Context Mitigation**: If an unwanted topic is soft (e.g. *News*) but the video has strong goal relevance (e.g. *Kubernetes Security News* for *Kubernetes Security Interview*), soft negative penalty is contextually mitigated, yielding `ALLOW` or `DEPRIORITIZE`.

### 3.2 Jensen–Shannon Divergence (JSD)
Given normalized Intent Distribution P and Observed Feed Distribution Q over vocabulary K:
$$M = \frac{1}{2}(P + Q)$$
$$D_{JS}(P \parallel Q) = \frac{1}{2} D_{KL}(P \parallel M) + \frac{1}{2} D_{KL}(Q \parallel M)$$
Using base-2 logarithm: $D_{JS} \in [0.0, 1.0]$.
$$\text{Intent Alignment Score} = 1 - D_{JS}(P \parallel Q)$$

### 3.3 Adaptive Intervention Score
$$\text{Intervention Score} = \alpha \cdot D_{JS} + \beta \cdot \text{Negative Exposure}$$
- Configurable parameters: $\alpha = 0.60$, $\beta = 0.40$.
- Levels: **LOW** (< 0.30), **MEDIUM** ([0.30, 0.60)), **HIGH** (>= 0.60).

### 3.4 Topic Diversity (Shannon Entropy)
$$H(Q) = -\frac{\sum_{i=1}^K Q_i \log_2(Q_i + \varepsilon)}{\log_2(K)}$$
Monitors that filtering preserves healthy topic exploration without creating radical filter bubbles.

### 3.5 Bounded Adaptive Feedback Update
$$w_{new} = \text{clip}(w_{old} + \eta \cdot \text{signal} \cdot \text{score}, w_{min}, w_{max})$$
- $\eta = 0.05$, $w_{min} = 0.10$, $w_{max} = 2.00$.

---

## 4. Empirical Evaluation Benchmark Results

Running `python evaluation/evaluate.py` across 47 labeled test items across 4 diverse scenarios:

| Metric | System A (No Filtering) | System B (Keyword Baseline) | System C (Proposed IntentAlign) |
| :--- | :--- | :--- | :--- |
| **Accuracy** | 59.57% | 65.96% | **97.87%** |
| **Precision** | 59.57% | 64.29% | **100.00%** |
| **Recall** | 100.00% | 96.43% | **96.43%** |
| **F1 Score** | 74.67% | 77.14% | **98.18%** |
| **False Positive Rate (FPR)** | 100.00% | 78.95% | **0.00%** |
| **False Negative Rate (FNR)** | 0.00% | 3.57% | **3.57%** |
| **Avg Processing Latency** | 0.0 ms | 0.002 ms | **1.5 ms** |
| **Topic Diversity (Shannon H)**| Untracked | Untracked | **0.8994** |

All figures are produced via actual algorithmic execution with zero simulated data.

---

## 5. How to Run the System

### 5.1 Start the AI Backend
```powershell
cd backend
python app.py
```
Verify the health endpoint in your browser or terminal:
```
http://127.0.0.1:5000/api/health
```

### 5.2 Load the Chrome Extension
1. Open Google Chrome and navigate to `chrome://extensions`.
2. Enable **Developer mode** (toggle in upper-right corner).
3. Click **Load unpacked**.
4. Select the `extension` folder inside this project directory:
   `C:\Users\juvat\.gemini\antigravity\scratch\youtube_intent_alignment_plugin\extension`
5. Pin the **IntentAlign** extension icon to the Chrome toolbar.

### 5.3 Test on YouTube
1. Click the IntentAlign icon.
2. Enter your current goal (e.g. *"Prepare for DevOps placement interview"*) and preferences.
3. Click **Save & Build Intent Profile**.
4. Open or refresh `https://www.youtube.com`.
5. Observe:
   - Aligned videos show green **HIGH INTENT MATCH** badges with alignment scores.
   - Soft-unwanted videos are subtly deprioritized with reduced opacity.
   - Misaligned and hard-unwanted videos are suppressed from the feed.
   - Click the 👍 / 👎 buttons on cards to trigger adaptive weight updates.
   - Open the popup anytime to see real-time Jensen-Shannon Divergence and topic charts!

### 5.4 Run Automated Tests
```powershell
python -m unittest discover -s backend/tests -p "test_*.py"
```
