import sys
import os
import time
import json
import hashlib
from typing import Dict, Any

sys.path.insert(
    0,
    os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)

from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS

from backend.services.embedding_service import EmbeddingService
from backend.services.keyword_service import KeywordService
from backend.services.intent_service import IntentService
from backend.services.content_service import ContentService
from backend.services.alignment_service import AlignmentService
from backend.services.divergence_service import DivergenceService
from backend.services.intervention_service import InterventionService
from backend.services.feedback_service import FeedbackService


# ============================================================
# APP INITIALIZATION
# ============================================================

app = Flask(__name__)

# ============================================================
# CORS CONFIGURATION
# ============================================================

ALLOWED_ORIGINS = [
    "https://www.youtube.com",
    "https://youtube.com",
    "http://127.0.0.1:5000",
    "http://localhost:5000"
]

CORS(
    app,
    resources={
        r"/api/*": {
            "origins": ALLOWED_ORIGINS,
            "methods": ["GET", "POST", "OPTIONS"],
            "allow_headers": ["Content-Type"],
            "supports_credentials": False
        },
        r"/intent": {
            "origins": ALLOWED_ORIGINS,
            "methods": ["POST", "OPTIONS"],
            "allow_headers": ["Content-Type"],
            "supports_credentials": False
        },
        r"/analyze": {
            "origins": ALLOWED_ORIGINS,
            "methods": ["POST", "OPTIONS"],
            "allow_headers": ["Content-Type"],
            "supports_credentials": False
        },
        r"/feed-alignment": {
            "origins": ALLOWED_ORIGINS,
            "methods": ["GET", "POST", "OPTIONS"],
            "allow_headers": ["Content-Type"],
            "supports_credentials": False
        }
    }
)

print("[IntentAlign V1.3] Initializing backend services...")

embedding_service = EmbeddingService()
keyword_service = KeywordService(
    embedding_service=embedding_service
)
intent_service = IntentService(
    embedding_service,
    keyword_service
)
content_service = ContentService(
    embedding_service,
    keyword_service
)

alignment_service = AlignmentService(
    weight_goal=0.45,
    weight_derived=0.20,
    weight_preferred=0.20,
    weight_interests=0.10,
    weight_hobbies=0.05,
    weight_negative=0.25
)

divergence_service = DivergenceService()

intervention_service = InterventionService(
    alpha=0.60,
    beta=0.40
)

feedback_service = FeedbackService(
    default_learning_rate=0.05
)


# ============================================================
# SESSION STORAGE
# ============================================================

SESSIONS: Dict[str, Dict[str, Any]] = {}


MAX_SESSION_HISTORY = 100
MAX_SESSIONS = 50


# ============================================================
# INTENT FINGERPRINT
# ============================================================

def _normalise_list(value):
    """
    Convert a value into a deterministic list of strings.
    """
    if value is None:
        return []

    if isinstance(value, str):
        return [
            x.strip().lower()
            for x in value.split(",")
            if x.strip()
        ]

    if isinstance(value, list):
        return [
            str(x).strip().lower()
            for x in value
            if str(x).strip()
        ]

    return [str(value).strip().lower()]


def _intent_fingerprint(intent: Dict[str, Any]) -> str:

    if not isinstance(intent, dict):
        intent = {}

    current_intent = intent.get("current_intent") or {}
    settings = intent.get("settings") or {}

    payload = {
        "goal": (
            intent.get("goal")
            or intent.get("current_goal")
            or current_intent.get("current_goal")
            or current_intent.get("goal")
            or ""
        ),

        "preferred_topics": _normalise_list(
            intent.get("preferredTopics")
            or intent.get("preferred_topics")
            or current_intent.get("preferred_topics")
            or []
        ),

        "unwanted_topics": _normalise_list(
            intent.get("unwantedTopics")
            or intent.get("unwanted_topics")
            or current_intent.get("unwanted_topics")
            or []
        ),

        "interests": _normalise_list(
            intent.get("profile", {}).get("interests")
            if isinstance(intent.get("profile"), dict)
            else intent.get("interests", [])
        ),

        "hobbies": _normalise_list(
            intent.get("profile", {}).get("hobbies")
            if isinstance(intent.get("profile"), dict)
            else intent.get("hobbies", [])
        ),

        "hard_negatives": _normalise_list(
            intent.get("hard_negatives")
            or intent.get("hardNegatives")
            or settings.get("hard_negatives")
            or settings.get("hardNegatives")
            or []
        ),

        "soft_negatives": _normalise_list(
            intent.get("soft_negatives")
            or intent.get("softNegatives")
            or settings.get("soft_negatives")
            or settings.get("softNegatives")
            or []
        ),

        "strictness": float(
            intent.get("strictness")
            if intent.get("strictness") is not None
            else settings.get("strictness", 0.65)
        )
    }

    raw = json.dumps(
        payload,
        sort_keys=True,
        ensure_ascii=False
    )

    return hashlib.sha256(
        raw.encode("utf-8")
    ).hexdigest()


# ============================================================
# SESSION ID GENERATION
# ============================================================

def _generate_session_id():
    """
    Generate a unique server-side fallback session ID.
    """
    return (
        "session_"
        + str(int(time.time() * 1000))
        + "_"
        + hashlib.sha1(
            str(time.time_ns()).encode()
        ).hexdigest()[:8]
    )


# ============================================================
# SESSION CREATION
# ============================================================

def _create_empty_session(
    sid: str,
    intent_profile: Dict[str, Any] = None
):
    now = time.time()

    intent_profile = intent_profile or {}

    SESSIONS[sid] = {
        "session_id": sid,
        "intent": intent_profile,
        "intent_fingerprint": _intent_fingerprint(
            intent_profile
        ),
        "feed_history": [],
        "feedback_history": [],
        "alignment_history": [],
        "created_at": now,
        "last_active": now
    }

    return SESSIONS[sid]


def _evict_old_sessions():
    """
    Prevent unlimited in-memory session growth.
    """
    if len(SESSIONS) < MAX_SESSIONS:
        return

    oldest_sid = min(
        SESSIONS.keys(),
        key=lambda k: SESSIONS[k].get(
            "last_active",
            0
        )
    )

    del SESSIONS[oldest_sid]


def get_or_create_session(
    session_id: str = None,
    intent_profile: Dict[str, Any] = None
):

    sid = str(
        session_id or _generate_session_id()
    ).strip()

    if not sid:
        sid = _generate_session_id()

    # --------------------------------------------------------
    # NEW SESSION
    # --------------------------------------------------------

    if sid not in SESSIONS:
        _evict_old_sessions()

        session = _create_empty_session(
            sid,
            intent_profile
        )

        print(
            f"[SESSION] Created new session: {sid}"
        )

        return session

    # --------------------------------------------------------
    # EXISTING SESSION
    # --------------------------------------------------------

    session = SESSIONS[sid]

    session["last_active"] = time.time()

    return session


# ============================================================
# START / REPLACE CURRENT INTENT
# ============================================================

def start_intent_session(
    session_id: str,
    intent_profile: Dict[str, Any]
):

    sid = str(
        session_id or _generate_session_id()
    ).strip()

    new_fingerprint = _intent_fingerprint(
        intent_profile
    )

    # --------------------------------------------------------
    # Completely new session
    # --------------------------------------------------------

    if sid not in SESSIONS:
        _evict_old_sessions()

        session = _create_empty_session(
            sid,
            intent_profile
        )

        print(
            f"[SESSION] New intent session created: {sid}"
        )

        return session, True

    # --------------------------------------------------------
    # Existing session
    # --------------------------------------------------------

    session = SESSIONS[sid]

    old_fingerprint = session.get(
        "intent_fingerprint"
    )

    intent_changed = (
        old_fingerprint
        and old_fingerprint != new_fingerprint
    )

    if intent_changed:
        print(
            "[SESSION] Intent changed inside same session."
        )
        print(
            "[SESSION] Clearing old feed history."
        )

        # IMPORTANT:
        # Old Kubernetes feed must never be used for
        # the new Python intent.
        session["feed_history"] = []

        session["created_at"] = time.time()

    session["intent"] = intent_profile
    session["intent_fingerprint"] = new_fingerprint
    session["last_active"] = time.time()

    return session, intent_changed


# ============================================================
# STATUS PAGE
# ============================================================

INDEX_HTML = r"""
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport"
      content="width=device-width, initial-scale=1">
<title>IntentAlign AI Backend</title>

<style>
body {
    margin: 0;
    padding: 40px;
    background: #0f172a;
    color: #f8fafc;
    font-family: Arial, sans-serif;
}

.container {
    max-width: 900px;
    margin: auto;
}

.header {
    text-align: center;
    margin-bottom: 35px;
}

.status {
    display: inline-block;
    padding: 8px 15px;
    border-radius: 20px;
    background: rgba(16,185,129,.15);
    color: #10b981;
    font-weight: bold;
}

h1 {
    margin-top: 20px;
}

.grid {
    display: grid;
    grid-template-columns:
        repeat(auto-fit, minmax(220px, 1fr));
    gap: 15px;
}

.card {
    background: #1e293b;
    border: 1px solid #334155;
    border-radius: 12px;
    padding: 20px;
}

.label {
    color: #94a3b8;
    font-size: 12px;
    text-transform: uppercase;
}

.value {
    margin-top: 8px;
    font-size: 20px;
    font-weight: bold;
}

.section {
    margin-top: 25px;
    background: #1e293b;
    border: 1px solid #334155;
    border-radius: 12px;
    padding: 25px;
}

code {
    color: #a5b4fc;
}
</style>
</head>

<body>

<div class="container">

<div class="header">
    <div class="status">
        ● Backend Online & Operational (V1.3)
    </div>

    <h1>IntentAlign AI Engine</h1>

    <p>
        AI-Based User-Intent Alignment and
        Adaptive Feed Intervention Plugin for YouTube
    </p>
</div>

<div class="grid">

<div class="card">
    <div class="label">Embedding Space</div>
    <div class="value">{{ model_name }}</div>
    <p>
        {{ 'Neural SentenceTransformer'
           if is_neural
           else 'Dense LSA Latent Space' }}
    </p>
</div>

<div class="card">
    <div class="label">Active Sessions</div>
    <div class="value">{{ active_sessions }}</div>
    <p>Isolated intent sessions</p>
</div>

<div class="card">
    <div class="label">Embedding Cache</div>
    <div class="value">{{ cache_size }}</div>
    <p>
        Hits: {{ cache_hits }}
        · Misses: {{ cache_misses }}
    </p>
</div>

</div>

<div class="section">

<h2>Chrome Extension</h2>

<p>
Load:
<code>
D:\youtube_intent_alignment_plugin\extension
</code>
</p>

<p>
Backend:
<code>
http://127.0.0.1:5000
</code>
</p>

<p>
Health:
<a href="/api/health">
/api/health
</a>
</p>

</div>

<div class="section">

<h2>API Endpoints</h2>

<ul>
<li>GET /api/health</li>
<li>POST /api/intent/analyze</li>
<li>POST /api/video/analyze</li>
<li>POST /api/video/analyze_batch</li>
<li>POST /api/feed/profile</li>
<li>POST /api/session/reset</li>
<li>POST /api/feedback</li>
</ul>

</div>

</div>

</body>
</html>
"""


# ============================================================
# INDEX
# ============================================================

@app.route("/", methods=["GET"])
def index():

    emb_status = embedding_service.status()

    return render_template_string(
        INDEX_HTML,
        model_name=emb_status["model"],
        is_neural=emb_status["is_neural"],
        active_sessions=len(SESSIONS),
        cache_size=emb_status["cache_size"],
        cache_hits=emb_status["cache_hits"],
        cache_misses=emb_status["cache_misses"]
    )


# ============================================================
# HEALTH
# ============================================================

@app.route("/api/health", methods=["GET"])
@app.route("/health", methods=["GET"])
def health():

    emb_status = embedding_service.status()

    return jsonify({
        "status": "ok",
        "service":
            "AI-Based User-Intent Alignment "
            "& Adaptive Feed Intervention Backend",

        "version": "1.3.0",

        "embedding_model":
            emb_status["model"],

        "is_neural":
            emb_status["is_neural"],

        "embedding_cache_size":
            emb_status["cache_size"],

        "cache_hits":
            emb_status["cache_hits"],

        "cache_misses":
            emb_status["cache_misses"],

        "keybert_available":
            keyword_service.is_keybert,

        "active_sessions_count":
            len(SESSIONS)
    })


# ============================================================
# INTENT ANALYSIS
# ============================================================

@app.route("/api/intent/analyze", methods=["POST"])
@app.route("/intent", methods=["POST"])
def analyze_intent():

    try:
        data = request.get_json(force=True) or {}
    except Exception:
        return jsonify({
            "error": "Invalid JSON in request body"
        }), 400

    try:
        profile = intent_service.build_intent_profile(
            data
        )
    except Exception as exc:
        return jsonify({
            "error": "Intent profile creation failed",
            "details": str(exc)
        }), 500

    sid = (
        profile.get("session_id")
        or data.get("session_id")
        or _generate_session_id()
    )

    profile["session_id"] = sid


    session, intent_changed = start_intent_session(
        sid,
        profile
    )

    return jsonify({
        **profile,
        "session_id": sid,
        "intent_changed": intent_changed,
        "feed_history_reset": intent_changed,
        "active_session": {
            "session_id": sid,
            "history_size":
                len(session["feed_history"])
        }
    })


# ============================================================
# VIDEO PROCESSING
# ============================================================

def _process_video_card(
    video_item: Dict[str, Any],
    intent: Dict[str, Any],
    strictness: float,
    feed_interv: Dict[str, Any]
):

    t0 = time.time()

    content_res = content_service.analyze_content(
        video_item,
        intent
    )

    align_res = alignment_service.compute_alignment(
        content_res,
        intent,
        strictness=strictness
    )

    decision_res = intervention_service.decide_video(
        content_res,
        align_res,
        feed_interv,
        strictness=strictness
    )

    latency_ms = round(
        (time.time() - t0) * 1000,
        2
    )

    decision = decision_res["decision"]

    controlled_visible = (
        decision in [
            "ALLOW",
            "DEPRIORITIZE"
        ]
    )

    topics_list = [
        {
            "topic": k,
            "score": v
        }
        for k, v in list(
            content_res.get(
                "topics",
                {}
            ).items()
        )[:6]
    ]

    return {

        "video_id":
            content_res["video_id"],

        "title":
            content_res["title"],

        "channel":
            content_res["channel"],

        "description":
            content_res["description"],

        "keywords":
            content_res["keywords"],

        "topics":
            topics_list,

        "topic_dict":
            content_res.get(
                "topics",
                {}
            ),

        "content_type":
            content_res.get(
                "content_type",
                "Other"
            ),

        "primary_topics":
            content_res.get(
                "primary_topics",
                []
            ),

        "secondary_topics":
            content_res.get(
                "secondary_topics",
                []
            ),

        "entities":
            content_res.get(
                "entities",
                []
            ),

        "is_incidental":
            content_res.get(
                "is_incidental",
                False
            ),

        "positive_relevance":
            content_res["positive_relevance"],

        "negative_relevance":
            content_res["negative_relevance"],

        "goal_relevance":
            content_res["goal_relevance"],

        "matched_negative":
            content_res["matched_negative"],

        "alignment":
            align_res["alignment"],

        "alignment_components":
            align_res.get(
                "components",
                {}
            ),

        "content_type_compatibility":
            align_res.get(
                "content_type_compatibility",
                1.0
            ),

        "confidence":
            decision_res["confidence"],

        "decision":
            decision,

        "label":
            decision_res["label"],

        "reason":
            decision_res["reason"],

        "is_uncertain":
            decision_res["is_uncertain"],

        "debug_analysis":
            content_res.get(
                "debug_analysis",
                {}
            ),

        # Raw exposure is recorded BEFORE intervention.
        "raw_exposed":
            True,

        # Controlled feed visibility is after intervention.
        "controlled_visible":
            controlled_visible,

        "latency_ms":
            latency_ms,

        "timestamp":
            time.strftime(
                "%Y-%m-%dT%H:%M:%SZ",
                time.gmtime()
            )
    }


# ============================================================
# SINGLE VIDEO
# ============================================================

@app.route(
    "/api/video/analyze",
    methods=["POST"]
)
def analyze_single_video():

    try:
        data = request.get_json(force=True) or {}
    except Exception:
        return jsonify({
            "error": "Invalid JSON"
        }), 400

    sid = (
        data.get("session_id")
        or (data.get("intent") or {}).get(
            "session_id"
        )
    )

    intent_from_request = (
        data.get("intent") or {}
    )

    # --------------------------------------------------------
    # If no session ID exists, create one.
    # --------------------------------------------------------

    if not sid:
        sid = _generate_session_id()

    session = get_or_create_session(
        sid,
        None
    )

    intent = session.get("intent") or {}

    # If this is a brand-new session and no stored intent
    # exists, accept the request intent.
    if not intent and intent_from_request:

        session["intent"] = intent_from_request

        session["intent_fingerprint"] = (
            _intent_fingerprint(
                intent_from_request
            )
        )

        intent = intent_from_request

    if not intent:
        return jsonify({
            "error":
                "No active intent profile for session",
            "session_id": sid
        }), 400

    strictness = float(
        data.get(
            "strictness",
            intent.get(
                "settings",
                {}
            ).get(
                "strictness",
                0.65
            )
        )
    )

    video = data.get("video") or {}

    # Current session feed ONLY.
    feed_eval = (
        divergence_service.evaluate_dual_feed(
            intent,
            session["feed_history"]
        )
    )

    feed_interv = (
        intervention_service
        .compute_feed_intervention(
            feed_eval,
            strictness=strictness
        )
    )

    result = _process_video_card(
        video,
        intent,
        strictness,
        feed_interv
    )

    session["feed_history"].append(
        result
    )

    if len(
        session["feed_history"]
    ) > MAX_SESSION_HISTORY:

        del session["feed_history"][0]

    session["last_active"] = time.time()

    return jsonify({
        **result,
        "session_id": sid
    })


# ============================================================
# BATCH VIDEO ANALYSIS
# ============================================================

@app.route(
    "/api/video/analyze_batch",
    methods=["POST"]
)
@app.route(
    "/analyze",
    methods=["POST"]
)
def analyze_batch():

    try:
        data = request.get_json(force=True) or {}
    except Exception:
        return jsonify({
            "error": "Invalid JSON"
        }), 400

    sid = (
        data.get("session_id")
        or (data.get("intent") or {}).get(
            "session_id"
        )
    )

    intent_from_request = (
        data.get("intent") or {}
    )

    if not sid:
        sid = _generate_session_id()

    session = get_or_create_session(
        sid,
        None
    )

    # --------------------------------------------------------
    # Use existing session intent.
    # --------------------------------------------------------

    intent = session.get("intent") or {}

    # First request for a new session.
    if not intent and intent_from_request:

        session["intent"] = (
            intent_from_request
        )

        session["intent_fingerprint"] = (
            _intent_fingerprint(
                intent_from_request
            )
        )

        intent = intent_from_request

    if not intent:
        return jsonify({
            "error":
                "No active intent profile for session",
            "session_id": sid,
            "results": []
        }), 400

    strictness = float(
        data.get(
            "strictness",
            intent.get(
                "settings",
                {}
            ).get(
                "strictness",
                0.65
            )
        )
    )

    videos = data.get("videos") or []

    # --------------------------------------------------------
    # IMPORTANT:
    # Evaluate current feed BEFORE adding this batch.
    # --------------------------------------------------------

    feed_eval = (
        divergence_service.evaluate_dual_feed(
            intent,
            session["feed_history"]
        )
    )

    feed_interv = (
        intervention_service
        .compute_feed_intervention(
            feed_eval,
            strictness=strictness
        )
    )

    results = []

    for video in videos:

        # Ignore malformed items safely.
        if not isinstance(video, dict):
            continue

        res = _process_video_card(
            video,
            intent,
            strictness,
            feed_interv
        )

        results.append(res)

        session["feed_history"].append(
            res
        )

    while len(
        session["feed_history"]
    ) > MAX_SESSION_HISTORY:

        del session["feed_history"][0]

    # --------------------------------------------------------
    # Recalculate after adding the batch.
    # --------------------------------------------------------

    updated_feed_eval = (
        divergence_service.evaluate_dual_feed(
            intent,
            session["feed_history"]
        )
    )

    updated_interv = (
        intervention_service
        .compute_feed_intervention(
            updated_feed_eval,
            strictness=strictness
        )
    )

    updated_feed_eval[
        "interventionScore"
    ] = updated_interv[
        "intervention_score"
    ]

    updated_feed_eval[
        "interventionLevel"
    ] = updated_interv[
        "intervention_level"
    ]

    session["last_active"] = time.time()

    # --------------------------------------------------------
    # IMPORTANT:
    # Extension expects:
    #
    # data.results
    # data.feedAlignment
    # --------------------------------------------------------

    return jsonify({

        "session_id":
            sid,

        "intent":
            intent,

        "results":
            results,

        "feedAlignment":
            updated_feed_eval,

        "feed_alignment":
            updated_feed_eval,

        "intervention":
            updated_interv,

        "session_history_size":
            len(
                session["feed_history"]
            )
    })


# ============================================================
# FEED PROFILE
# ============================================================

@app.route(
    "/api/feed/profile",
    methods=["POST"]
)
@app.route(
    "/feed-alignment",
    methods=["POST"]
)
def feed_profile():

    try:
        data = request.get_json(force=True) or {}
    except Exception:
        return jsonify({
            "error": "Invalid JSON"
        }), 400

    sid = (
        data.get("session_id")
        or (data.get("intent") or {}).get(
            "session_id"
        )
    )

    if not sid:
        sid = _generate_session_id()

    session = get_or_create_session(
        sid,
        None
    )

    intent = session.get("intent") or {}

    if not intent:
        intent = data.get("intent") or {}

        if intent:
            session["intent"] = intent
            session["intent_fingerprint"] = (
                _intent_fingerprint(intent)
            )

    if not intent:
        return jsonify({
            "error":
                "No active intent profile",
            "session_id": sid
        }), 400

    history = (
        data.get("results")
        or session["feed_history"]
    )

    strictness = float(
        data.get(
            "strictness",
            intent.get(
                "settings",
                {}
            ).get(
                "strictness",
                0.65
            )
        )
    )

    feed_eval = (
        divergence_service.evaluate_dual_feed(
            intent,
            history
        )
    )

    interv = (
        intervention_service
        .compute_feed_intervention(
            feed_eval,
            strictness=strictness
        )
    )

    feed_eval[
        "interventionScore"
    ] = interv[
        "intervention_score"
    ]

    feed_eval[
        "interventionLevel"
    ] = interv[
        "intervention_level"
    ]

    return jsonify({
        **feed_eval,
        "session_id": sid
    })


# ============================================================
# RESET SESSION
# ============================================================

@app.route(
    "/api/session/reset",
    methods=["POST"]
)
def reset_session():

    try:
        data = request.get_json(
            force=True
        ) or {}
    except Exception:
        data = {}

    sid = (
        data.get("session_id")
        or "session_default"
    )

    if sid in SESSIONS:

        SESSIONS[sid]["feed_history"] = []

        SESSIONS[sid]["last_active"] = (
            time.time()
        )

        return jsonify({
            "status": "reset",
            "session_id": sid,
            "history_size": 0
        })

    # If session doesn't exist, create it.
    get_or_create_session(sid)

    return jsonify({
        "status": "reset",
        "session_id": sid,
        "history_size": 0
    })


# ============================================================
# FEEDBACK
# ============================================================

@app.route(
    "/api/feedback",
    methods=["POST"]
)
def feedback_endpoint():

    try:
        data = request.get_json(
            force=True
        ) or {}
    except Exception:
        return jsonify({
            "error": "Invalid JSON"
        }), 400

    intent_profile = (
        data.get("intent_profile")
        or data.get("intent")
        or {}
    )

    feedback_item = (
        data.get("feedback")
        if isinstance(
            data.get("feedback"),
            dict
        )
        else data
    )

    sid = (
        feedback_item.get("session_id")
        or intent_profile.get("session_id")
    )

    if not sid:
        return jsonify({
            "error":
                "session_id is required for feedback"
        }), 400

    session = get_or_create_session(
        sid,
        None
    )

    active_intent = (
        session.get("intent")
        or intent_profile
    )

    if not active_intent:
        return jsonify({
            "error":
                "No active intent for session",
            "session_id": sid
        }), 400

    # --------------------------------------------------------
    # Feedback learning
    # --------------------------------------------------------

    res = feedback_service.record_feedback(
        feedback_item,
        active_intent
    )

    updated_profile = (
        res.get("intent_profile")
        or active_intent
    )

    # IMPORTANT:
    # Feedback updates the CURRENT intent.
    #
    # Do NOT clear feed history here.
    # Feedback is learning, not a new user goal.
    # --------------------------------------------------------

    session["intent"] = updated_profile
    session["intent_fingerprint"] = (
        _intent_fingerprint(
            updated_profile
        )
    )
    session["last_active"] = time.time()

    # --------------------------------------------------------
    # Closed-loop recalculation
    # --------------------------------------------------------

    feed_eval = (
        divergence_service.evaluate_dual_feed(
            updated_profile,
            session["feed_history"]
        )
    )

    strictness = float(
        updated_profile.get(
            "settings",
            {}
        ).get(
            "strictness",
            0.65
        )
    )

    interv = (
        intervention_service
        .compute_feed_intervention(
            feed_eval,
            strictness=strictness
        )
    )

    # --------------------------------------------------------
    # Restore wrongly hidden video
    # --------------------------------------------------------

    action = str(
        feedback_item.get("feedback")
        or ""
    ).lower().strip()

    vid = feedback_item.get(
        "video_id"
    )

    if (
        action in [
            "wrongly_hidden",
            "wrongfully_hidden",
            "show_anyway"
        ]
        and vid
    ):

        for item in session[
            "feed_history"
        ]:

            if item.get(
                "video_id"
            ) == vid:

                item[
                    "controlled_visible"
                ] = True

                item[
                    "decision"
                ] = "RESTORED"

                item[
                    "feedback_override"
                ] = True

    return jsonify({

        "status":
            "updated",

        "session_id":
            sid,

        "intent_profile":
            updated_profile,

        "weight_changes":
            res.get(
                "weight_changes",
                {}
            ),

        "feed_alignment":
            feed_eval,

        "intervention":
            interv
    })


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    import os

    port = int(os.environ.get("PORT", 5000))

    print(f"[IntentAlign V1.3] Starting backend on 0.0.0.0:{port}")

    app.run(
        host="0.0.0.0",
        port=port,
        debug=False
    )
