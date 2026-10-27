/**
 * IntentAlign YouTube Content Script
 * V1.3 - Session-Aware Closed-Loop Engine
 *
 * Features:
 * - YouTube Home/Search/Subscriptions/Watch feed monitoring
 * - Current intent analysis
 * - ALLOW / DEPRIORITIZE / SUPPRESS
 * - Intent-based Home feed reordering
 * - Reversible "Show Anyway"
 * - Relevant / Not Relevant feedback
 * - Session-aware caching
 * - SPA navigation support
 * - Backend + offline fallback
 */

(() => {
  if (window.__INTENT_ALIGN_INJECTED__) return;
  window.__INTENT_ALIGN_INJECTED__ = true;

  console.log(
    "[IntentAlign V1.3] Content script initialized on YouTube."
  );

  const state = {
    queue: [],
    processing: false,
    analysisCache: new Map(),

    // IntentAlign recommendations shown at the top of YouTube Home
    recommendations: new Map(),
    recommendationPanel: null,

    batchSize: 8,
    backendOnline: true,
    lastAlignmentProfile: null,
    sessionId: null,
    lastUrl: location.href
  };

  const SELECTORS = [
    "ytd-rich-item-renderer",
    "ytd-video-renderer",
    "ytd-grid-video-renderer",
    "ytd-compact-video-renderer"
  ];

  // ============================================================
  // VIDEO EXTRACTION
  // ============================================================

  function extractVideoId(card, href = "") {
    if (href) {
      const match = href.match(
        /[?&]v=([a-zA-Z0-9_-]{11})/
      );

      if (match) return match[1];
    }

    const idEl = card.querySelector(
      "[data-video-id], a[href*='/watch?v=']"
    );

    if (idEl) {
      const url =
        idEl.getAttribute("href") || "";

      const match = url.match(
        /[?&]v=([a-zA-Z0-9_-]{11})/
      );

      if (match) return match[1];
    }

    return "";
  }

  function extractCardTitle(card) {
    const el = card.querySelector(
      "#video-title, a#video-title, h3 a, #video-title-link"
    );

    return (
      el?.textContent?.trim() ||
      el?.getAttribute("title")?.trim() ||
      ""
    );
  }

  function extractCardChannel(card) {
    const el = card.querySelector(
      "#channel-name, ytd-channel-name, .ytd-channel-name"
    );

    return el?.textContent?.trim() || "";
  }

  function extractCardDescription(card) {
    const el = card.querySelector(
      "#description-text, .metadata-snippet-text, #metadata-line"
    );

    return el?.textContent?.trim() || "";
  }

  function extractCardUrl(card) {
    const el = card.querySelector(
      "a#thumbnail, a#video-title, a.ytd-thumbnail"
    );

    return el?.href || "";
  }

  // ============================================================
  // STORAGE / INTENT
  // ============================================================

  async function getSettingsAndIntent() {
    try {
      const storage =
        await chrome.storage.local.get([
          "settings",
          "intentProfile",
          "sessionId"
        ]);

      const settings =
        storage.settings || {
          enabled: true,
          strictness: 0.65,
          backendUrl:
            "http://127.0.0.1:5000",
          showBadges: true
        };

      const intent =
        storage.intentProfile;

      const sessionId =
        storage.sessionId ||
        intent?.session_id ||
        null;

      if (sessionId !== state.sessionId) {
        state.sessionId = sessionId;
        state.analysisCache.clear();
        state.queue = [];
      }

      return {
        settings,
        intent,
        sessionId
      };

    } catch (error) {
      if (
        String(error?.message || "")
          .toLowerCase()
          .includes(
            "extension context invalidated"
          )
      ) {
        console.warn(
          "[IntentAlign] Extension context invalidated. Refresh YouTube."
        );

        state.queue = [];
        state.processing = false;

        return {
          settings: {
            enabled: false
          },
          intent: null,
          sessionId: null
        };
      }

      console.error(
        "[IntentAlign] Storage error:",
        error
      );

      return {
        settings: {
          enabled: false
        },
        intent: null,
        sessionId: null
      };
    }
  }

  // ============================================================
  // COLLECT YOUTUBE CARDS
  // ============================================================

  async function collectAndQueue() {
    const {
      settings,
      intent
    } = await getSettingsAndIntent();

    if (!settings.enabled || !intent) {
      document
        .querySelectorAll(
          SELECTORS.join(",")
        )
        .forEach(card => {
          resetCard(card);
        });

      return;
    }

    const cards =
      document.querySelectorAll(
        SELECTORS.join(",")
      );

    const newItems = [];

    for (const card of cards) {
      if (
        card.dataset.intentAlignSeen ===
        "1"
      ) {
        continue;
      }

      const title =
        extractCardTitle(card);

      if (!title) {
        continue;
      }

      const url =
        extractCardUrl(card);

      const videoId =
        extractVideoId(card, url) ||
        title;

      card.dataset.intentAlignSeen =
        "1";

      card.dataset.intentAlignVid =
        videoId;

      const cacheKey =
        `${state.sessionId || "no-session"}:${videoId}`;

      if (
        state.analysisCache.has(
          cacheKey
        )
      ) {
        const cached =
          state.analysisCache.get(
            cacheKey
          );

        applyIntervention(
          card,
          cached,
          settings
        );

        continue;
      }

      newItems.push({
        card,
        data: {
          video_id: videoId,
          title,
          channel:
            extractCardChannel(card),
          description:
            extractCardDescription(card),
          url
        }
      });
    }

    if (newItems.length > 0) {
      state.queue.push(...newItems);

      processBatchQueue();

    } else {
      setTimeout(
        reorderHomeFeed,
        100
      );
    }
  }

  // ============================================================
  // BACKEND BATCH ANALYSIS
  // ============================================================

  async function processBatchQueue() {
    if (
      state.processing ||
      state.queue.length === 0
    ) {
      return;
    }

    state.processing = true;

    const {
      settings,
      intent,
      sessionId
    } = await getSettingsAndIntent();

    if (
      !settings.enabled ||
      !intent
    ) {
      const items =
        state.queue.splice(0);

      items.forEach(item =>
        resetCard(item.card)
      );

      state.processing = false;
      return;
    }

    const batch =
      state.queue.splice(
        0,
        state.batchSize
      );

    const backendUrl =
      settings.backendUrl ||
      "http://127.0.0.1:5000";

    try {
      console.log(
        `[IntentAlign] Analyzing batch of ${batch.length} cards`
      );

      const response =
        await fetch(
          `${backendUrl}/api/video/analyze_batch`,
          {
            method: "POST",
            headers: {
              "Content-Type":
                "application/json"
            },
            body: JSON.stringify({
              session_id: sessionId,
              intent,
              videos:
                batch.map(
                  item => item.data
                ),
              strictness:
                settings.strictness ??
                0.65
            })
          }
        );

      if (!response.ok) {
        throw new Error(
          `HTTP ${response.status}`
        );
      }

      const data =
        await response.json();

      state.backendOnline = true;

      const results =
        Array.isArray(data.results)
          ? data.results
          : [];

      results.forEach(
        (result, index) => {
          if (result.video_id) {
            state.analysisCache.set(
              `${sessionId}:${result.video_id}`,
              result
            );
          }

          if (batch[index]) {
            applyIntervention(
              batch[index].card,
              result,
              settings
            );
          }
        }
      );

      if (data.feedAlignment) {
        state.lastAlignmentProfile =
          data.feedAlignment;

        await updateStatsStorage(
          results,
          data.feedAlignment
        );
      }

      setTimeout(
        reorderHomeFeed,
        150
      );

    } catch (error) {
      console.warn(
        "[IntentAlign] Backend unavailable. Using offline fallback.",
        error
      );

      state.backendOnline = false;

      const fallbackResults =
        batch.map(item =>
          evaluateOfflineFallback(
            item.data,
            intent
          )
        );

      fallbackResults.forEach(
        (result, index) => {
          if (result.video_id) {
            state.analysisCache.set(
              `${sessionId}:${result.video_id}`,
              result
            );
          }

          applyIntervention(
            batch[index].card,
            result,
            settings
          );
        }
      );

      await updateStatsStorage(
        fallbackResults,
        null
      );

      setTimeout(
        reorderHomeFeed,
        150
      );

    } finally {
      state.processing = false;

      if (state.queue.length > 0) {
        setTimeout(
          processBatchQueue,
          60
        );
      }
    }
  }

  // ============================================================
// INTENTALIGN HOME RECOMMENDATIONS
// ============================================================

function renderIntentRecommendations() {
  if (!isYouTubeHomePage()) {
    return;
  }

  const recommendations = Array.from(
    state.recommendations.values()
  )
    .filter(item => {
      return (
        item &&
        item.video_id &&
        item.title &&
        item.decision !== "SUPPRESS"
      );
    })
    .sort((a, b) => {
      return (
        Number(b.alignment || 0) -
        Number(a.alignment || 0)
      );
    })
    .filter(item => Number(item.alignment || 0) >= 0.55)
    .slice(0, 6);

  let panel =
    document.querySelector(
      "#intentalign-recommendation-panel"
    );

  if (!panel) {
    panel = document.createElement("div");

    panel.id =
      "intentalign-recommendation-panel";

    panel.style.cssText = `
      width: calc(100% - 32px);
      max-width: 1200px;
      margin: 16px auto 24px auto;
      padding: 18px;
      background: #18181f;
      border: 1px solid rgba(99,102,241,.45);
      border-radius: 14px;
      box-sizing: border-box;
      font-family: Roboto, Arial, sans-serif;
      color: white;
      position: relative;
      z-index: 20;
    `;

    const title =
      document.createElement("div");

    title.textContent =
      "🎯 IntentAlign — Recommended for Your Current Goal";

    title.style.cssText = `
      font-size: 18px;
      font-weight: 700;
      margin-bottom: 5px;
      color: #a5b4fc;
    `;

    const subtitle =
      document.createElement("div");

    subtitle.id =
      "intentalign-recommendation-subtitle";

    subtitle.style.cssText = `
      font-size: 12px;
      color: #9ca3af;
      margin-bottom: 15px;
    `;

    panel.append(
      title,
      subtitle
    );

    const list =
      document.createElement("div");

    list.id =
      "intentalign-recommendation-list";

    list.style.cssText = `
      display: grid;
      grid-template-columns:
        repeat(auto-fit, minmax(220px, 1fr));
      gap: 12px;
    `;

    panel.appendChild(list);

    /*
     * Insert near the top of YouTube Home.
     * We deliberately do not move YouTube's
     * own renderer elements.
     */
    const target =
      document.querySelector(
        "ytd-browse[page-subtype='home'] #contents"
      ) ||
      document.querySelector(
        "ytd-browse #contents"
      );

    if (target) {
      target.prepend(panel);
    } else if (document.body) {
      document.body.prepend(panel);
    }

    state.recommendationPanel =
      panel;
  }

  const subtitle =
    panel.querySelector(
      "#intentalign-recommendation-subtitle"
    );

  const list =
    panel.querySelector(
      "#intentalign-recommendation-list"
    );

  if (!subtitle || !list) {
    return;
  }

  /*
   * Read current intent.
   */
  chrome.storage.local
    .get("intentProfile")
    .then(data => {

      const intent =
        data.intentProfile || {};

      const currentIntent =
        intent.current_intent || {};

      const goal =
        intent.goal ||
        intent.current_goal ||
        currentIntent.current_goal ||
        "your current goal";

      subtitle.textContent =
        `Focused on: ${goal}`;

    })
    .catch(() => {
      subtitle.textContent =
        "Videos aligned with your current intent";
    });

  list.replaceChildren();

  if (recommendations.length === 0) {

    const empty =
      document.createElement("div");

    empty.style.cssText = `
      padding: 18px;
      border: 1px dashed
        rgba(156,163,175,.35);
      border-radius: 10px;
      color: #9ca3af;
      font-size: 13px;
      grid-column: 1 / -1;
      text-align: center;
    `;

    empty.textContent =
      "No strongly relevant videos have been loaded yet. " +
      "Scroll the Home feed to let IntentAlign analyze more videos.";

    list.appendChild(empty);

    return;
  }

  recommendations.forEach(item => {

    const card =
      document.createElement("div");

    card.style.cssText = `
      background: #22222b;
      border: 1px solid
        rgba(255,255,255,.10);
      border-radius: 10px;
      overflow: hidden;
      cursor: pointer;
      transition:
        transform .15s ease,
        border-color .15s ease;
    `;

    card.addEventListener(
      "mouseenter",
      () => {
        card.style.transform =
          "translateY(-2px)";
        card.style.borderColor =
          "rgba(129,140,248,.7)";
      }
    );

    card.addEventListener(
      "mouseleave",
      () => {
        card.style.transform =
          "translateY(0)";
        card.style.borderColor =
          "rgba(255,255,255,.10)";
      }
    );

    const link =
      document.createElement("a");

    const videoId =
      String(item.video_id);

    link.href =
      `https://www.youtube.com/watch?v=${encodeURIComponent(videoId)}`;

    link.target = "_self";

    link.style.cssText = `
      display:block;
      color:inherit;
      text-decoration:none;
    `;

    /*
     * Thumbnail
     */
    const thumbnail =
      document.createElement("img");

    thumbnail.src =
      `https://i.ytimg.com/vi/${encodeURIComponent(videoId)}/hqdefault.jpg`;

    thumbnail.alt =
      item.title || "YouTube video";

    thumbnail.style.cssText = `
      width:100%;
      aspect-ratio:16/9;
      object-fit:cover;
      display:block;
      background:#111;
    `;

    /*
     * Information area
     */
    const info =
      document.createElement("div");

    info.style.cssText = `
      padding:10px;
    `;

    const videoTitle =
      document.createElement("div");

    videoTitle.textContent =
      item.title || "Untitled video";

    videoTitle.style.cssText = `
      font-size:13px;
      font-weight:600;
      line-height:1.35;
      color:#f3f4f6;
      display:-webkit-box;
      -webkit-line-clamp:2;
      -webkit-box-orient:vertical;
      overflow:hidden;
    `;

    const channel =
      document.createElement("div");

    channel.textContent =
      item.channel || "YouTube";

    channel.style.cssText = `
      margin-top:5px;
      font-size:11px;
      color:#9ca3af;
      white-space:nowrap;
      overflow:hidden;
      text-overflow:ellipsis;
    `;

    const match =
      document.createElement("div");

    const percentage =
      Math.round(
        Number(item.alignment || 0) *
        100
      );

    match.textContent =
      `🎯 ${percentage}% Intent Match`;

    match.style.cssText = `
      margin-top:8px;
      font-size:11px;
      font-weight:700;
      color:#34d399;
    `;

    info.append(
      videoTitle,
      channel,
      match
    );

    link.append(
      thumbnail,
      info
    );

    card.appendChild(link);

    list.appendChild(card);
  });
}


  // ============================================================
  // HOME PAGE DETECTION
  // ============================================================

  function isYouTubeHomePage() {
    return (
      (
        location.hostname ===
          "www.youtube.com" ||
        location.hostname ===
          "youtube.com"
      ) &&
      (
        location.pathname === "/" ||
        location.pathname ===
          "/feed/home"
      )
    );
  }

 // ============================================================
// INTENT-BASED HOME FEED REORDERING - SAFE VERSION
// ============================================================

function reorderHomeFeed() {
  if (!isYouTubeHomePage()) {
    return;
  }

  try {
    const cards = Array.from(
      document.querySelectorAll(
        SELECTORS.join(",")
      )
    );

    if (cards.length < 2) {
      return;
    }

    /*
     * Only use cards that have already been analyzed.
     */
    const analyzedCards = cards
      .filter(card => {
        return (
          card &&
          card.isConnected &&
          card.dataset.intentAlignState
        );
      })
      .filter(card => {
        return (
          card.dataset.intentAlignState !==
          "SUPPRESS"
        );
      });

    if (analyzedCards.length < 2) {
      return;
    }

    /*
     * Convert each card into a sortable object.
     */
    const scoredCards = analyzedCards.map(card => {

      const decision =
        card.dataset.intentAlignState ||
        "ALLOW";

      const score =
        parseFloat(
          card.dataset.intentAlignScore ||
          "0"
        ) || 0;

      return {
        card,
        decision,
        score,
        parent: card.parentElement
      };

    });

    /*
     * YouTube has many different containers.
     * Only reorder cards that currently have
     * the SAME parent.
     */
    const groups = new Map();

    scoredCards.forEach(item => {

      if (
        !item.parent ||
        !item.card.isConnected
      ) {
        return;
      }

      if (!groups.has(item.parent)) {
        groups.set(
          item.parent,
          []
        );
      }

      groups
        .get(item.parent)
        .push(item);

    });

    const decisionWeight = {
      ALLOW: 3,
      RESTORED: 3,
      DEPRIORITIZE: 1
    };

    groups.forEach((items, parent) => {

      /*
       * Make absolutely sure the parent still
       * exists before modifying the DOM.
       */
      if (
        !parent ||
        !parent.isConnected ||
        items.length < 2
      ) {
        return;
      }

      /*
       * Make sure every card still belongs
       * to this exact parent.
       */
      const validItems =
        items.filter(item => {

          return (
            item.card &&
            item.card.isConnected &&
            item.card.parentElement ===
              parent
          );

        });

      if (validItems.length < 2) {
        return;
      }

      /*
       * Sort:
       *
       * ALLOW       → highest priority
       * RESTORED    → high priority
       * DEPRIORITIZE→ lower priority
       *
       * Then use alignment score.
       */
      validItems.sort((a, b) => {

        const weightA =
          decisionWeight[
            a.decision
          ] || 0;

        const weightB =
          decisionWeight[
            b.decision
          ] || 0;

        if (
          weightA !== weightB
        ) {
          return (
            weightB -
            weightA
          );
        }

        return (
          b.score -
          a.score
        );

      });

      /*
       * Use a DocumentFragment.
       *
       * IMPORTANT:
       * We capture the parent BEFORE moving
       * anything, so parentElement cannot
       * suddenly become null.
       */
      const fragment =
        document.createDocumentFragment();

      validItems.forEach(item => {

        if (
          item.card &&
          item.card.parentElement ===
            parent
        ) {
          fragment.appendChild(
            item.card
          );
        }

      });

      /*
       * Parent may have changed while
       * YouTube was updating its DOM.
       */
      if (
        parent &&
        parent.isConnected &&
        fragment.childNodes.length > 0
      ) {

        parent.appendChild(
          fragment
        );

      }

    });

    console.log(
      "[IntentAlign] Home feed reordered safely according to current intent."
    );

  } catch (error) {

    /*
     * YouTube continuously changes its DOM.
     * Never allow a DOM race to crash
     * the complete extension.
     */
    console.warn(
      "[IntentAlign] Home feed reorder skipped:",
      error
    );

  }
}
  // ============================================================
  // OFFLINE FALLBACK
  // ============================================================

  function evaluateOfflineFallback(
    video,
    intent
  ) {
    const text =
      `${video.title || ""} ` +
      `${video.channel || ""} ` +
      `${video.description || ""}`
        .toLowerCase();

    const currentIntent =
      intent.current_intent || {};

    const preferred =
      intent.preferredTopics ||
      currentIntent.preferred_topics ||
      [];

    const unwanted =
      intent.unwantedTopics ||
      currentIntent.unwanted_topics ||
      [];

    const derived =
      currentIntent
        .derived_goal_topics ||
      intent.derived_goal_topics ||
      [];

    const goal =
      intent.goal ||
      intent.current_goal ||
      currentIntent.current_goal ||
      "";

    const goalTerms =
      [
        goal,
        ...derived
      ]
        .filter(Boolean)
        .map(value =>
          String(value)
            .toLowerCase()
        );

    const goalMatch =
      goalTerms.some(term =>
        text.includes(term)
      );

    const preferredMatch =
      preferred.some(topic =>
        text.includes(
          String(topic)
            .toLowerCase()
        )
      );

    const unwantedMatch =
      unwanted.some(topic =>
        text.includes(
          String(topic)
            .toLowerCase()
        )
      );

    let decision = "ALLOW";
    let alignment = 0.50;
    let reason =
      "No strong lexical match.";

    if (goalMatch) {
      decision = "ALLOW";
      alignment = 0.90;
      reason =
        "Matches current goal.";
    } else if (
      unwantedMatch &&
      !preferredMatch
    ) {
      decision = "SUPPRESS";
      alignment = 0.10;
      reason =
        "Matches unwanted topic.";
    } else if (
      unwantedMatch &&
      preferredMatch
    ) {
      decision = "DEPRIORITIZE";
      alignment = 0.40;
      reason =
        "Mixed relevance.";
    } else if (preferredMatch) {
      decision = "ALLOW";
      alignment = 0.85;
      reason =
        "Matches preferred topic.";
    }

    return {
      video_id:
        video.video_id,

      title:
        video.title,

      channel:
        video.channel,

      decision,

      alignment,

      label:
        decision === "ALLOW"
          ? "Intent Match"
          : decision ===
              "DEPRIORITIZE"
            ? "Low Intent"
            : "Blocked",

      reason,

      positive_relevance:
        preferredMatch ||
        goalMatch
          ? 0.85
          : 0.20,

      negative_relevance:
        unwantedMatch
          ? 0.85
          : 0.10,

      goal_relevance:
        goalMatch
          ? 0.90
          : 0.10,

      confidence:
        goalMatch ||
        preferredMatch ||
        unwantedMatch
          ? 0.80
          : 0.40,

      is_uncertain:
        !(
          goalMatch ||
          preferredMatch ||
          unwantedMatch
        )
    };
  }

  // ============================================================
  // RESET CARD
  // ============================================================

  function resetCard(card) {
    if (!card) return;

    card.style.display = "";
    card.style.opacity = "";
    card.style.filter = "";

    delete card.dataset.intentAlignState;
    delete card.dataset.intentAlignScore;

    const placeholder =
      card.querySelector(
        ".intentalign-placeholder"
      );

    if (placeholder) {
      placeholder.remove();
    }

    const innerContent =
      card.querySelector(
        "#dismissible, #content, ytd-rich-grid-media"
      );

    if (innerContent) {
      innerContent.style.display =
        "";
    }

    Array.from(card.children)
      .forEach(child => {
        if (
          child.style.display ===
          "none"
        ) {
          child.style.display = "";
        }
      });

    const badge =
      card.querySelector(
        ".intentalign-badge-container"
      );

    if (badge) {
      badge.remove();
    }
  }

  // ============================================================
  // APPLY INTERVENTION
  // ============================================================

  function applyIntervention(
    card,
    result,
    settings
  ) {
    if (!card || !result) {
      return;
    }
        // Store analyzed videos for the IntentAlign
    // recommendation section.
    if (
      result.video_id &&
      result.title &&
      result.decision !== "SUPPRESS"
    ) {
      state.recommendations.set(
        String(result.video_id),
        {
          video_id: result.video_id,
          title: result.title,
          channel: result.channel || "",
          alignment:
            Number(result.alignment || 0),
          decision:
            result.decision || "ALLOW",
          reason:
            result.reason || ""
        }
      );

      /*
       * Keep memory bounded.
       */
      if (
        state.recommendations.size > 30
      ) {
        const firstKey =
          state.recommendations
            .keys()
            .next()
            .value;

        if (firstKey) {
          state.recommendations.delete(
            firstKey
          );
        }
      }

      /*
       * Render/update the recommendation
       * section after this video is analyzed.
       */
      setTimeout(
        renderIntentRecommendations,
        100
      );
    }

    card.dataset.intentAlignState =
      result.decision || "ALLOW";

    card.dataset.intentAlignScore =
      String(
        result.alignment ?? 0
      );

    card.style.transition =
      "opacity 0.25s ease, filter 0.25s ease";

    const oldPlaceholder =
      card.querySelector(
        ".intentalign-placeholder"
      );

    if (oldPlaceholder) {
      oldPlaceholder.remove();
    }

    // ----------------------------------------------------------
    // SUPPRESS
    // ----------------------------------------------------------

    if (
      result.decision ===
      "SUPPRESS"
    ) {
      const innerContent =
        card.querySelector(
          "#dismissible, #content, ytd-rich-grid-media"
        );

      if (innerContent) {
        innerContent.style.display =
          "none";
      } else {
        Array.from(card.children)
          .forEach(child => {
            if (
              !child.classList.contains(
                "intentalign-placeholder"
              )
            ) {
              child.style.display =
                "none";
            }
          });
      }

      const placeholder =
        document.createElement(
          "div"
        );

      placeholder.className =
        "intentalign-placeholder";

      placeholder.style.cssText = `
        display:flex;
        flex-direction:column;
        align-items:center;
        justify-content:center;
        padding:16px 14px;
        margin:6px 0;
        background:rgba(22,22,30,.95);
        border:1px dashed rgba(239,68,68,.45);
        border-radius:10px;
        color:#e2e8f0;
        font-family:Roboto,Arial,sans-serif;
        font-size:12px;
        text-align:center;
        box-sizing:border-box;
        min-height:105px;
        z-index:10;
      `;

      const heading =
        document.createElement(
          "div"
        );

      heading.style.cssText =
        "font-weight:700;" +
        "color:#f87171;" +
        "margin-bottom:4px;" +
        "display:flex;" +
        "align-items:center;" +
        "gap:6px;";

      const headingText =
        document.createElement(
          "span"
        );

      headingText.textContent =
        "🛡️ Intent Filtered";

      const matchText =
        document.createElement(
          "span"
        );

      matchText.textContent =
        `${Math.round(
          (result.alignment || 0) *
            100
        )}% Match`;

      matchText.style.cssText =
        "font-size:10px;" +
        "background:rgba(239,68,68,.2);" +
        "padding:2px 6px;" +
        "border-radius:4px;" +
        "border:1px solid rgba(239,68,68,.3);";

      heading.append(
        headingText,
        matchText
      );

      const reasonText =
        document.createElement(
          "div"
        );

      reasonText.textContent =
        result.reason ||
        "Suppressed based on your current goal and unwanted preferences.";

      reasonText.style.cssText =
        "color:#94a3b8;" +
        "font-size:11px;" +
        "margin-bottom:8px;" +
        "max-width:92%;" +
        "word-break:break-word;" +
        "line-height:1.3;";

      const restoreButton =
        document.createElement(
          "button"
        );

      restoreButton.className =
        "ia-restore-btn";

      restoreButton.textContent =
        "Show Anyway";

      restoreButton.style.cssText =
        "background:rgba(99,102,241,.25);" +
        "border:1px solid rgba(99,102,241,.5);" +
        "color:#c7d2fe;" +
        "padding:4px 12px;" +
        "border-radius:6px;" +
        "font-size:11px;" +
        "font-weight:600;" +
        "cursor:pointer;";

      placeholder.append(
        heading,
        reasonText,
        restoreButton
      );

      restoreButton.onclick =
        async event => {
          event.stopPropagation();
          event.preventDefault();

          if (innerContent) {
            innerContent.style.display =
              "";
          } else {
            Array.from(card.children)
              .forEach(child => {
                child.style.display =
                  "";
              });
          }

          placeholder.remove();

          card.style.display = "";
          card.style.opacity = "1";
          card.style.filter = "none";

          card.dataset.intentAlignState =
            "RESTORED";

          attachCardBadge(
            card,
            {
              ...result,
              decision: "RESTORED",
              label: "Restored"
            }
          );

          await sendFeedback(
            result,
            "wrongly_hidden"
          );
        };

      card.appendChild(
        placeholder
      );

      return;
    }

    // ----------------------------------------------------------
    // ALLOW / DEPRIORITIZE
    // ----------------------------------------------------------

    const innerContent =
      card.querySelector(
        "#dismissible, #content, ytd-rich-grid-media"
      );

    if (innerContent) {
      innerContent.style.display =
        "";
    }

    Array.from(card.children)
      .forEach(child => {
        if (
          child.style.display ===
          "none"
        ) {
          child.style.display = "";
        }
      });

    if (
      result.decision ===
      "DEPRIORITIZE"
    ) {
      card.style.display = "";
      card.style.opacity = "0.35";
      card.style.filter =
        "grayscale(40%)";
    } else {
      card.style.display = "";
      card.style.opacity = "1";
      card.style.filter = "none";
    }

    if (
      settings.showBadges !== false
    ) {
      attachCardBadge(
        card,
        result
      );
    }
  }

  // ============================================================
  // BADGES + FEEDBACK BUTTONS
  // ============================================================

  function attachCardBadge(
    card,
    result
  ) {
    let container =
      card.querySelector(
        ".intentalign-badge-container"
      );

    if (!container) {
      container =
        document.createElement(
          "div"
        );

      container.className =
        "intentalign-badge-container";

      container.style.cssText = `
        display:flex;
        align-items:center;
        justify-content:space-between;
        margin-top:4px;
        padding:3px 6px;
        border-radius:4px;
        font-size:11px;
        line-height:1.2;
        background:rgba(26,26,34,.88);
        border:1px solid rgba(255,255,255,.12);
        z-index:10;
      `;

      const meta =
        card.querySelector(
          "#details, #meta, .ytd-video-meta-block"
        );

      if (meta) {
        meta.appendChild(
          container
        );
      } else {
        card.appendChild(
          container
        );
      }
    }

    const isAllow =
      result.decision ===
        "ALLOW" ||
      result.decision ===
        "RESTORED";

    const badgeColor =
      isAllow
        ? "#10b981"
        : "#f59e0b";

    const alignPct =
      Math.round(
        (result.alignment || 0) *
          100
      );

    container.replaceChildren();

    const info =
      document.createElement(
        "div"
      );

    info.style.cssText =
      "display:flex;" +
      "align-items:center;" +
      "gap:5px;";

    info.title =
      result.reason || "";

    const label =
      document.createElement(
        "span"
      );

    label.textContent =
      result.label ||
      result.decision ||
      "Analyzed";

    label.style.cssText =
      `font-weight:700;` +
      `color:${badgeColor};` +
      `font-size:10px;` +
      `text-transform:uppercase;`;

    const percent =
      document.createElement(
        "span"
      );

    percent.textContent =
      `${alignPct}%`;

    percent.style.cssText =
      "color:#9e9eb4;" +
      "font-size:10px;";

    info.append(
      label,
      percent
    );

    const controls =
      document.createElement(
        "div"
      );

    controls.style.cssText =
      "display:flex;gap:4px;";

    const relevantBtn =
      document.createElement(
        "button"
      );

    relevantBtn.textContent =
      "👍";

    relevantBtn.title =
      "Mark as Relevant";

    relevantBtn.style.cssText =
      "background:none;" +
      "border:none;" +
      "cursor:pointer;" +
      "font-size:11px;" +
      "padding:0 2px;";

    const notRelevantBtn =
      document.createElement(
        "button"
      );

    notRelevantBtn.textContent =
      "👎";

    notRelevantBtn.title =
      "Mark as Not Relevant";

    notRelevantBtn.style.cssText =
      "background:none;" +
      "border:none;" +
      "cursor:pointer;" +
      "font-size:11px;" +
      "padding:0 2px;";

    controls.append(
      relevantBtn,
      notRelevantBtn
    );

    container.append(
      info,
      controls
    );

    relevantBtn.onclick =
      event => {
        event.preventDefault();
        event.stopPropagation();

        sendFeedback(
          result,
          "relevant"
        );

        container.style.borderColor =
          "#10b981";
      };

    notRelevantBtn.onclick =
      event => {
        event.preventDefault();
        event.stopPropagation();

        sendFeedback(
          result,
          "not_relevant"
        );

        container.style.borderColor =
          "#ef4444";

        card.style.opacity =
          "0.25";

        card.style.filter =
          "grayscale(60%)";
      };
  }

  // ============================================================
  // FEEDBACK
  // ============================================================

  async function sendFeedback(
    result,
    action
  ) {
    try {
      const data =
        await chrome.storage.local.get([
          "settings",
          "intentProfile",
          "feedbackHistory",
          "sessionId"
        ]);

      const settings =
        data.settings || {};

      const intent =
        data.intentProfile;

      const sessionId =
        data.sessionId ||
        intent?.session_id ||
        state.sessionId;

      const feedbackPayload = {
        session_id: sessionId,
        video_id:
          result.video_id,
        title:
          result.title,
        feedback: action,
        topics:
          result.topic_dict ||
          result.topics ||
          {},
        timestamp:
          new Date().toISOString()
      };

      const history =
        Array.isArray(
          data.feedbackHistory
        )
          ? data.feedbackHistory
          : [];

      history.push(
        feedbackPayload
      );

      await chrome.storage.local.set({
        feedbackHistory:
          history.slice(-200)
      });

      const response =
        await fetch(
          `${
            settings.backendUrl ||
            "http://127.0.0.1:5000"
          }/api/feedback`,
          {
            method: "POST",
            headers: {
              "Content-Type":
                "application/json"
            },
            body: JSON.stringify({
              session_id:
                sessionId,
              intent_profile:
                intent,
              feedback:
                feedbackPayload
            })
          }
        );

      if (!response.ok) {
        return;
      }

      const updated =
        await response.json();

      if (
        updated.intent_profile
      ) {
        await chrome.storage.local.set({
          intentProfile:
            updated.intent_profile
        });

        // New learned weights mean old
        // cached analyses are invalid.
        state.analysisCache.clear();
        state.queue = [];

        document
          .querySelectorAll(
            SELECTORS.join(",")
          )
          .forEach(card => {
            resetCard(card);

            delete card.dataset
              .intentAlignSeen;
          });

        setTimeout(
          collectAndQueue,
          300
        );
      }

    } catch (error) {
      console.warn(
        "[IntentAlign] Feedback request failed:",
        error
      );
    }
  }

  // ============================================================
  // STATISTICS
  // ============================================================

  async function updateStatsStorage(
    results,
    feedAlignment
  ) {
    try {
      const data =
        await chrome.storage.local.get(
          "stats"
        );

      const stats =
        data.stats || {
          analyzed: 0,
          allowed: 0,
          suppressed: 0,
          deprioritized: 0
        };

      stats.analyzed +=
        results.length;

      results.forEach(result => {
        const decision =
          (
            result.decision ||
            ""
          ).toUpperCase();

        if (
          decision === "ALLOW" ||
          decision === "RESTORED"
        ) {
          stats.allowed += 1;
        } else if (
          decision === "SUPPRESS"
        ) {
          stats.suppressed += 1;
        } else if (
          decision === "DEPRIORITIZE"
        ) {
          stats.deprioritized += 1;
        }
      });

      if (feedAlignment) {
        stats.rawAlignment =
          feedAlignment.raw_alignment ??
          0;

        stats.controlledAlignment =
          feedAlignment.controlled_alignment ??
          0;

        stats.alignmentImprovement =
          feedAlignment.alignment_improvement ??
          0;

        stats.rawJsd =
          feedAlignment.raw_jsd ??
          0;

        stats.controlledJsd =
          feedAlignment.controlled_jsd ??
          0;

        stats.rawNegativeExposure =
          feedAlignment.raw_negative_exposure ??
          0;

        stats.unwantedExposure =
          feedAlignment.controlled_negative_exposure ??
          0;

        stats.rawPreferredExposure =
          feedAlignment.raw_preferred_exposure ??
          0;

        stats.preferredExposure =
          feedAlignment.controlled_preferred_exposure ??
          0;

        stats.topicDiversity =
          feedAlignment.topic_diversity ??
          0;

        stats.intentDistribution =
          feedAlignment.intent_distribution ||
          {};

        stats.rawFeedDistribution =
          feedAlignment.raw_feed_distribution ||
          {};

        stats.controlledFeedDistribution =
          feedAlignment.controlled_feed_distribution ||
          {};

        stats.interventionLevel =
          feedAlignment.interventionLevel ||
          "LOW";
      }

      stats.lastUpdated =
        new Date().toISOString();

      await chrome.storage.local.set({
        stats
      });

    } catch (error) {
      console.warn(
        "[IntentAlign] Could not update stats:",
        error
      );
    }
  }

  // ============================================================
  // SPA NAVIGATION
  // ============================================================

  function handleSpaNavigation() {
    console.log(
      "[IntentAlign] YouTube SPA navigation detected."
    );

    state.queue = [];
    state.analysisCache.clear();
    state.recommendations.clear();

    const oldPanel =
      document.querySelector(
        "#intentalign-recommendation-panel"
      );

    if (oldPanel) {
      oldPanel.remove();
    }

    state.recommendationPanel = null;

    document
      .querySelectorAll(
        SELECTORS.join(",")
      )
      .forEach(card => {
        delete card.dataset
          .intentAlignSeen;
      });

    setTimeout(
      collectAndQueue,
      350
    );
  }

  document.addEventListener(
    "yt-navigate-finish",
    handleSpaNavigation
  );

  document.addEventListener(
    "yt-page-data-updated",
    handleSpaNavigation
  );

  window.addEventListener(
    "popstate",
    handleSpaNavigation
  );

  // ============================================================
  // URL WATCHER
  // ============================================================

  setInterval(() => {
    if (
      location.href !==
      state.lastUrl
    ) {
      state.lastUrl =
        location.href;

      handleSpaNavigation();
    }
  }, 1000);

  // ============================================================
  // MUTATION OBSERVER
  // ============================================================

  let debounceTimer = null;

  const observer =
    new MutationObserver(() => {
      if (debounceTimer) {
        clearTimeout(
          debounceTimer
        );
      }

      debounceTimer =
        setTimeout(
          collectAndQueue,
          150
        );
    });

  observer.observe(
    document.documentElement,
    {
      childList: true,
      subtree: true
    }
  );

  // ============================================================
  // INITIAL SCAN
  // ============================================================

  setTimeout(
    collectAndQueue,
    800
  );

  setInterval(
    collectAndQueue,
    3500
  );

  // ============================================================
  // POPUP / INTENT UPDATE
  // ============================================================

  chrome.runtime.onMessage.addListener(
    message => {
      if (
        message.type ===
          "settingsChanged" ||
        message.type ===
          "intentUpdated"
      ) {
        console.log(
          "[IntentAlign] Intent/settings updated. Rebuilding feed..."
        );

        state.analysisCache.clear();
        state.queue = [];
        state.recommendations.clear();

        const oldPanel =
          document.querySelector(
            "#intentalign-recommendation-panel"
          );

        if (oldPanel) {
          oldPanel.remove();
        }

        state.recommendationPanel = null;

        document
          .querySelectorAll(
            SELECTORS.join(",")
          )
          .forEach(card => {
            resetCard(card);

            delete card.dataset
              .intentAlignSeen;
          });

        /*
         * When user changes the intent,
         * return to YouTube Home.
         */
        if (
          message.type ===
            "intentUpdated" &&
          location.pathname !== "/" &&
          location.pathname !==
            "/feed/home"
        ) {
          location.href =
            "https://www.youtube.com/";

          return;
        }

        setTimeout(
          collectAndQueue,
          700
        );

        setTimeout(
          reorderHomeFeed,
          1200
        );

      } else if (
        message.type ===
          "resetSession" ||
        message.type ===
          "newSession"
      ) {
        state.analysisCache.clear();
        state.queue = [];
        state.sessionId = null;

        document
          .querySelectorAll(
            SELECTORS.join(",")
          )
          .forEach(card => {
            resetCard(card);

            delete card.dataset
              .intentAlignSeen;
          });

        setTimeout(
          collectAndQueue,
          500
        );
      }
    }
  );

})();