/**
 * IntentAlign Extension Popup Logic
 * V1.4 - Intent Update + New Session + Home Feed Rebuild
 *
 * Main behavior:
 * User changes Current Intent
 *        ↓
 * New intent session created
 *        ↓
 * Backend builds new Intent Profile
 *        ↓
 * Old feed statistics/cache reset
 *        ↓
 * New profile stored
 *        ↓
 * YouTube tabs notified
 *        ↓
 * YouTube Home rebuilt and re-analyzed
 */

const $ = (id) => document.getElementById(id);

let currentIntentProfile = null;
let currentSettings = null;
let currentSessionId = "session_default";


// ============================================================
// DEFAULT STATS
// ============================================================

function getBlankStats() {
  return {
    analyzed: 0,
    allowed: 0,
    suppressed: 0,
    deprioritized: 0,

    lastUpdated: null,

    intentAlignment: null,
    rawAlignment: null,
    controlledAlignment: null,

    alignmentImprovement: 0.0,

    jsDivergence: null,
    rawJsd: null,
    controlledJsd: null,

    interventionLevel: "LOW",

    preferredExposure: 0,
    rawPreferredExposure: 0,

    unwantedExposure: 0,
    rawNegativeExposure: 0,

    topicDiversity: 0,

    intentDistribution: {},
    rawFeedDistribution: {},
    feedDistribution: {},
    controlledFeedDistribution: {}
  };
}


// ============================================================
// TAB SWITCHING
// ============================================================

document
  .querySelectorAll(".tab-btn")
  .forEach(btn => {

    btn.addEventListener("click", () => {

      document
        .querySelectorAll(".tab-btn")
        .forEach(b =>
          b.classList.remove("active")
        );

      document
        .querySelectorAll(".tab-content")
        .forEach(c =>
          c.classList.remove("active")
        );

      btn.classList.add("active");

      const targetId =
        btn.getAttribute("data-tab");

      if ($(targetId)) {
        $(targetId).classList.add("active");
      }
    });

  });


// ============================================================
// STRICTNESS
// ============================================================

if ($("strictness")) {

  $("strictness").addEventListener(
    "input",
    (e) => {

      if ($("strictnessVal")) {
        $("strictnessVal").textContent =
          `${e.target.value}%`;
      }

    }
  );

}


// ============================================================
// INITIALIZE POPUP
// ============================================================

async function initPopup() {

  try {

    const data =
      await chrome.storage.local.get([
        "intentProfile",
        "settings",
        "stats",
        "backendStatus",
        "feedbackHistory",
        "sessionId"
      ]);

    currentIntentProfile =
      data.intentProfile || null;

    currentSettings =
      data.settings || {

        enabled: true,

        strictness: 0.65,

        backendUrl:
          "https://intentalign-youtube-extension.onrender.com",

        feedbackLearningEnabled:
          true,

        hardNegatives: [
          "Politics",
          "Celebrity Gossip"
        ],

        softNegatives: [
          "News",
          "Entertainment",
          "Gaming",
          "Sports"
        ],

        showBadges: true
      };


    currentSessionId =
      data.sessionId ||
      currentIntentProfile?.session_id ||
      (
        "session_" +
        Date.now().toString(36)
      );


    await chrome.storage.local.set({
      sessionId:
        currentSessionId
    });


    if ($("sessionTagDisplay")) {

      $("sessionTagDisplay").textContent =
        currentSessionId;

      $("sessionTagDisplay").title =
        currentSessionId;
    }


    if ($("sessionStatus")) {

      $("sessionStatus").textContent =
        `Session: ${currentSessionId}`;
    }


    populateForm(
      currentIntentProfile,
      currentSettings
    );


    updateToggleUI(
      currentSettings.enabled
    );


    renderAnalytics(
      data.stats || getBlankStats()
    );


    renderFeedback(
      data.feedbackHistory || []
    );


    checkBackendHealth(
      currentSettings.backendUrl
    );

  } catch (error) {

    console.error(
      "[IntentAlign Popup] Initialization error:",
      error
    );

  }

}


// ============================================================
// CREATE NEW SESSION
// ============================================================

function createSessionId() {

  return (
    "session_" +
    Date.now().toString(36) +
    "_" +
    Math.random()
      .toString(36)
      .substring(2, 8)
  );

}


// ============================================================
// NEW SESSION BUTTON
// ============================================================

if ($("btnNewSession")) {

  $("btnNewSession").addEventListener(
    "click",
    async () => {

      const newSid =
        createSessionId();

      currentSessionId =
        newSid;


      await chrome.storage.local.set({
        sessionId:
          newSid
      });


      if ($("sessionTagDisplay")) {

        $("sessionTagDisplay").textContent =
          newSid;

        $("sessionTagDisplay").title =
          newSid;
      }


      if ($("sessionStatus")) {

        $("sessionStatus").textContent =
          `Session: ${newSid}`;
      }


      const backendUrl =
        currentSettings?.backendUrl ||
        "https://intentalign-youtube-extension.onrender.com";


      try {

        await fetch(
          `${backendUrl}/api/session/reset`,
          {
            method: "POST",

            headers: {
              "Content-Type":
                "application/json"
            },

            body: JSON.stringify({
              session_id:
                newSid
            })
          }
        );

      } catch (error) {

        console.warn(
          "[IntentAlign] Backend session reset unavailable.",
          error
        );

      }


      const blankStats =
        getBlankStats();


      await chrome.storage.local.set({
        stats:
          blankStats,

        feedbackHistory:
          []
      });


      renderAnalytics(
        blankStats
      );


      renderFeedback([]);


      notifyYouTubeTabs({
        type:
          "newSession",

        sessionId:
          newSid
      });

    }
  );

}


// ============================================================
// POPULATE FORM
// ============================================================

function populateForm(
  profile,
  settings
) {

  if (profile) {

    const userProf =
      profile.user_profile ||
      profile.profile ||
      {};

    const intent =
      profile.current_intent ||
      {};


    if ($("profession")) {

      $("profession").value =
        userProf.profession || "";
    }


    if ($("interests")) {

      $("interests").value =
        Array.isArray(
          userProf.interests
        )
          ? userProf.interests.join(", ")
          : "";
    }


    if ($("hobbies")) {

      $("hobbies").value =
        Array.isArray(
          userProf.hobbies
        )
          ? userProf.hobbies.join(", ")
          : "";
    }


    if ($("goal")) {

      $("goal").value =
        intent.current_goal ||
        profile.goal ||
        "";
    }


    if ($("preferred")) {

      $("preferred").value =
        (
          intent.preferred_topics ||
          profile.preferredTopics ||
          []
        ).join(", ");
    }


    if ($("unwanted")) {

      $("unwanted").value =
        (
          intent.unwanted_topics ||
          profile.unwantedTopics ||
          []
        ).join(", ");
    }


    const strictVal =
      Math.round(
        (
          intent.strictness ??
          settings.strictness ??
          0.65
        ) * 100
      );


    if ($("strictness")) {

      $("strictness").value =
        strictVal;
    }


    if ($("strictnessVal")) {

      $("strictnessVal").textContent =
        `${strictVal}%`;
    }


    renderDerivedTags(
      intent.derived_goal_topics ||
      profile.derivedTopics ||
      []
    );

  }


  if ($("hardNegatives")) {

    $("hardNegatives").value =
      (
        settings.hardNegatives ||
        [
          "Politics",
          "Celebrity Gossip"
        ]
      ).join(", ");
  }


  if ($("softNegatives")) {

    $("softNegatives").value =
      (
        settings.softNegatives ||
        [
          "News",
          "Entertainment",
          "Gaming",
          "Sports"
        ]
      ).join(", ");
  }


  if ($("feedbackLearning")) {

    $("feedbackLearning").checked =
      settings.feedbackLearningEnabled ??
      true;
  }


  if ($("backendUrl")) {

    $("backendUrl").value =
      settings.backendUrl ||
      "https://intentalign-youtube-extension.onrender.com";
  }

}


// ============================================================
// DERIVED TOPICS
// ============================================================

function renderDerivedTags(
  derivedList
) {

  const container =
    $("derivedContainer");

  const tagsWrapper =
    $("derivedTags");


  if (!container || !tagsWrapper) {
    return;
  }


  tagsWrapper.innerHTML = "";


  if (
    !derivedList ||
    derivedList.length === 0
  ) {

    container.classList.add(
      "hidden"
    );

    return;
  }


  container.classList.remove(
    "hidden"
  );


  derivedList.forEach(
    (item, index) => {

      const topicName =
        typeof item === "string"
          ? item
          : (
              item.topic ||
              ""
            );


      const weight =
        typeof item === "object" &&
        item.weight
          ? Math.round(
              item.weight * 100
            )
          : null;


      const tag =
        document.createElement(
          "div"
        );

      tag.className =
        "topic-tag";


      const text =
        document.createElement(
          "span"
        );

      text.textContent =
        topicName +
        (
          weight
            ? ` ${weight}%`
            : ""
        );


      const remove =
        document.createElement(
          "span"
        );

      remove.className =
        "remove-btn";

      remove.title =
        "Remove topic";

      remove.textContent =
        "✕";

      remove.dataset.index =
        index;


      tag.append(
        text,
        remove
      );


      tagsWrapper.appendChild(
        tag
      );

    }
  );


  tagsWrapper
    .querySelectorAll(
      ".remove-btn"
    )
    .forEach(btn => {

      btn.addEventListener(
        "click",
        async (event) => {

          const idx =
            parseInt(
              event.target
                .dataset
                .index,
              10
            );


          if (
            !currentIntentProfile
          ) {
            return;
          }


          const intent =
            currentIntentProfile
              .current_intent ||
            {};


          const dTopics =
            intent.derived_goal_topics ||
            currentIntentProfile.derivedTopics;


          if (
            dTopics &&
            dTopics[idx]
          ) {

            dTopics.splice(
              idx,
              1
            );


            await chrome.storage.local.set({
              intentProfile:
                currentIntentProfile
            });


            renderDerivedTags(
              dTopics
            );

          }

        }
      );

    });

}


// ============================================================
// TOGGLE UI
// ============================================================

function updateToggleUI(
  enabled
) {

  const btn =
    $("toggleFiltering");


  if (!btn) {
    return;
  }


  if (enabled) {

    btn.textContent =
      "Active";

    btn.className =
      "toggle-btn active";

  } else {

    btn.textContent =
      "Disabled";

    btn.className =
      "toggle-btn";

  }

}


// ============================================================
// BACKEND HEALTH
// ============================================================

async function checkBackendHealth(
  url
) {

  const badge =
    $("backendBadge");


  if (!badge) {
    return;
  }


  try {

    const res =
      await fetch(
        `${url}/api/health`,
        {
          method: "GET"
        }
      );


    if (res.ok) {

      const data =
        await res.json();


      badge.textContent =
        data.is_neural
          ? "AI Neural"
          : "AI Active";


      badge.className =
        "status-badge online";


      badge.title =
        `Backend online (${data.embedding_model || "AI model"})`;

    } else {

      throw new Error(
        `HTTP ${res.status}`
      );

    }

  } catch (error) {

    badge.textContent =
      "Offline Fallback";

    badge.className =
      "status-badge offline";

    badge.title =
      "Backend unreachable; local fallback active.";

  }

}


// ============================================================
// SAVE / UPDATE INTENT
// ============================================================

if ($("saveProfile")) {

  $("saveProfile").addEventListener(
    "click",
    async () => {

      const goal =
        $("goal")?.value.trim() ||
        "";


      const statusEl =
        $("intentStatus");


      if (!goal) {

        if (statusEl) {

          statusEl.textContent =
            "Please enter your Current Goal.";

          statusEl.className =
            "status-msg error";
        }

        return;
      }


      if (statusEl) {

        statusEl.textContent =
          "Creating a new intent session...";

        statusEl.className =
          "status-msg info";
      }


      const backendUrl =
        $("backendUrl")?.value.trim() ||
        "https://intentalign-youtube-extension.onrender.com";


      const strictness =
        parseFloat(
          $("strictness")?.value ||
          "65"
        ) / 100;


      // ========================================================
      // IMPORTANT:
      // Every intent update gets a NEW SESSION.
      // This prevents old Kubernetes/ML/etc.
      // feed history from contaminating the new intent.
      // ========================================================

      const oldSessionId =
        currentSessionId;


      const newSessionId =
        createSessionId();


      currentSessionId =
        newSessionId;


      const payload = {

        session_id:
          newSessionId,

        profile: {

          profession:
            $("profession")
              ?.value
              .trim() ||
            "",

          interests:
            (
              $("interests")
                ?.value ||
              ""
            )
              .split(",")
              .map(s =>
                s.trim()
              )
              .filter(Boolean),

          hobbies:
            (
              $("hobbies")
                ?.value ||
              ""
            )
              .split(",")
              .map(s =>
                s.trim()
              )
              .filter(Boolean)

        },

        goal:
          goal,

        preferredTopics:
          (
            $("preferred")
              ?.value ||
            ""
          )
            .split(",")
            .map(s =>
              s.trim()
            )
            .filter(Boolean),

        unwantedTopics:
          (
            $("unwanted")
              ?.value ||
            ""
          )
            .split(",")
            .map(s =>
              s.trim()
            )
            .filter(Boolean),

        strictness:
          strictness
      };


      // ========================================================
      // RESET LOCAL FEED STATE IMMEDIATELY
      // ========================================================

      const blankStats =
        getBlankStats();


      await chrome.storage.local.set({

        sessionId:
          newSessionId,

        stats:
          blankStats,

        feedbackHistory:
          [],

        backendStatus:
          "updating"

      });


      if ($("sessionTagDisplay")) {

        $("sessionTagDisplay").textContent =
          newSessionId;

        $("sessionTagDisplay").title =
          newSessionId;
      }


      if ($("sessionStatus")) {

        $("sessionStatus").textContent =
          `Session: ${newSessionId}`;
      }


      renderAnalytics(
        blankStats
      );


      renderFeedback([]);


      // ========================================================
      // TELL BACKEND ABOUT NEW SESSION
      // ========================================================

      try {

        await fetch(
          `${backendUrl}/api/session/reset`,
          {
            method: "POST",

            headers: {
              "Content-Type":
                "application/json"
            },

            body: JSON.stringify({
              session_id:
                newSessionId,

              previous_session_id:
                oldSessionId
            })
          }
        );

      } catch (error) {

        console.warn(
          "[IntentAlign] New session backend reset unavailable.",
          error
        );

      }


      // ========================================================
      // BUILD NEW SEMANTIC INTENT PROFILE
      // ========================================================

      try {

        const res =
          await fetch(
            `${backendUrl}/api/intent/analyze`,
            {
              method: "POST",

              headers: {
                "Content-Type":
                  "application/json"
              },

              body:
                JSON.stringify(payload)
            }
          );


        if (!res.ok) {

          throw new Error(
            `Backend error ${res.status}`
          );

        }


        const profileData =
          await res.json();


        // ======================================================
        // FORCE SESSION ID TO MATCH NEW SESSION
        // ======================================================

        profileData.session_id =
          newSessionId;


        if (
          profileData.current_intent
        ) {

          profileData
            .current_intent
            .session_id =
              newSessionId;
        }


        currentIntentProfile =
          profileData;


        currentSettings = {

          ...currentSettings,

          enabled:
            true,

          strictness:
            strictness,

          backendUrl:
            backendUrl

        };


        await chrome.storage.local.set({

          intentProfile:
            profileData,

          settings:
            currentSettings,

          sessionId:
            newSessionId,

          stats:
            blankStats,

          feedbackHistory:
            []

        });


        if (statusEl) {

          statusEl.textContent =
            "Intent updated! Rebuilding YouTube Home feed...";

          statusEl.className =
            "status-msg success";
        }


        renderDerivedTags(
          profileData.derivedTopics ||
          profileData
            .current_intent
            ?.derived_goal_topics ||
          []
        );


        updateToggleUI(
          true
        );


        // ======================================================
        // CRITICAL:
        // Notify every YouTube tab.
        // Content script will clear cache,
        // return to Home and re-analyze.
        // ======================================================

        notifyYouTubeTabs({

          type:
            "intentUpdated",

          profile:
            profileData,

          sessionId:
            newSessionId,

          previousSessionId:
            oldSessionId,

          forceHome:
            true,

          rebuildFeed:
            true

        });


        // Give YouTube time to receive
        // the message, then explicitly
        // focus Home tab if possible.

        setTimeout(
          focusYouTubeHome,
          250
        );


      } catch (error) {

        console.warn(
          "[IntentAlign] Backend unavailable. Using local intent profile.",
          error
        );


        // ======================================================
        // OFFLINE FALLBACK PROFILE
        // ======================================================

        const fallbackDerived =
          payload.preferredTopics.map(
            topic => ({
              topic,
              weight: 1.0
            })
          );


        const fallbackProfile = {

          version:
            "1.4.0",

          session_id:
            newSessionId,

          profile:
            payload.profile,

          user_profile:
            payload.profile,

          goal:
            payload.goal,

          preferredTopics:
            payload.preferredTopics,

          unwantedTopics:
            payload.unwantedTopics,

          derivedTopics:
            fallbackDerived,

          current_intent: {

            session_id:
              newSessionId,

            current_goal:
              payload.goal,

            preferred_topics:
              payload.preferredTopics,

            unwanted_topics:
              payload.unwantedTopics,

            derived_goal_topics:
              fallbackDerived,

            strictness:
              strictness

          },

          settings: {

            strictness:
              strictness

          }

        };


        currentIntentProfile =
          fallbackProfile;


        currentSettings = {

          ...currentSettings,

          enabled:
            true,

          strictness:
            strictness,

          backendUrl:
            backendUrl

        };


        await chrome.storage.local.set({

          intentProfile:
            fallbackProfile,

          settings:
            currentSettings,

          sessionId:
            newSessionId,

          stats:
            blankStats,

          feedbackHistory:
            []

        });


        if (statusEl) {

          statusEl.textContent =
            "Intent saved locally. Rebuilding YouTube Home feed...";

          statusEl.className =
            "status-msg success";
        }


        renderDerivedTags(
          fallbackProfile
            .derivedTopics
        );


        updateToggleUI(
          true
        );


        notifyYouTubeTabs({

          type:
            "intentUpdated",

          profile:
            fallbackProfile,

          sessionId:
            newSessionId,

          previousSessionId:
            oldSessionId,

          forceHome:
            true,

          rebuildFeed:
            true

        });


        setTimeout(
          focusYouTubeHome,
          250
        );

      }

    }
  );

}


// ============================================================
// FOCUS YOUTUBE HOME
// ============================================================

async function focusYouTubeHome() {

  try {

    const tabs =
      await chrome.tabs.query({
        url:
          "*://*.youtube.com/*"
      });


    if (!tabs.length) {
      return;
    }


    // Prefer the currently active YouTube tab.
    let target =
      tabs.find(
        tab => tab.active
      );


    // Otherwise use first YouTube tab.
    if (!target) {
      target = tabs[0];
    }


    if (!target?.id) {
      return;
    }


    await chrome.tabs.update(
      target.id,
      {
        url:
          "https://www.youtube.com/"
      }
    );


    try {

      await chrome.tabs.update(
        target.id,
        {
          active:
            true
        }
      );

    } catch (_) {}

  } catch (error) {

    console.warn(
      "[IntentAlign] Could not focus YouTube Home:",
      error
    );

  }

}


// ============================================================
// TOGGLE FILTERING
// ============================================================

if ($("toggleFiltering")) {

  $("toggleFiltering").addEventListener(
    "click",
    async () => {

      const newEnabled =
        !currentSettings.enabled;


      currentSettings.enabled =
        newEnabled;


      await chrome.storage.local.set({
        settings:
          currentSettings
      });


      updateToggleUI(
        newEnabled
      );


      notifyYouTubeTabs({

        type:
          "settingsChanged",

        settings:
          currentSettings

      });

    }
  );

}


// ============================================================
// SAVE SETTINGS
// ============================================================

if ($("saveSettings")) {

  $("saveSettings").addEventListener(
    "click",
    async () => {

      const hardNegs =
        (
          $("hardNegatives")
            ?.value ||
          ""
        )
          .split(",")
          .map(s =>
            s.trim()
          )
          .filter(Boolean);


      const softNegs =
        (
          $("softNegatives")
            ?.value ||
          ""
        )
          .split(",")
          .map(s =>
            s.trim()
          )
          .filter(Boolean);


      const feedbackLearning =
        $("feedbackLearning")
          ?.checked ??
        true;


      const backendUrl =
        $("backendUrl")
          ?.value
          .trim() ||
        "https://intentalign-youtube-extension.onrender.com";


      currentSettings = {

        ...currentSettings,

        hardNegatives:
          hardNegs,

        softNegatives:
          softNegs,

        feedbackLearningEnabled:
          feedbackLearning,

        backendUrl:
          backendUrl

      };


      await chrome.storage.local.set({
        settings:
          currentSettings
      });


      if ($("intentStatus")) {

        $("intentStatus").textContent =
          "Settings updated.";

        $("intentStatus").className =
          "status-msg success";
      }


      checkBackendHealth(
        backendUrl
      );


      notifyYouTubeTabs({

        type:
          "settingsChanged",

        settings:
          currentSettings

      });

    }
  );

}


// ============================================================
// RESET SESSION DATA
// ============================================================

if ($("resetSession")) {

  $("resetSession").addEventListener(
    "click",
    async () => {

      if (
        !confirm(
          "Reset current session feed statistics and cache?"
        )
      ) {
        return;
      }


      const blankStats =
        getBlankStats();


      await chrome.storage.local.set({

        stats:
          blankStats,

        feedbackHistory:
          []

      });


      renderAnalytics(
        blankStats
      );


      renderFeedback([]);


      const backendUrl =
        currentSettings?.backendUrl ||
        "https://intentalign-youtube-extension.onrender.com";


      fetch(
        `${backendUrl}/api/session/reset`,
        {
          method: "POST",

          headers: {
            "Content-Type":
              "application/json"
          },

          body: JSON.stringify({
            session_id:
              currentSessionId
          })

        }
      ).catch(() => {});


      notifyYouTubeTabs({

        type:
          "resetSession",

        sessionId:
          currentSessionId

      });

    }
  );

}


// ============================================================
// ANALYTICS
// ============================================================

function renderAnalytics(
  stats
) {

  if (!stats) {
    return;
  }


  if ($("countAnalyzed")) {

    $("countAnalyzed").textContent =
      stats.analyzed || 0;
  }


  if ($("countAllowed")) {

    $("countAllowed").textContent =
      stats.allowed || 0;
  }


  if ($("countDeprioritized")) {

    $("countDeprioritized").textContent =
      stats.deprioritized || 0;
  }


  if ($("countSuppressed")) {

    $("countSuppressed").textContent =
      stats.suppressed || 0;
  }


  // Controlled alignment

  const ctrlAlign =
    stats.controlledAlignment ??
    stats.intentAlignment;


  if ($("metricAlignment")) {

    if (ctrlAlign != null) {

      $("metricAlignment")
        .textContent =
        `${Math.round(
          ctrlAlign * 100
        )}%`;

    } else {

      $("metricAlignment")
        .textContent =
        "--%";
    }

  }


  // Raw alignment

  if ($("metricRawAlignment")) {

    if (
      stats.rawAlignment != null
    ) {

      $("metricRawAlignment")
        .textContent =
        `${Math.round(
          stats.rawAlignment * 100
        )}%`;

    } else {

      $("metricRawAlignment")
        .textContent =
        "--%";
    }

  }


  // Alignment gain

  if ($("metricGain")) {

    if (
      stats.alignmentImprovement !=
      null
    ) {

      const gainPct =
        (
          stats.alignmentImprovement *
          100
        ).toFixed(1);


      $("metricGain").textContent =
        stats.alignmentImprovement >=
        0
          ? `+${gainPct}%`
          : `${gainPct}%`;


      $("metricGain").className =
        `metric-value gain ${
          stats.alignmentImprovement >=
          0
            ? "gain-pos"
            : "gain-neg"
        }`;

    } else {

      $("metricGain").textContent =
        "+0.0%";
    }

  }


  // Diversity

  if ($("metricDiversity")) {

    if (
      stats.topicDiversity !=
      null
    ) {

      $("metricDiversity")
        .textContent =
        Number(
          stats.topicDiversity
        ).toFixed(2);

    } else {

      $("metricDiversity")
        .textContent =
        "--";
    }

  }


  // Exposure

  const prefExp =
    Math.round(
      (
        stats.preferredExposure ||
        0
      ) * 100
    );


  const rawPrefExp =
    Math.round(
      (
        stats.rawPreferredExposure ||
        0
      ) * 100
    );


  const unwExp =
    Math.round(
      (
        stats.unwantedExposure ||
        0
      ) * 100
    );


  const rawUnwExp =
    Math.round(
      (
        stats.rawNegativeExposure ||
        0
      ) * 100
    );


  if ($("prefExposureVal")) {

    $("prefExposureVal").textContent =
      `${prefExp}%`;
  }


  if ($("rawPrefVal")) {

    $("rawPrefVal").textContent =
      `${rawPrefExp}%`;
  }


  if ($("unwExposureVal")) {

    $("unwExposureVal").textContent =
      `${unwExp}%`;
  }


  if ($("rawUnwVal")) {

    $("rawUnwVal").textContent =
      `${rawUnwExp}%`;
  }


  if ($("prefProgressBar")) {

    $("prefProgressBar")
      .style.width =
      `${prefExp}%`;
  }


  if ($("unwProgressBar")) {

    $("unwProgressBar")
      .style.width =
      `${unwExp}%`;
  }


  renderDistributionsChart(

    stats.intentDistribution,

    stats.rawFeedDistribution,

    stats.controlledFeedDistribution ||
    stats.feedDistribution

  );

}


// ============================================================
// DISTRIBUTION CHART
// ============================================================

function renderDistributionsChart(
  pDist,
  qRawDist,
  qCtrlDist
) {

  const container =
    $("chartContainer");


  if (!container) {
    return;
  }


  if (
    !pDist &&
    !qRawDist &&
    !qCtrlDist
  ) {

    return;
  }


  const allTopics =
    new Set([

      ...Object.keys(
        pDist || {}
      ),

      ...Object.keys(
        qRawDist || {}
      ),

      ...Object.keys(
        qCtrlDist || {}
      )

    ]);


  if (
    allTopics.size === 0
  ) {

    return;
  }


  const sortedTopics =
    Array.from(
      allTopics
    ).sort(
      (a, b) => {

        const sumA =
          (
            pDist?.[a] ||
            0
          ) +
          (
            qRawDist?.[a] ||
            0
          ) +
          (
            qCtrlDist?.[a] ||
            0
          );


        const sumB =
          (
            pDist?.[b] ||
            0
          ) +
          (
            qRawDist?.[b] ||
            0
          ) +
          (
            qCtrlDist?.[b] ||
            0
          );


        return sumB - sumA;

      }
    );


  let html = "";


  sortedTopics
    .slice(0, 8)
    .forEach(topic => {

      const pVal =
        Math.round(
          (
            pDist?.[topic] ||
            0
          ) * 100
        );


      const qRawVal =
        Math.round(
          (
            qRawDist?.[topic] ||
            0
          ) * 100
        );


      const qCtrlVal =
        Math.round(
          (
            qCtrlDist?.[topic] ||
            0
          ) * 100
        );


      html += `

        <div class="chart-row">

          <div class="chart-row-label">

            <span>
              <b>
                ${escapeHtml(topic)}
              </b>
            </span>

            <span>
              P: <b>${pVal}%</b>
              · Raw: ${qRawVal}%
              · Ctrl:
              <b style="color:#10b981;">
                ${qCtrlVal}%
              </b>
            </span>

          </div>


          <div class="bars-wrap">

            <div
              class="bar-intent"
              style="width:${Math.min(
                100,
                pVal * 1.8
              )}%"
              title="Target Intent P: ${pVal}%"
            ></div>


            <div
              class="bar-feed"
              style="width:${Math.min(
                100,
                qRawVal * 1.8
              )}%"
              title="Raw Feed Q: ${qRawVal}%"
            ></div>


            <div
              class="bar-ctrl"
              style="width:${Math.min(
                100,
                qCtrlVal * 1.8
              )}%"
              title="Controlled Feed Q_ctrl: ${qCtrlVal}%"
            ></div>

          </div>

        </div>

      `;

    });


  container.innerHTML =
    html;

}


// ============================================================
// FEEDBACK DISPLAY
// ============================================================

function renderFeedback(
  history
) {

  const container =
    $("feedbackList");


  if (!container) {
    return;
  }


  if (
    !history ||
    history.length === 0
  ) {

    container.innerHTML =
      '<div class="empty-note">No feedback recorded yet. Use the feedback buttons on YouTube cards to guide the system.</div>';

    return;
  }


  let html = "";


  history
    .slice(-10)
    .reverse()
    .forEach(item => {

      const actionLabel =
        item.feedback ===
        "relevant"

          ? "Relevant"

          : item.feedback ===
            "not_relevant"

            ? "Not Relevant"

            : "Wrongly Hidden";


      const title =
        String(
          item.title ||
          "Video"
        );


      html += `

        <div class="feedback-item">

          <span
            title="${escapeHtml(
              title
            )}"
          >
            ${escapeHtml(
              title.substring(
                0,
                32
              )
            )}${title.length > 32 ? "..." : ""}
          </span>

          <b>
            ${actionLabel}
          </b>

        </div>

      `;

    });


  container.innerHTML =
    html;

}


// ============================================================
// SEND MESSAGE TO YOUTUBE TABS
// ============================================================

function notifyYouTubeTabs(
  message
) {

  chrome.tabs.query(
    {
      url:
        "*://*.youtube.com/*"
    },
    tabs => {

      tabs.forEach(tab => {

        if (!tab.id) {
          return;
        }


        chrome.tabs
          .sendMessage(
            tab.id,
            message
          )
          .catch(() => {

            // Old content scripts can
            // be invalid after extension
            // reload. Ignore safely.

          });

      });

    }
  );

}


// ============================================================
// HTML ESCAPE
// ============================================================

function escapeHtml(
  str
) {

  return String(
    str ?? ""
  ).replace(
    /[&<>'"]/g,
    c => ({
      "&":
        "&amp;",
      "<":
        "&lt;",
      ">":
        "&gt;",
      "'":
        "&#39;",
      '"':
        "&quot;"
    }[c])
  );

}


// ============================================================
// AUTO REFRESH ANALYTICS
// ============================================================

setInterval(
  async () => {

    try {

      const data =
        await chrome.storage.local.get([
          "stats",
          "feedbackHistory"
        ]);


      renderAnalytics(
        data.stats ||
        getBlankStats()
      );


      renderFeedback(
        data.feedbackHistory ||
        []
      );

    } catch (error) {

      console.warn(
        "[IntentAlign] Analytics refresh failed:",
        error
      );

    }

  },
  1200
);


// ============================================================
// START
// ============================================================

initPopup();