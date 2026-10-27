/**
 * IntentAlign Background Service Worker (Manifest V3)
 * Handles extension installation, initial storage setup, badge updates, and message dispatch.
 */

chrome.runtime.onInstalled.addListener(() => {
  chrome.storage.local.get(["settings", "stats", "intentProfile"], (result) => {
    const defaultSettings = {
      enabled: true,
      mode: "adaptive",
      strictness: 0.65,
      backendUrl: "http://127.0.0.1:5000",
      observationWindow: 50,
      feedbackLearningEnabled: true,
      showExplanations: true,
      hardNegatives: ["Politics", "Celebrity Gossip"],
      softNegatives: ["News", "Entertainment", "Gaming", "Sports"]
    };

    const updates = {};
    if (!result.settings) {
      updates.settings = defaultSettings;
    }
    if (!result.stats) {
      updates.stats = {
        analyzed: 0,
        allowed: 0,
        suppressed: 0,
        deprioritized: 0,
        lastUpdated: null,
        intentAlignment: null,
        jsDivergence: null,
        interventionLevel: "LOW",
        preferredExposure: 0,
        unwantedExposure: 0,
        topicDiversity: 0
      };
    }
    if (!result.feedbackHistory) {
      updates.feedbackHistory = [];
    }

    if (Object.keys(updates).length > 0) {
      chrome.storage.local.set(updates, () => {
        console.log("[IntentAlign] Initialized default extension storage.");
      });
    }
    updateBadge(result.settings?.enabled ?? true, "LOW");
  });
});

chrome.storage.onChanged.addListener((changes, areaName) => {
  if (areaName === "local") {
    if (changes.settings || changes.stats) {
      chrome.storage.local.get(["settings", "stats"], (data) => {
        const enabled = data.settings?.enabled ?? true;
        const level = data.stats?.interventionLevel ?? "LOW";
        updateBadge(enabled, level);
      });
    }
  }
});

function updateBadge(enabled, level) {
  if (!enabled) {
    chrome.action.setBadgeText({ text: "OFF" });
    chrome.action.setBadgeBackgroundColor({ color: "#757575" });
  } else {
    chrome.action.setBadgeText({ text: level ? level.substring(0, 3) : "ON" });
    if (level === "HIGH") {
      chrome.action.setBadgeBackgroundColor({ color: "#D32F2F" });
    } else if (level === "MEDIUM") {
      chrome.action.setBadgeBackgroundColor({ color: "#F57C00" });
    } else {
      chrome.action.setBadgeBackgroundColor({ color: "#388E3C" });
    }
  }
}
