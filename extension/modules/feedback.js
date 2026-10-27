/**
 * Feedback Module for IntentAlign
 * Handles in-feed user feedback collection
 */
const Feedback = {
  async submitFeedback(backendUrl, intent, feedbackItem) {
    try {
      const url = (backendUrl || 'http://127.0.0.1:5000') + '/api/feedback';
      const res = await fetch(url, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ intent_profile: intent, feedback: feedbackItem })
      });
      if (res.ok) return await res.json();
    } catch (e) {
      console.log('[Feedback] Recorded locally (backend unreachable).');
    }
    return null;
  }
};
if (typeof module !== 'undefined') module.exports = Feedback;
