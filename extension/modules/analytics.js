/**
 * Analytics Module for IntentAlign
 */
const Analytics = {
  calculateAlignmentPct(jsd) {
    if (jsd == null) return null;
    return Math.round(Math.max(0, Math.min(1, 1 - jsd)) * 100);
  }
};
if (typeof module !== 'undefined') module.exports = Analytics;
