/**
 * Content Analyzer Module for IntentAlign
 * Dispatches analysis to Flask backend or falls back to local lexical evaluator
 */
const Analyzer = {
  async analyzeBatch(backendUrl, intent, batch, strictness) {
    const url = (backendUrl || 'http://127.0.0.1:5000') + '/api/video/analyze_batch';
    const res = await fetch(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        intent,
        videos: batch.map(b => b.data),
        strictness: strictness ?? 0.65
      })
    });
    if (!res.ok) throw new Error('HTTP ' + res.status);
    return await res.json();
  }
};
if (typeof module !== 'undefined') module.exports = Analyzer;
