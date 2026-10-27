/**
 * Intervention Module for IntentAlign
 * Safe client-side DOM presentation alterations (ALLOW, DEPRIORITIZE, SUPPRESS)
 */
const Intervention = {
  apply(card, result, settings) {
    card.dataset.intentAlignState = result.decision;
    card.dataset.intentAlignScore = String(result.alignment ?? 0);
    card.style.transition = 'opacity 0.25s ease, filter 0.25s ease';

    if (result.decision === 'SUPPRESS') {
      card.style.display = 'none';
      return;
    }
    if (result.decision === 'DEPRIORITIZE') {
      card.style.display = '';
      card.style.opacity = '0.32';
      card.style.filter = 'grayscale(40%)';
    } else {
      card.style.display = '';
      card.style.opacity = '1';
      card.style.filter = 'none';
    }
  },
  reset(card) {
    card.style.display = '';
    card.style.opacity = '';
    card.style.filter = '';
    delete card.dataset.intentAlignState;
    const badge = card.querySelector('.intentalign-badge-container');
    if (badge) badge.remove();
  }
};
if (typeof module !== 'undefined') module.exports = Intervention;
