/**
 * Intent Profile Client Module
 */
const Intent = {
  normalizeTopic(topic) {
    const map = {
      'k8s': 'Kubernetes',
      'ml': 'Machine Learning',
      'ai': 'Artificial Intelligence',
      'ci/cd': 'CI/CD',
      'ci cd': 'CI/CD'
    };
    const clean = (topic || '').trim().toLowerCase();
    return map[clean] || topic;
  }
};
if (typeof module !== 'undefined') module.exports = Intent;
