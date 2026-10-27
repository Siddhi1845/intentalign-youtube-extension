/**
 * Storage Module for IntentAlign
 * Provides safe Promise-based wrappers for chrome.storage.local
 */
const Storage = {
  async get(keys) {
    return new Promise((resolve) => {
      chrome.storage.local.get(keys, (res) => resolve(res || {}));
    });
  },
  async set(items) {
    return new Promise((resolve) => {
      chrome.storage.local.set(items, () => resolve(true));
    });
  },
  async getSettings() {
    const data = await this.get('settings');
    return data.settings || { enabled: true, strictness: 0.65, backendUrl: 'http://127.0.0.1:5000' };
  },
  async getIntentProfile() {
    const data = await this.get('intentProfile');
    return data.intentProfile || null;
  }
};
if (typeof module !== 'undefined') module.exports = Storage;
