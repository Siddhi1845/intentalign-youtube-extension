/**
 * Feed Monitor Module for IntentAlign
 * Selectors and metadata extraction for YouTube video cards
 */
const FeedMonitor = {
  selectors: [
    'ytd-rich-item-renderer',
    'ytd-video-renderer',
    'ytd-grid-video-renderer',
    'ytd-compact-video-renderer'
  ],
  extractVideoId(card, href) {
    if (href) {
      const m = href.match(/[?&]v=([a-zA-Z0-9_-]{11})/);
      if (m) return m[1];
    }
    const idEl = card.querySelector('[data-video-id], a[href*="/watch?v="]');
    if (idEl) {
      const h = idEl.getAttribute('href') || '';
      const m = h.match(/[?&]v=([a-zA-Z0-9_-]{11})/);
      if (m) return m[1];
    }
    return '';
  },
  extractMetadata(card) {
    const titleEl = card.querySelector('#video-title, a#video-title, h3 a, #video-title-link');
    const title = titleEl?.textContent?.trim() || titleEl?.getAttribute('title')?.trim() || '';
    if (!title) return null;

    const channelEl = card.querySelector('#channel-name, ytd-channel-name, .ytd-channel-name, #text.ytd-channel-name');
    const channel = channelEl?.textContent?.trim() || '';

    const descEl = card.querySelector('#description-text, .metadata-snippet-text, #metadata-line');
    const description = descEl?.textContent?.trim() || '';

    const linkEl = card.querySelector('a#thumbnail, a#video-title, a.ytd-thumbnail');
    const url = linkEl?.href || '';
    const videoId = this.extractVideoId(card, url) || title;

    return { video_id: videoId, title, channel, description, url };
  }
};
if (typeof module !== 'undefined') module.exports = FeedMonitor;
