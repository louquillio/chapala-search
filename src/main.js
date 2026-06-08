import { pipeline } from '@huggingface/transformers';

const STATUS = {
  LOADING: { dot: 'bg-amber-400 animate-pulse', text: 'Loading search engine...' },
  READY:   { dot: 'bg-green-500', text: 'Search engine ready' },
  ERROR:   { dot: 'bg-red-500', text: 'Failed to load — check console' },
};

const MAX_GROUPS = 15;
const MIN_QUERY_LENGTH = 2;

// DOM refs
const $ = (id) => document.getElementById(id);
const input    = $('search-input');
const list     = $('results-list');
const empty    = $('empty-state');
const info     = $('results-info');
const count    = $('results-count');
const statusDot   = $('status-dot');
const statusText  = $('status-text');

// State
let extractPipeline = null;
let indexData = [];

function setStatus(state) {
  statusDot.className = `status-dot ${state.dot}`;
  statusText.textContent = state.text;
}

// --- Cosine similarity ---
function cosineSimilarity(a, b) {
  if (a.length !== b.length) return 0;
  let dot = 0, magA = 0, magB = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    magA += a[i] * a[i];
    magB += b[i] * b[i];
  }
  const denom = Math.sqrt(magA) * Math.sqrt(magB);
  return denom === 0 ? 0 : dot / denom;
}

// --- Points heat indicator ---
function pointsColor(p) {
  if (p >= 10) return { bg: 'bg-emerald-100', text: 'text-emerald-700', ring: 'ring-emerald-300', label: 'hot' };
  if (p >= 6)  return { bg: 'bg-amber-100',  text: 'text-amber-700',  ring: 'ring-amber-300',  label: 'warm' };
  if (p >= 3)  return { bg: 'bg-blue-100',   text: 'text-blue-700',   ring: 'ring-blue-300',   label: 'solid' };
  return { bg: 'bg-gray-100', text: 'text-gray-500', ring: 'ring-gray-200', label: 'low' };
}

function pointsBadgeHtml(p) {
  if (!p && p !== 0) return '';
  const c = pointsColor(p);
  return `<span class="inline-flex items-center px-2 py-0.5 rounded-full text-xs font-medium ring-1 ${c.bg} ${c.text} ${c.ring}">${p} pts</span>`;
}

// --- Render grouped results with collapse ---
function renderResults(scoredEntries) {
  list.innerHTML = '';

  if (scoredEntries.length === 0) {
    info.classList.add('hidden');
    empty.classList.remove('hidden');
    return;
  }

  empty.classList.add('hidden');
  info.classList.remove('hidden');
  count.textContent = scoredEntries.length;

  // Group by thread URL
  const groups = new Map();
  for (const entry of scoredEntries) {
    if (!groups.has(entry.url)) {
      groups.set(entry.url, { title: entry.title, url: entry.url, items: [] });
    }
    groups.get(entry.url).items.push(entry);
  }

  // Sort groups by their top item's score
  const sortedGroups = Array.from(groups.values())
    .map(g => {
      g.items.sort((a, b) => b.score - a.score);
      g.topScore = g.items[0].score;
      return g;
    })
    .sort((a, b) => b.topScore - a.topScore)
    .slice(0, MAX_GROUPS);

  for (const group of sortedGroups) {
    const [featured, ...rest] = group.items;

    // Group container
    const groupEl = document.createElement('div');
    groupEl.className = 'mb-5';

    // --- Thread header ---
    const header = document.createElement('a');
    header.href = group.url;
    header.target = '_blank';
    header.rel = 'noopener noreferrer';
    header.className = 'block mb-2';
    header.innerHTML = `
      <div class="text-brand-800 font-semibold text-lg hover:text-blue-600 transition-colors">
        ${escapeHtml(group.title)}
      </div>
    `;

    // --- Featured (top) entry ---
    const linkTo = featured.postUrl || featured.url;
    const main = document.createElement('a');
    main.href = linkTo;
    main.target = '_blank';
    main.rel = 'noopener noreferrer';
    main.className = 'result-card block mb-2';
    main.innerHTML = `
      <div class="flex items-start gap-2 mb-2">
        <p class="result-snippet flex-1">${escapeHtml(featured.text)}</p>
        <div class="flex-shrink-0 mt-0.5">${pointsBadgeHtml(featured.points)}</div>
      </div>
      <div class="result-meta flex items-center gap-2">
        <span class="result-link">View post</span>
        ${featured.date ? '<span>' + featured.date.split('T')[0] + '</span>' : ''}
        <span class="ml-auto">${(featured.score * 100).toFixed(0)}% match</span>
      </div>
    `;

    groupEl.appendChild(header);
    groupEl.appendChild(main);

    // --- Collapsible nested entries ---
    if (rest.length > 0) {
      const nestWrapper = document.createElement('div');
      nestWrapper.className = 'ml-5';

      // Toggle button
      const toggle = document.createElement('button');
      toggle.className = 'flex items-center gap-2 text-sm text-brand-400 hover:text-brand-600 cursor-pointer mb-2 w-full text-left py-1 px-3 rounded-lg hover:bg-brand-100 transition-colors';
      toggle.innerHTML = `<span>▼</span> <span>${rest.length} more post${rest.length > 1 ? 's' : ''} from this thread</span>`;
      toggle.setAttribute('aria-expanded', 'false');

      // Nested list (hidden by default)
      const nestList = document.createElement('div');
      nestList.className = 'border-l-2 border-brand-200 pl-4 space-y-2 hidden';

      for (const item of rest) {
        const itemLink = item.postUrl || item.url;
        const nested = document.createElement('a');
        nested.href = itemLink;
        nested.target = '_blank';
        nested.rel = 'noopener noreferrer';
        nested.className = 'result-card block !p-3';
        nested.innerHTML = `
          <div class="flex items-start gap-2 mb-1">
            <p class="text-sm text-brand-600 leading-relaxed flex-1">${escapeHtml(item.text)}</p>
            <div class="flex-shrink-0 mt-0.5">${pointsBadgeHtml(item.points)}</div>
          </div>
          <div class="text-xs text-brand-400 flex items-center gap-2">
            <span class="result-link">View post</span>
            <span>${(item.score * 100).toFixed(0)}% match</span>
          </div>
        `;
        nestList.appendChild(nested);
      }

      // Toggle click handler
      toggle.addEventListener('click', () => {
        const isHidden = nestList.classList.contains('hidden');
        nestList.classList.toggle('hidden');
        toggle.setAttribute('aria-expanded', !isHidden);
        toggle.innerHTML = isHidden
          ? `<span>▲</span> <span>Hide posts</span>`
          : `<span>▼</span> <span>${rest.length} more post${rest.length > 1 ? 's' : ''} from this thread</span>`;
      });

      nestWrapper.appendChild(toggle);
      nestWrapper.appendChild(nestList);
      groupEl.appendChild(nestWrapper);
    }

    list.appendChild(groupEl);
  }
}

function escapeHtml(str) {
  if (!str) return '';
  const div = document.createElement('div');
  div.textContent = str;
  return div.innerHTML;
}

// --- Search ---
async function performSearch(query) {
  if (!extractPipeline || !indexData.length || query.trim().length < MIN_QUERY_LENGTH) {
    renderResults([]);
    return;
  }

  try {
    const output = await extractPipeline(query, { pooling: 'mean', normalize: true });
    const queryVec = Array.from(output.data);

    const scored = indexData.map((chunk) => ({
      ...chunk,
      score: cosineSimilarity(queryVec, chunk.vector),
    }));

    scored.sort((a, b) => b.score - a.score);

    // Dedupe by postUrl+text, filter low scores
    const seen = new Set();
    const top = [];
    for (const r of scored) {
      const key = (r.postUrl || r.url) + '::' + r.text.substring(0, 80);
      if (seen.has(key)) continue;
      seen.add(key);

      if (r.score > 0.1 || top.length === 0) {
        top.push(r);
      }
    }

    renderResults(top);
  } catch (err) {
    console.error('Search error:', err);
  }
}

// --- Debounced search ---
let searchTimer = null;
input.addEventListener('input', () => {
  clearTimeout(searchTimer);
  searchTimer = setTimeout(() => performSearch(input.value), 200);
});

input.addEventListener('keydown', (e) => {
  if (e.key === 'Enter') {
    clearTimeout(searchTimer);
    performSearch(input.value);
  }
});

// --- Init ---
async function init() {
  try {
    setStatus(STATUS.LOADING);

    const resp = await fetch('./index.json');
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
    indexData = await resp.json();

    extractPipeline = await pipeline('feature-extraction', 'onnx-community/all-MiniLM-L6-v2-ONNX', {
      quantized: true,
    });

    input.disabled = false;
    input.placeholder = 'Type a question or topic...';
    input.focus();
    setStatus(STATUS.READY);

    empty.classList.remove('hidden');

  } catch (err) {
    console.error('Init error:', err);
    setStatus(STATUS.ERROR);
    input.placeholder = 'Search engine failed to load';
  }
}

init();
