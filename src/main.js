import { pipeline, cos_sim } from '@huggingface/transformers';

const STATUS = {
  LOADING: { dot: 'bg-amber-400 animate-pulse', text: 'Loading search engine...' },
  READY:   { dot: 'bg-green-500', text: 'Search engine ready' },
  ERROR:   { dot: 'bg-red-500', text: 'Failed to load — check console' },
};

const MAX_RESULTS = 20;
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

// --- Cosine similarity (redundant with cos_sim from transformers, but safe) ---
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

// --- Render results ---
function renderResults(results) {
  list.innerHTML = '';

  if (results.length === 0) {
    info.classList.add('hidden');
    empty.classList.remove('hidden');
    return;
  }

  empty.classList.add('hidden');
  info.classList.remove('hidden');
  count.textContent = results.length;

  for (const r of results) {
    const card = document.createElement('a');
    card.href = r.url;
    card.target = '_blank';
    card.rel = 'noopener noreferrer';
    card.className = 'result-card block';

    card.innerHTML = `
      <p class="result-snippet">${escapeHtml(r.text)}</p>
      <div class="result-title">${escapeHtml(r.title)}</div>
      <div class="result-meta">
        ${r.date ? r.date.split('T')[0] : ''}
        &middot; score: ${(r.score * 100).toFixed(0)}%
      </div>
    `;

    list.appendChild(card);
  }
}

function escapeHtml(str) {
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
    // Compute query embedding
    const output = await extractPipeline(query, { pooling: 'mean', normalize: true });
    const queryVec = Array.from(output.data);

    // Score all chunks
    const scored = indexData.map((chunk) => ({
      ...chunk,
      score: cosineSimilarity(queryVec, chunk.vector),
    }));

    scored.sort((a, b) => b.score - a.score);

    // Filter low scores and dedupe by URL+text
    const seen = new Set();
    const top = [];
    for (const r of scored) {
      const key = r.url + '::' + r.text.substring(0, 80);
      if (seen.has(key)) continue;
      seen.add(key);

      if (r.score > 0.1 || top.length === 0) {
        top.push(r);
        if (top.length >= MAX_RESULTS) break;
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

    // Load index data
    const resp = await fetch('./index.json');
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
    indexData = await resp.json();

    // Initialize pipeline
    extractPipeline = await pipeline('feature-extraction', 'onnx-community/all-MiniLM-L6-v2-ONNX', {
      quantized: true,
    });

    // Enable search
    input.disabled = false;
    input.placeholder = 'Type a question or topic...';
    input.focus();
    setStatus(STATUS.READY);

    // Show empty state hint
    empty.classList.remove('hidden');

  } catch (err) {
    console.error('Init error:', err);
    setStatus(STATUS.ERROR);
    input.placeholder = 'Search engine failed to load';
  }
}

init();
