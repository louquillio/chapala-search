#!/usr/bin/env node
/**
 * build-index.js — Forum scraper + vector index builder
 *
 * Crawls the Ajijic/Chapala/Guadalajara forum on chapala.com/webboard,
 * extracts qualified posts (>= 3 reactions), chunks them, generates
 * embeddings via all-MiniLM-L6-v2 (ONNX), and writes public/index.json.
 *
 * Usage:
 *   node scraper/build-index.js             # default: 3 pages
 *   node scraper/build-index.js --pages=20   # crawl 20 pages
 *   node scraper/build-index.js --pages=100  # deep historical crawl
 */

import * as fs from 'node:fs';
import * as path from 'node:path';
import { fileURLToPath } from 'node:url';
import * as cheerio from 'cheerio';
import { pipeline } from '@huggingface/transformers';

// --- Config ---
const FORUM_URL = 'https://chapala.com/webboard/index.php?/forum/1-ajijicchapalaguadalajara/';
const BASE_DOMAIN = 'https://chapala.com';
const INDEX_PATH = path.resolve(
  fileURLToPath(new URL('../public/index.json', import.meta.url))
);
const MAX_RETRIES = 3;
const RETRY_BASE_MS = 2000;
const MIN_REACTIONS = 1;
const MAX_CHUNK_CHARS = 400;
const REQUEST_TIMEOUT_MS = 15000;
const PACING_DELAY_MS = 800; // delay between topic fetches to avoid 429

// Parse --pages=N arg
const PAGES = parsePagesArg(process.argv);
const REBUILD = process.argv.includes('--rebuild');

function parsePagesArg(argv) {
  for (const arg of argv) {
    const m = arg.match(/^--pages=(\d+)$/);
    if (m) return parseInt(m[1], 10);
  }
  return 3; // default
}

// --- Helpers ---

function sleep(ms) {
  return new Promise((r) => setTimeout(r, ms));
}

/**
 * Fetch a URL with exponential backoff retry.
 * Returns the response text or null after exhausting retries.
 */
async function fetchWithRetry(url, attempt = 1) {
  try {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS);
    const resp = await fetch(url, { signal: controller.signal });
    clearTimeout(timer);

    if (!resp.ok) {
      // For 5xx errors, retry; for 4xx, bail immediately
      if (resp.status >= 500 && attempt < MAX_RETRIES) {
        const delay = RETRY_BASE_MS * Math.pow(2, attempt - 1);
        console.error(`[WARN] HTTP ${resp.status} on ${url}, retrying in ${delay}ms (attempt ${attempt}/${MAX_RETRIES})`);
        await sleep(delay);
        return fetchWithRetry(url, attempt + 1);
      }
      console.error(`[ERROR] HTTP ${resp.status} on ${url} — skipping`);
      return null;
    }
    return await resp.text();
  } catch (err) {
    if (attempt < MAX_RETRIES) {
      const delay = RETRY_BASE_MS * Math.pow(2, attempt - 1);
      console.error(`[WARN] Network error on ${url}: ${err.message}, retrying in ${delay}ms (attempt ${attempt}/${MAX_RETRIES})`);
      await sleep(delay);
      return fetchWithRetry(url, attempt + 1);
    }
    console.error(`[ERROR] Failed to fetch ${url} after ${MAX_RETRIES} attempts — skipping`);
    return null;
  }
}

/**
 * Resolve a relative URL against the base domain.
 */
function resolveUrl(href) {
  if (!href) return null;
  if (href.startsWith('http')) return href;
  if (href.startsWith('//')) return 'https:' + href;
  return BASE_DOMAIN + (href.startsWith('/') ? href : '/' + href);
}

// --- Topic list parsing ---

/**
 * Parse topic URLs from a forum listing page HTML.
 * Returns array of { url, title } objects.
 */
function parseTopicList(html) {
  const $ = cheerio.load(html);
  const topics = [];

  // Find all topic links within the listing
  $('a[href*="/topic/"]').each((i, el) => {
    const href = $(el).attr('href');
    const text = $(el).text().trim();

    // Skip pagination links (they contain /topic/ but with /page/ suffix)
    if (!href || !text) return;
    if (href.includes('/page/')) return;
    if (href.includes('#comments')) return;
    if (text.match(/^\d+$/)) return; // pagination numbers

    const url = resolveUrl(href);
    if (!url) return;

    // Dedupe by URL
    if (topics.some((t) => t.url === url)) return;

    // Skip pinned/announcement topics (badge with ipsBadge_positive)
    const heading = $(el).closest('h4.ipsDataItem_title');
    if (heading.find('.ipsBadge_positive').length > 0) {
      console.error(`[SKIP] Pinned topic: "${text}"`);
      return;
    }

    topics.push({ url, title: text });
  });

  return topics;
}

// --- Topic page parsing ---

/**
 * Parse a topic page to extract ALL qualifying posts and their reactions.
 *
 * Returns { title: string, posts: [{ commentId, reactions, date, text }] }
 * or null if no posts qualify.
 */
function parseTopicPage(html, fallbackTitle) {
  const $ = cheerio.load(html);

  // Get the title from the page heading
  const pageTitle =
    $('h1[class*="ipsType"]').first().text().trim() ||
    $('h1').first().text().trim() ||
    fallbackTitle;

  // Find all post containers
  const allPosts = $('.ipsComment');
  if (!allPosts.length) {
    console.error('[WARN] No post containers (.ipsComment) found');
    return null;
  }

  const qualifying = [];

  allPosts.each((i, el) => {
    const $post = $(el);

    // Extract comment ID from id="elComment_NNNNNN"
    const elId = $post.attr('id') || '';
    const idMatch = elId.match(/^elComment_(\d+)$/);
    if (!idMatch) return;
    const commentId = idMatch[1];

    // Calculate total reactions for this post
    let totalReactions = 0;
    $post.find('.ipsReact_reactCount').each((_, countEl) => {
      const val = parseInt($(countEl).text().trim(), 10);
      if (!isNaN(val)) totalReactions += val;
    });

    // Skip posts below the reaction threshold
    if (totalReactions < MIN_REACTIONS) return;

    // Extract post content
    const contentEl = $post.find('div[data-role="commentContent"]');
    if (!contentEl.length) return;

    const rawText = contentEl.text().trim();
    if (!rawText) return;

    // Extract date
    const dateText = $post.find('time').first().attr('datetime') ||
                     $post.find('time').first().text().trim() || '';

    qualifying.push({
      commentId,
      reactions: totalReactions,
      date: dateText,
      text: rawText,
    });
  });

  if (!qualifying.length) {
    console.error(`[SKIP] No qualifying posts in "${pageTitle}"`);
    return null;
  }

  console.log(`[INDEX] ${qualifying.length} qualifying post(s) in "${pageTitle}"`);
  for (const p of qualifying) {
    console.log(`       Post #${p.commentId}: ${p.reactions} pts — "${p.text.substring(0, 70)}..."`);
  }

  return {
    title: pageTitle,
    posts: qualifying,
  };
}

// --- Chunking ---

/**
 * Split post text into chunks for embedding.
 * - Split by double newlines first
 * - If any paragraph > 400 chars, split by sentence boundaries
 */
function chunkText(text) {
  const paragraphs = text.split(/\n\n+/).map((p) => p.trim()).filter(Boolean);
  const chunks = [];

  for (const para of paragraphs) {
    if (para.length <= MAX_CHUNK_CHARS) {
      chunks.push(para);
    } else {
      // Split by sentence boundaries
      const sentences = para.split(/(?<=[.!?])\s+/).filter(Boolean);
      let current = '';
      for (const sentence of sentences) {
        if ((current + ' ' + sentence).length > MAX_CHUNK_CHARS && current) {
          chunks.push(current.trim());
          current = sentence;
        } else {
          current += (current ? ' ' : '') + sentence;
        }
      }
      if (current.trim()) chunks.push(current.trim());
    }
  }

  return chunks;
}

// --- Embedding generation ---

let embedPipeline = null;

async function getEmbeddingPipeline() {
  if (!embedPipeline) {
    console.log('[INFO] Loading embedding model (all-MiniLM-L6-v2)...');
    embedPipeline = await pipeline('feature-extraction', 'onnx-community/all-MiniLM-L6-v2-ONNX', {
      quantized: true,
    });
    console.log('[INFO] Model loaded.');
  }
  return embedPipeline;
}

/**
 * Generate embedding vectors for an array of text chunks.
 * Returns array of { text, vector } objects.
 */
async function generateEmbeddings(chunks) {
  const pipe = await getEmbeddingPipeline();
  const results = [];

  for (const text of chunks) {
    try {
      const output = await pipe(text, { pooling: 'mean', normalize: true });
      const vector = Array.from(output.data);
      results.push({ text, vector });
    } catch (err) {
      console.error(`[WARN] Embedding failed for chunk: ${err.message}`);
      // Insert a zero vector as placeholder (won't match anything useful)
      results.push({ text, vector: null });
    }
  }

  return results;
}

// --- Index file I/O ---

/**
 * Load existing index data (append mode).
 */
function loadExistingIndex() {
  try {
    if (fs.existsSync(INDEX_PATH)) {
      const raw = fs.readFileSync(INDEX_PATH, 'utf-8');
      const data = JSON.parse(raw);
      if (Array.isArray(data)) return data;
    }
  } catch (err) {
    console.error(`[WARN] Could not load existing index: ${err.message}`);
  }
  return [];
}

/**
 * Save the complete index to file.
 * Vectors go to index.bin (Float32), metadata to index.json.
 */
function saveIndex(data) {
  const dir = path.dirname(INDEX_PATH);
  if (!fs.existsSync(dir)) fs.mkdirSync(dir, { recursive: true });

  // Write metadata (text, title, url, postUrl, date, points) — no vectors
  const meta = data.map(({ vector, ...rest }) => rest);
  fs.writeFileSync(INDEX_PATH, JSON.stringify(meta), 'utf-8');

  // Write vectors as flat Float32 binary
  const totalVectors = data.length;
  const dim = data[0]?.vector?.length || 0;
  const binPath = INDEX_PATH.replace(/\.json$/, '.bin');
  const buf = new Float32Array(totalVectors * dim);
  for (let i = 0; i < totalVectors; i++) {
    for (let j = 0; j < dim; j++) {
      buf[i * dim + j] = data[i].vector?.[j] || 0;
    }
  }
  fs.writeFileSync(binPath, Buffer.from(buf.buffer));

  const jsonSize = (fs.statSync(INDEX_PATH).size / 1024 / 1024).toFixed(1);
  const binSize = (fs.statSync(binPath).size / 1024 / 1024).toFixed(1);
  console.log(`[DONE] Written: ${INDEX_PATH} (${jsonSize} MB), ${binPath} (${binSize} MB) — ${totalVectors} entries`);
}

// --- Page iterator ---

/**
 * Build the URL for a given page of the topic listing.
 */
function getPageUrl(pageNum) {
  if (pageNum <= 1) return FORUM_URL;
  return FORUM_URL.replace(/\/?$/, '') + `/page/${pageNum}/`;
}

// --- Main ---

async function main() {
  console.log(`[START] Building index — crawling up to ${PAGES} page(s) of ${FORUM_URL}`);
  console.log(`[INDEX] ${INDEX_PATH}`);

  // Load existing index for deduplication (unless --rebuild)
  const existing = REBUILD ? [] : loadExistingIndex();
  const seenUrls = new Set(existing.map((e) => e.url));
  const seenPostUrls = new Set(existing.map((e) => e.postUrl).filter(Boolean));
  console.log(`[INFO] Existing index: ${existing.length} entries, ${seenUrls.size} unique threads`);

  // Collect unique topic URLs from the listing pages
  const allTopics = [];
  for (let page = 1; page <= PAGES; page++) {
    const pageUrl = getPageUrl(page);
    console.log(`\n[PAGE] ${page}/${PAGES} — ${pageUrl}`);
    const html = await fetchWithRetry(pageUrl);
    if (!html) {
      console.error(`[SKIP] Page ${page} failed to load`);
      continue;
    }

    const topics = parseTopicList(html);
    console.log(`[INFO] Found ${topics.length} topics on page ${page}`);

    for (const t of topics) {
      if (!seenUrls.has(t.url)) {
        allTopics.push(t);
        seenUrls.add(t.url);
      }
    }
  }

  console.log(`\n[CRAWL] ${allTopics.length} new topics to process`);

  // Process each topic
  let processed = 0;
  let indexed = 0;
  const newEntries = [];

  for (const topic of allTopics) {
    processed++;
    console.log(`\n[${processed}/${allTopics.length}] Fetching: ${topic.title}`);

    const html = await fetchWithRetry(topic.url);
    if (!html) {
      console.error(`[SKIP] Could not load topic: ${topic.title}`);
      continue;
    }

    const result = parseTopicPage(html, topic.title);
    if (!result || !result.posts.length) continue;

    // Process each qualifying post in this thread
    for (const post of result.posts) {
      const chunks = chunkText(post.text);
      const embedded = await generateEmbeddings(chunks);
      const postUrl = topic.url + '#findComment-' + post.commentId;

      for (const chunk of embedded) {
        if (!chunk.vector) continue; // skip failed embeddings
        if (seenPostUrls.has(postUrl)) continue; // already indexed this post

        newEntries.push({
          text: chunk.text,
          title: result.title,
          url: topic.url,
          postUrl: postUrl,
          date: post.date,
          points: post.reactions,
          vector: chunk.vector,
        });
        indexed++;
      }
    }

    // Pace requests to avoid 429 rate limiting
    if (processed < allTopics.length) {
      await sleep(PACING_DELAY_MS);
    }
  }

  // Merge with existing and save
  const merged = [...existing, ...newEntries];
  saveIndex(merged);

  console.log(`\n=== SUMMARY ===`);
  console.log(`  Pages crawled:   ${Math.min(PAGES, processed)}`);
  console.log(`  Topics fetched:  ${processed}`);
  console.log(`  Posts indexed:    ${new Set(newEntries.map((e) => e.postUrl)).size}`);
  console.log(`  Chunks added:    ${indexed}`);
  console.log(`  Total entries:   ${merged.length}`);
}

main().catch((err) => {
  console.error('[FATAL]', err);
  process.exit(1);
});
