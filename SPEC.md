## Custom AI-search for chapala.com/webboard (Vite + Tailwind Architecture)

You are an expert systems engineer specializing in zero-backend, client-side vector search architectures, automated web scrapers, and highly resilient local batch execution pipelines. Generate a production-ready, self-recovering system using Vite, Tailwind CSS, Node.js, and GitHub Pages based on the precise specification below. Do not include introductory text or placeholders.

### Architecture & Constraints
1. NO BACKEND SERVER or active database for the web search interface. The search application must run entirely client-side using a static JSON index file containing text chunks and vector coordinates.
2. NO FORUM API ACCESS or database dumps. Data collection must look like generic public web browsing using standard CSS selector extraction.
3. The target site is https://chapala.com/webboard/, specifically focusing on the "Ajijic/Chapala/Guadalajara" sub-forum.
4. INFRASTRUCTURE & EXECUTION BOUNDARIES:
   - The public-facing web interface must be hosted for free using GitHub Pages.
   - The scraping, chunking, and compilation pipeline runs entirely headlessly via cron on a local Linux workstation.
   - Vector generation compute uses a remote, cloud-hosted DeepSeek Embeddings API endpoint, authenticated via a local environment variable (`DEEPSEEK_API_KEY`) stored on the execution workstation.
5. OPERATIONAL RESILIENCE: The local pipeline must use defensive programming to ensure it can run headlessly for years without manual intervention, self-recovering from transient network drops or gateway timeouts.

### Repository File Structure
The project tree must strictly follow this layout:
- /package.json
- /vite.config.js
- /tailwind.config.js
- /postcss.config.js
- /README.md
- /scraper/build-index.js
- /scraper/run-pipeline.sh
- /src/index.html
- /src/main.js
- /docs/ (Vite production build target folder tracked by Git)

### Component 1: Local Ingestion & Vector Generator (`scraper/build-index.js`)
Create a robust Node.js command-line application executing on the local workstation.
- Arguments: Accept a `--pages=X` flag to restrict how deep the script crawls the topic list index to prevent unintentional rate-limiting.
- Selector Targets for Invision Community:
  - Topic List Row: `.ipsDataItem`
  - Post Container: `.ipsComment`
  - Reaction Trigger: Locate the reaction total wrapper (`.ipsReact_count`). Safely parse the text content to an integer. If the cumulative sum of reactions ('Thanks' or 'Likes') is >= 3, proceed with parsing the thread. If the element is missing or below 3, immediately skip the thread.
  - Text Content: `div[data-role="commentContent"]`
- Content Parsing & Chunking:
  - If a thread qualifies, extract Title, URL, Date, and Raw Text from the first post. Clean out raw HTML tags.
  - Chunking Engine: Split the post text by double newlines (`\n\n`). Fallback: If any single paragraph exceeds 400 characters, break it into smaller logical blocks by sentence punctuation boundaries (`.`, `!`, `?`).
- External Cloud API Vector Pipeline:
  - Read the `DEEPSEEK_API_KEY` environment variable from the local system execution environment.
  - Pass the text chunks to the remote DeepSeek cloud embeddings API endpoint to generate vector coordinates.
- Error Handling & Self-Recovery:
  - Wrap network requests in an exponential backoff retry loop (up to 3 attempts per page).
  - If a page permanently fails to load due to a bad gateway or gateway timeout, log the error clearly to `stderr`, skip that page, and continue processing the queue rather than crashing the execution loop.
- Output: Append or overwrite the verified metadata and vector payloads directly into the staging data path for deployment.

### Component 2: Frontend SPA Client (`/src/index.html` & `/src/main.js`)
Build a highly responsive, clean search tool using a reserved layout compiled via Vite.
- UI Framework: Integrate Tailwind CSS utility classes. Design a conservative, scannable layout optimized for a senior audience: clear typographic hierarchies, a high-contrast centered search box, a model status loader, and a clean vertical results list. Avoid vibrant accent loops or glowing/neon design trends.
- Browser Machine Learning Engine:
  - Import the standard modern production ESM package: `https://cdn.jsdelivr.net/npm/@huggingface/transformers`
  - Initialize a pipeline for `feature-extraction` using a compact, performant browser model: `onnx-community/all-MiniLM-L6-v2-ONNX`. Ensure it handles multi-language search context mapping gracefully.
- Runtime Execution Flow:
  - On first page load, keep the search input `disabled` with a placeholder reading "Loading search engine...".
  - Fetch and cache the static model files alongside the static `index.json` data array into the browser memory.
  - Once promises resolve, change the placeholder to "Type a question or topic..." and enable the input.
  - Search Execution: When a user types a plain language query, run the text through the pipeline to extract the query vector.
  - Comparison Math: Compute the cosine similarity score between the query vector and every vector item inside the loaded `index.json` dataset.
  - Rendering Engine: Sort matches in descending order. Display the exact matching contextual snippet, the original thread title, and a direct hyperlink pointing to the source forum thread URL. Render results immediately as the user types or hits enter.

### Component 3: PM2 & Git Automation Loop (`scraper/run-pipeline.sh`)
Write a complete production bash script to manage automated weekly execution via cron on the local Linux workstation.
1. Force execution tracking via PM2 (Process Manager 2) to ensure runtime process visibility and basic error logging.
2. Run the compilation pipeline (`node scraper/build-index.js --pages=3`) to append the latest curated data straight into the data file path.
3. Execute the Vite production build (`npm run build`) to compile the site assets cleanly into the public `/docs` output directory.
4. Verify if changes exist via Git tracking (`git diff --exit-code docs/index.json`). If modifications are found, automatically execute a headless commit and push the updated `/docs` block directly to the `main` GitHub repository branch.
5. Notification Dead-Man's Switch: Include an optional curl block at the very end of a verified successful run to ping an external monitoring endpoint (e.g., Healthchecks.io). If the script fails or stalls, the missing ping will alert the developer.
6. Provide the exact Linux crontab syntax configuration to safely trigger this execution script once every Sunday at midnight without interactive terminal overhead.

### Component 4: Repository Documentation (`README.md`)
Generate a clean, professional markdown file documenting the project lifecycle for maintenance.
- Project Summary: Explain the hybrid local-ingestion/cloud-embedding/client-side vector search architecture pattern.
- Requirements: List dependencies including Node.js LTS, Tailwind, Vite, a cloud-hosted DeepSeek API key set as an environment variable, and PM2.
- Local Development & Testing: Document how to install dependencies, how to execute a deep historical crawl manually using `node scraper/build-index.js --pages=100`, and how to start the local Vite development server (`npm run dev`) to test changes and styles locally before pushing upstream.
- Automation Instructions: Detail how to configure the environment variables for the DeepSeek key, initialize the PM2 process tracker, and add the cron entry for hands-off long-running deployment.

Use the initialized (but empty) repo at `~/projects/chapala-search`.

