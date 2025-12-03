<!-- 4ed6791f-acdc-4ad1-a87f-6fc7985d72bf 4ef4e25c-262f-4e48-9a2e-5542cc5ffdca -->
# TOON Proposal Notebook Restructure

## New Notebook Structure

### Part 1: Introduction (Keep Existing)

- Keep the existing executive summary and everything **above** the table of contents unchanged.
- Update the **Table of Contents** to reflect the new schema-focused structure below.
- Add one sentence at the end of the intro framing the rest of the notebook as:

> “The core of this notebook now evaluates TOON on three Hebbia-style schemas: Matrix strip-profile rows, ISD citation objects, and Agent 2.0 context history, plus one end-to-end research turn.”

---

### Part 2: Setup & Configuration

- Imports and dependencies.
- **Configurable scale parameters** exposed at the top, used consistently in all benchmarks:
```python
SCALE_CONFIG = {
    "matrix_rows": 1000,      # Can be bumped to 5000+ for Hebbia-realistic scale
    "isd_citations": 5000,    # Can be bumped to 17500+ for Hebbia-realistic scale
    "context_messages": 100,  # Can be bumped to 200+ for multi-turn sessions
}
```

- Helper to compute “Hebbia-realistic” presets (e.g., a small dict or function that multiplies these values).
- OpenAI API key detection with **graceful fallback**:

  - If `OPENAI_API_KEY` not set → print a clear note: “Live accuracy tests will be skipped; all token benchmarks still run.”
  - All benchmark code must run without any network calls.

---

### Part 3: The 3 Hebbia-Style Schemas

Short markdown section that:

1. States the goal:

> “We approximate three internal data shapes that Hebbia publicly describes in its blogs: Matrix strip-profile rows, ISD citation objects, and Agent 2.0 context history.”

2. Introduces each schema explicitly:

   1. **`MatrixStripProfileRow` (Matrix strip-profile rows)**

      - Scenario: LevFin / equity workflows where analysts compare issuers on revenue, margins, leverage, etc., in a Matrix grid with source-linked cells.
      - Fields (described, not full JSON):

        - `row_id`, `company_name`, `ticker`, `sector`
        - `fiscal_year`, `fiscal_quarter`, `currency`
        - `metrics` (e.g., `revenue`, `ebitda`, `ebitda_margin`, `net_debt`)
        - `source` (doc id, section path, page, citation ids)

   1. **`ISDCitation` (ISD citation objects)**

      - Scenario: ISD line-level citations for clauses, risk factors, and snippets from credit agreements, 10-Ks, PIBs, etc.
      - Fields:

        - `citation_id`, `doc_id`, `doc_type`, `symbol`, `fiscal_year`
        - `section_path`, `page` or transcript position
        - `span` (`start_char`, `end_char`)
        - `snippet` (1–2 sentences)
        - `labels` / `tags`, `relevance_score`, `confidence`

   1. **`AgentTurnContext` (Agent 2.0 context history)**

      - Scenario: Agent 2.0 orchestrator + subagents + tools working on a single Matrix or Chat turn.
      - Fields:

        - `turn_id`, `session_id`, `product`, `entry_point`
        - `message_history`: list of messages with `message_id`, `timestamp`, `agent`, `role`, `content`, `tool_name`, `tool_args`, `visible_to`
        - `metrics`: latency, model family, token counts
        - `trace`: observability IDs (e.g., `datadog_trace_id`, `maximizer_request_id`, `partitions`)

3. For each schema, show:

   - A **compact JSON example** (1–2 rows / citations / messages) in a code block.
   - A short note: “We will later show the analogous TOON encoding with shared headers.”

No benchmarking yet in this part: just shape definition and narrative context.

---

### Part 4: Benchmark 1 – Matrix Strip-Profile Rows

**Section title suggestion:**

> “Benchmark 1: Matrix strip-profile rows (LevFin ‘who’s most profitable?’ grid)”

Content:

- Implement `generate_matrix_rows(num_companies, num_metrics, quarters, scale_cfg)` in `toon_benchmark.py`:

  - Generates a list of `MatrixStripProfileRow` objects.
  - Uses realistic tickers (`AAPL`, `MSFT`, `GOOGL`, `AMZN`, `META`, `NVDA`, etc.), plausible fiscal periods, and financial metrics.

- **Micro-benchmark (structure only)**

  - Run `compare_formats` on:

    - Single row.
    - `SCALE_CONFIG["matrix_rows"] // 10` rows.
    - `SCALE_CONFIG["matrix_rows"]` rows.

  - Display a small table:

| Rows | JSON tokens | TOON tokens | Savings % |

| ---- | ----------: | ----------: | --------: |

  - One bar chart (JSON vs TOON) only; keep it minimal.

- **Prompt-level benchmark (OutputAgent-style)**

  - Scenario line in markdown:

> “A Matrix OutputAgent is asked: ‘Which of the big tech giants has the highest EBITDA margin and by how many points do they exceed the peer median? Cite your sources.’”

  - Build two prompts:

    - JSON prompt: question + strip-profile context encoded as JSON.
    - TOON prompt: same content encoded as TOON.
  - Use `compare_prompt_tokens` to compute token counts and savings.
  - Show a table summarizing the prompt-level savings.

---

### Part 5: Benchmark 2 – ISD Citation Objects

**Section title suggestion:**

> “Benchmark 2: ISD citation objects (credit agreements & PIB footnotes)”

Content:

- Implement `generate_isd_citations(num_docs, citations_per_doc, scale_cfg)`:

  - Produces `num_docs * citations_per_doc ≈ SCALE_CONFIG["isd_citations"]` `ISDCitation` objects.
  - Mixes doc types (`10-K`, `10-Q`, credit agreements, PIBs), sectors, and section paths.

- **Token benchmark (structure only)**

  - Run `run_benchmark` on the full citation list.

  - Show:

| Docs | Citations | JSON tokens | TOON tokens | Savings % |

| ---- | --------: | ----------: | ----------: | --------: |

  - One simple visualization: histogram of per-citation savings or a bar of total tokens—pick one, not both.

- **Prompt-level benchmark (risk section authoring)**

  - Scenario line:

> “A research / Deeper-style OutputAgent is asked to write a ‘Cloud concentration risk’ paragraph, citing relevant clauses across all loaded agreements.”

  - Build prompts:

    - JSON: citations as a JSON array.
    - TOON: citations in TOON form (`citations[...]` block).
  - Use `compare_prompt_tokens` to show savings at the prompt level.

---

### Part 6: Benchmark 3 – Agent Context History

**Section title suggestion:**

> “Benchmark 3: Agent 2.0 context history (multi-agent LevFin turn)”

Content:

- Implement `generate_context_history(messages_per_turn, scale_cfg)`:

  - Creates a single `AgentTurnContext` object:

    - ~`SCALE_CONFIG["context_messages"]` messages with a mix of:

      - User messages.
      - Orchestrator reasoning.
      - Subagent tool calls (`read_matrix`, `search_isd`, etc.).
      - OutputAgent final answer.

- **Token benchmark (message_history + full context)**

  - Compare JSON vs TOON for:

    - `context["message_history"]` only.
    - Full context object.
  - Show a table:

| Messages | JSON tokens | TOON tokens | Savings % |

| -------- | ----------: | ----------: | --------: |

- **Prompt-level benchmark (observability / debugging)**

  - Scenario line:

> “An internal observability assistant is asked: ‘Given this Agent 2.0 context, explain why this turn took more than 4 seconds and suggest one optimization.’”

  - Build JSON vs TOON prompts embedding the context.
  - Measure token savings with `compare_prompt_tokens`.

---

### Part 7: End-to-End Combined Benchmark

**Section title suggestion:**

> “Benchmark 4: End-to-end Agent 2.0 → Matrix → ISD research turn”

Composite payload simulating a single realistic research / LevFin turn:

- Scenario narrative:

> “A LevFin associate asks: ‘Benchmark covenant tightness and EBITDA margins for the top 10 SaaS issuers vs 2024 deals, and summarize where our client is off-market.’”

- Use generators with `SCALE_CONFIG`:

  - ~200 `MatrixStripProfileRow` entries (subset of `matrix_rows`).
  - `SCALE_CONFIG["isd_citations"]` `ISDCitation` objects backing those rows (~5,000 by default).
  - One `AgentTurnContext` with `SCALE_CONFIG["context_messages"]` messages (~100).

- Implement:

  - `assemble_json_payload(question, matrix_rows, citations, context)`
  - `assemble_toon_payload(question, matrix_rows, citations, context)`

- Compute token counts per component and overall:

| Component             | JSON tokens | TOON tokens | Savings % |

| --------------------- | ----------: | ----------: | --------: |

| Matrix rows           |           … |           … |       … % |

| ISD citations         |           … |           … |       … % |

| Agent context history |           … |           … |       … % |

| **Total payload**     |       **…** |       **…** |   **… %** |

- Optional: a single stacked bar chart showing total JSON vs TOON tokens for the combined payload.

---

### Part 8: Live LLM Accuracy Validation (Optional)

- Entire section guarded by `if OPENAI_API_KEY:`.
- Clear printed message if skipped:

> “Skipping live accuracy tests (no OPENAI_API_KEY set). All token benchmarks above remain valid.”

For each schema (3 total):

1. Build a minimal JSON vs TOON example and a **single** question:

   - Matrix: “What is EBITDA margin for NVDA in Q3 2025?”
   - ISD: “Extract the exact covenant threshold for net leverage from this citation list.”
   - Context: “What tool did the ReadMatrix agent call first in this turn?”

2. Make **one JSON call** and **one TOON call** to the same model (e.g., `gpt-4o`):

   - Compare answers by simple string/number normalization (no heavy eval metrics).
   - Print both outputs and a short verdict: “Matched / Mismatched”.

This section is proof-of-plausibility, not a full eval suite.

---

### Part 9: Hebbia-Scale Cost Analysis (Adapted from Existing)

- Reuse existing cost model, but parameterize it using observed savings from Parts 4–7:

  - `observed_savings_matrix`, `observed_savings_isd`, `observed_savings_context`, and `observed_savings_combined`.
- Inputs:

  - Calls per month (e.g., 250B baseline, editable).
  - Average tokens per call.
  - Input/output pricing per million tokens for multiple vendors (OpenAI / Anthropic / Gemini).
- Outputs:

  - Annual cost with JSON.
  - Annual cost with TOON (using observed savings).
  - Delta (`$` and `%`) per vendor.

Keep visualizations modest: one table + one bar plot of “Annual cost JSON vs TOON” per vendor.

---

### Part 10: Context Compression & Caching (Adapted from Existing)

- Short markdown section explaining how TOON composes with Hebbia’s existing ideas:

  - **Context distillation**:

    - TOON makes raw structured payloads cheaper *before* distillation and still useful when you can’t fully distill (e.g., full strip table + citations needed).

  - **Maximizer / license requests**:

    - Maximizer optimizes **how many** LLM calls you can run under rate limits.
    - TOON reduces **how large** each call is in tokens.

- Re-use / lightly trim existing analysis of:

  - Context distillation / compression strategies.
  - Cache policies (LRU/LFU/FIFO) for pre-TOON vs post-TOON payloads.

- Explicitly frame the conclusion as:

> “TOON doesn’t replace your compression stack or Maximizer; it gives them better raw material.”

---

### Part 11: Q&A – Anticipated Objections (Adapted from Existing)

- Keep existing Q&A but reorder / edit to align with the 3 schemas and end-to-end benchmark:

  - “Does accuracy degrade?” → point to Part 8.
  - “What about parsing overhead?” → note encoding cost vs LLM inference time.
  - “Does this fit our schema?” → reference the three Hebbia-style schemas.
  - “How does this interact with Agent 2.0 and Matrix?” → reference Parts 4–7.

- Optionally add 1–2 new questions:

  - “What if some products use a proprietary internal schema?”

    - Answer: TOON can encode that schema directly; the notebook just uses public approximations.
  - “Is this worth it if we already have aggressive context distillation?”

    - Answer: even a small % improvement at Hebbia’s scale is meaningful; see Part 9.

---

### Part 12: Implementation Roadmap & Conclusion

- 4-phase rollout (keep your existing structure, but tie phases to schemas):

  1. **Phase 1 – Instrumentation**

     - Add token logging for existing JSON payloads for the three schema types.

  1. **Phase 2 – Shadow TOON Encoding**

     - Implement TOON encoders for internal equivalents of `MatrixStripProfileRow`, `ISDCitation`, `AgentTurnContext`.
     - Run A/B shadow tests on a small slice of traffic (no user-visible change).

  1. **Phase 3 – Partial Rollout**

     - Enable TOON for the highest-volume / easiest schemas first (likely citations and Matrix rows).
     - Integrate with Maximizer awareness of reduced payload size where applicable.

  1. **Phase 4 – Full Rollout & Monitoring**

     - TOON as default for structured payloads in prompts.
     - Dashboard measuring real token savings vs notebook projections.

- Success metrics table:

  - Token reduction per schema.
  - $ savings at current traffic.
  - Error / incident rate (should remain flat).

- Close with a concise conclusion:

  - Restate observed savings range.
  - Emphasize that the schemas, benchmarks, and end-to-end turn are all approximations of Hebbia’s *actual* workloads, not generic JSON.

---

## Files to Modify

### `experiments/toon_benchmark.py`

Add or extend:

- `generate_matrix_rows(num_companies, num_metrics, quarters, scale_cfg)`
- `generate_isd_citations(num_docs, citations_per_doc, scale_cfg)`
- `generate_context_history(messages_per_turn, scale_cfg)`
- `assemble_json_payload(question, matrix_rows, citations, context)`
- `assemble_toon_payload(question, matrix_rows, citations, context)`
- Helper functions for:

  - Token counting per component.
  - Simple normalization for live accuracy checks (numeric/string compare).

### `notebooks/toon_proposal_hebbia.ipynb`

- Restructure according to Parts 1–12 above.
- Aim for ~25–30 cells:

  - ~10 cells for benchmarks (code + plots).
  - ~10 cells for narrative/markdown.
  - Remaining for configuration, cost model, and Q&A.

---

## Key Implementation Details

1. **Synthetic data realism**

   - Use plausible tickers, fiscal periods, and magnitudes for metrics.
   - Make `section_path` strings and snippets look like real risk / covenant / MD&A text.

2. **TOON encoding for nested structures**

   - For `metrics`, `source`, `span`, `labels`, and `tool_args`, design headers that avoid excessive nesting but keep it readable.
   - Where necessary, serialize deeply nested dicts as JSON strings inside TOON (e.g., `tool_args_json`) to keep the encoder simple.

3. **Graceful API fallback**

   - All token-counting and benchmark sections must run without any external API.
   - Live accuracy tests are clearly labeled as “optional” and fully skippable if `OPENAI_API_KEY` is missing.

4. **Configurable scale**

   - All generators and benchmarks must read from `SCALE_CONFIG`.
   - Optionally, add a single helper cell to switch between:

     - “Demo scale” (fast, default).
     - “Hebbia-like scale” (larger values, slower but still notebook-friendly).

### To-dos

- [ ] Add synthetic data generators to toon_benchmark.py for 3 schemas
- [ ] Create markdown section introducing the 3 Hebbia-style schemas
- [ ] Implement Benchmark 1: Matrix Strip-Profile Rows
- [ ] Implement Benchmark 2: ISD Citation Objects
- [ ] Implement Benchmark 3: Agent Context History
- [ ] Implement End-to-End combined benchmark
- [ ] Add optional live LLM accuracy validation section
- [ ] Integrate existing cost model, caching, Q&A sections