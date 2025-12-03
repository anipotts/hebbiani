"""
TOON vs JSON Benchmark Utilities

Provides reusable functions to measure token savings and accuracy when
using TOON (Token-Oriented Object Notation) versus JSON for LLM prompts.

Reference: https://github.com/toon-format/toon
"""

import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import tiktoken


# ============================================================================
# TOON ENCODER (Pure Python implementation)
# ============================================================================
# The toon_format pip package may not be available, so we implement core
# encoding logic based on the TOON spec: https://toonformat.dev


def _escape_toon_value(value: str) -> str:
    """Escape special characters in TOON values."""
    if value is None:
        return ""
    s = str(value)
    # Escape backslashes first, then quotes and newlines
    s = s.replace("\\", "\\\\")
    s = s.replace('"', '\\"')
    s = s.replace("\n", "\\n")
    s = s.replace("\r", "\\r")
    s = s.replace("\t", "\\t")
    # If value contains comma, wrap in quotes
    if "," in s or s != s.strip():
        return f'"{s}"'
    return s


def _format_value(value: Any) -> str:
    """Format a single value for TOON output."""
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, str):
        return _escape_toon_value(value)
    if isinstance(value, (list, dict)):
        # Nested structures get JSON-encoded inline
        return _escape_toon_value(json.dumps(value))
    return _escape_toon_value(str(value))


def encode_toon(data: Any, delimiter: str = ",") -> str:
    """
    Encode data to TOON format.

    TOON format for arrays of objects:
    ```
    arrayName[count]{field1,field2,field3}:
      value1,value2,value3
      value4,value5,value6
    ```

    For single objects:
    ```
    key1: value1
    key2: value2
    ```
    """
    if data is None:
        return ""

    if isinstance(data, list):
        if not data:
            return "[]"

        # Check if it's an array of objects with consistent keys
        if all(isinstance(item, dict) for item in data):
            # Get all unique keys in order of first appearance
            keys = []
            for item in data:
                for k in item.keys():
                    if k not in keys:
                        keys.append(k)

            # Build header
            header = f"[{len(data)}]{{{delimiter.join(keys)}}}:"

            # Build rows
            rows = []
            for item in data:
                values = [_format_value(item.get(k)) for k in keys]
                rows.append(f"  {delimiter.join(values)}")

            return header + "\n" + "\n".join(rows)
        else:
            # Array of primitives
            values = [_format_value(v) for v in data]
            return f"[{len(data)}]: {delimiter.join(values)}"

    if isinstance(data, dict):
        lines = []
        for key, value in data.items():
            if (
                isinstance(value, list)
                and value
                and all(isinstance(v, dict) for v in value)
            ):
                # Nested array of objects
                nested = encode_toon(value, delimiter)
                lines.append(f"{key}{nested}")
            elif isinstance(value, dict):
                # Nested object - inline or expand
                nested = encode_toon(value, delimiter)
                lines.append(f"{key}:\n{_indent(nested, 2)}")
            else:
                lines.append(f"{key}: {_format_value(value)}")
        return "\n".join(lines)

    return _format_value(data)


def _indent(text: str, spaces: int) -> str:
    """Indent each line of text."""
    prefix = " " * spaces
    return "\n".join(prefix + line for line in text.split("\n"))


def encode_toon_for_prompt(data: Dict[str, Any], title: str = "data") -> str:
    """
    Encode data for inclusion in an LLM prompt.

    Wraps TOON output in a code block with the 'toon' language tag.
    """
    toon_content = encode_toon(data)
    return f"```toon\n{title}:\n{_indent(toon_content, 2)}\n```"


# ============================================================================
# TOKEN COUNTING
# ============================================================================


def count_tokens(text: str, model: str = "gpt-4o") -> int:
    """
    Count tokens in text using tiktoken.

    Args:
        text: The text to tokenize
        model: The model to use for tokenization (default: gpt-4o)

    Returns:
        Number of tokens
    """
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        # Fall back to cl100k_base for unknown models
        encoding = tiktoken.get_encoding("cl100k_base")

    return len(encoding.encode(text))


def count_tokens_batch(texts: List[str], model: str = "gpt-4o") -> List[int]:
    """Count tokens for multiple texts efficiently."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")

    return [len(encoding.encode(text)) for text in texts]


# ============================================================================
# BENCHMARK DATA STRUCTURES
# ============================================================================


@dataclass
class TokenComparison:
    """Results of comparing JSON vs TOON token usage."""

    json_text: str
    toon_text: str
    json_tokens: int
    toon_tokens: int
    savings_absolute: int
    savings_percent: float
    data_hash: str  # For verification


@dataclass
class BenchmarkResult:
    """Aggregate results across multiple data samples."""

    sample_count: int
    total_json_tokens: int
    total_toon_tokens: int
    total_savings: int
    avg_savings_percent: float
    min_savings_percent: float
    max_savings_percent: float
    comparisons: List[TokenComparison]


# ============================================================================
# BENCHMARK FUNCTIONS
# ============================================================================


def compare_formats(
    data: Any, model: str = "gpt-4o", json_indent: Optional[int] = 2
) -> TokenComparison:
    """
    Compare token usage between JSON and TOON for a data structure.

    Args:
        data: The data to encode
        model: Model for tokenization
        json_indent: Indentation for JSON (None for compact)

    Returns:
        TokenComparison with detailed metrics
    """
    # Encode to JSON
    json_text = json.dumps(data, indent=json_indent, ensure_ascii=False)

    # Encode to TOON
    toon_text = encode_toon(data)

    # Count tokens
    json_tokens = count_tokens(json_text, model)
    toon_tokens = count_tokens(toon_text, model)

    # Calculate savings
    savings_absolute = json_tokens - toon_tokens
    savings_percent = (savings_absolute / json_tokens * 100) if json_tokens > 0 else 0

    # Create hash for verification
    data_hash = str(hash(json.dumps(data, sort_keys=True)))[:8]

    return TokenComparison(
        json_text=json_text,
        toon_text=toon_text,
        json_tokens=json_tokens,
        toon_tokens=toon_tokens,
        savings_absolute=savings_absolute,
        savings_percent=savings_percent,
        data_hash=data_hash,
    )


def run_benchmark(
    samples: List[Any], model: str = "gpt-4o", json_indent: Optional[int] = 2
) -> BenchmarkResult:
    """
    Run benchmark across multiple data samples.

    Args:
        samples: List of data structures to benchmark
        model: Model for tokenization
        json_indent: Indentation for JSON

    Returns:
        BenchmarkResult with aggregate statistics
    """
    comparisons = [compare_formats(s, model, json_indent) for s in samples]

    total_json = sum(c.json_tokens for c in comparisons)
    total_toon = sum(c.toon_tokens for c in comparisons)
    savings_percents = [c.savings_percent for c in comparisons]

    return BenchmarkResult(
        sample_count=len(samples),
        total_json_tokens=total_json,
        total_toon_tokens=total_toon,
        total_savings=total_json - total_toon,
        avg_savings_percent=(
            sum(savings_percents) / len(savings_percents) if savings_percents else 0
        ),
        min_savings_percent=min(savings_percents) if savings_percents else 0,
        max_savings_percent=max(savings_percents) if savings_percents else 0,
        comparisons=comparisons,
    )


# ============================================================================
# HEBBIA-SPECIFIC UTILITIES
# ============================================================================


def calculate_hebbia_savings(
    savings_percent: float,
    calls_per_month: int = 250_000_000_000,  # 250B from notes
    avg_tokens_per_call: int = 500,
    input_cost_per_million: float = 3.0,  # GPT-4o pricing
    output_cost_per_million: float = 15.0,
) -> Dict[str, float]:
    """
    Calculate estimated annual savings for Hebbia-scale operations.

    Based on George Sivulka's statement: "250B LLM calls/month"

    Returns dict with:
        - monthly_token_savings
        - monthly_cost_savings
        - annual_cost_savings
    """
    # Calculate baseline token usage
    monthly_tokens = calls_per_month * avg_tokens_per_call

    # Assume 80% input, 20% output (typical for RAG/retrieval)
    monthly_input_tokens = monthly_tokens * 0.8
    monthly_output_tokens = monthly_tokens * 0.2

    # Calculate baseline cost
    monthly_input_cost = (monthly_input_tokens / 1_000_000) * input_cost_per_million
    monthly_output_cost = (monthly_output_tokens / 1_000_000) * output_cost_per_million
    monthly_total_cost = monthly_input_cost + monthly_output_cost

    # Calculate savings (TOON only affects input tokens)
    monthly_token_savings = monthly_input_tokens * (savings_percent / 100)
    monthly_cost_savings = (monthly_token_savings / 1_000_000) * input_cost_per_million
    annual_cost_savings = monthly_cost_savings * 12

    return {
        "monthly_baseline_tokens": monthly_tokens,
        "monthly_baseline_cost": monthly_total_cost,
        "monthly_token_savings": monthly_token_savings,
        "monthly_cost_savings": monthly_cost_savings,
        "annual_cost_savings": annual_cost_savings,
        "savings_percent_applied": savings_percent,
    }


def chunk_transcript(
    transcript: str, chunk_size: int = 2000, overlap: int = 200
) -> List[Dict[str, Any]]:
    """
    Chunk a transcript into overlapping segments with metadata.

    Returns list of chunk objects suitable for TOON benchmarking.
    """
    chunks = []
    sentences = re.split(r"(?<=[.!?])\s+", transcript)

    current_chunk = []
    current_length = 0
    chunk_idx = 0

    for sentence in sentences:
        sentence_len = len(sentence)

        if current_length + sentence_len > chunk_size and current_chunk:
            # Save current chunk
            chunk_text = " ".join(current_chunk)
            chunks.append(
                {
                    "chunk_id": chunk_idx,
                    "text": chunk_text,
                    "char_count": len(chunk_text),
                    "sentence_count": len(current_chunk),
                }
            )
            chunk_idx += 1

            # Keep overlap
            overlap_sentences = []
            overlap_len = 0
            for s in reversed(current_chunk):
                if overlap_len + len(s) <= overlap:
                    overlap_sentences.insert(0, s)
                    overlap_len += len(s)
                else:
                    break

            current_chunk = overlap_sentences
            current_length = overlap_len

        current_chunk.append(sentence)
        current_length += sentence_len

    # Don't forget the last chunk
    if current_chunk:
        chunk_text = " ".join(current_chunk)
        chunks.append(
            {
                "chunk_id": chunk_idx,
                "text": chunk_text,
                "char_count": len(chunk_text),
                "sentence_count": len(current_chunk),
            }
        )

    return chunks


def extract_metadata_for_toon(document: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract document metadata suitable for TOON encoding.

    Designed for Hebbia-style document indexing.
    """
    return {
        "symbol": document.get("symbol", ""),
        "fiscal_year": document.get("fiscal_year", ""),
        "fiscal_quarter": document.get("fiscal_quarter", ""),
        "call_date": document.get("call_date", ""),
        "source": document.get("source", ""),
    }


def create_hebbia_style_index(
    document: Dict[str, Any], chunk_size: int = 2000
) -> List[Dict[str, Any]]:
    """
    Create a Hebbia-style preprocessed index from a document.

    This simulates the rich schema indexing that Hebbia does during
    their preprocessing/prepopulating pipeline.
    """
    metadata = extract_metadata_for_toon(document)
    transcript = document.get("transcript", "")
    chunks = chunk_transcript(transcript, chunk_size)

    # Enrich each chunk with document metadata
    indexed_chunks = []
    for chunk in chunks:
        indexed_chunks.append(
            {
                **metadata,
                **chunk,
            }
        )

    return indexed_chunks


# ============================================================================
# PROMPT TEMPLATE UTILITIES
# ============================================================================


def build_prompt_with_json(
    question: str,
    context_data: Any,
    system_prompt: str = "You are a financial analyst assistant.",
) -> str:
    """Build a prompt with JSON-formatted context."""
    json_context = json.dumps(context_data, indent=2, ensure_ascii=False)
    return f"""{system_prompt}

Context data:
```json
{json_context}
```

Question: {question}

Please provide a concise, accurate answer based on the context above."""


def build_prompt_with_toon(
    question: str,
    context_data: Any,
    system_prompt: str = "You are a financial analyst assistant.",
) -> str:
    """Build a prompt with TOON-formatted context."""
    toon_context = encode_toon(context_data)
    return f"""{system_prompt}

Context data:
```toon
{toon_context}
```

Question: {question}

Please provide a concise, accurate answer based on the context above."""


def compare_prompt_tokens(
    question: str,
    context_data: Any,
    system_prompt: str = "You are a financial analyst assistant.",
    model: str = "gpt-4o",
) -> Dict[str, Any]:
    """
    Compare token usage between JSON and TOON prompts.

    Returns detailed comparison including the full prompts.
    """
    json_prompt = build_prompt_with_json(question, context_data, system_prompt)
    toon_prompt = build_prompt_with_toon(question, context_data, system_prompt)

    json_tokens = count_tokens(json_prompt, model)
    toon_tokens = count_tokens(toon_prompt, model)

    savings = json_tokens - toon_tokens
    savings_pct = (savings / json_tokens * 100) if json_tokens > 0 else 0

    return {
        "json_prompt": json_prompt,
        "toon_prompt": toon_prompt,
        "json_tokens": json_tokens,
        "toon_tokens": toon_tokens,
        "savings_tokens": savings,
        "savings_percent": savings_pct,
    }


# ============================================================================
# VISUALIZATION HELPERS
# ============================================================================


def format_comparison_table(comparison: TokenComparison) -> str:
    """Format a comparison as a readable table."""
    return f"""
┌─────────────────────────────────────────────────────────┐
│                 Token Comparison                        │
├─────────────────────────────────────────────────────────┤
│ Format    │ Tokens     │ Savings                        │
├───────────┼────────────┼────────────────────────────────┤
│ JSON      │ {comparison.json_tokens:>10,} │                                │
│ TOON      │ {comparison.toon_tokens:>10,} │ {comparison.savings_absolute:>+10,} ({comparison.savings_percent:>5.1f}%) │
└─────────────────────────────────────────────────────────┘
"""


def format_benchmark_summary(result: BenchmarkResult) -> str:
    """Format benchmark results as a summary."""
    return f"""
╔═══════════════════════════════════════════════════════════╗
║                  TOON Benchmark Summary                   ║
╠═══════════════════════════════════════════════════════════╣
║ Samples analyzed:     {result.sample_count:>10}                          ║
║ Total JSON tokens:    {result.total_json_tokens:>10,}                          ║
║ Total TOON tokens:    {result.total_toon_tokens:>10,}                          ║
║ Total savings:        {result.total_savings:>10,} tokens                  ║
╠═══════════════════════════════════════════════════════════╣
║ Average savings:      {result.avg_savings_percent:>10.1f}%                          ║
║ Min savings:          {result.min_savings_percent:>10.1f}%                          ║
║ Max savings:          {result.max_savings_percent:>10.1f}%                          ║
╚═══════════════════════════════════════════════════════════╝
"""


def format_cost_savings(savings: Dict[str, float]) -> str:
    """Format Hebbia cost savings as readable output."""
    return f"""
╔═══════════════════════════════════════════════════════════════════╗
║            Estimated Hebbia-Scale Cost Savings                    ║
║                  (Based on 250B LLM calls/month)                  ║
╠═══════════════════════════════════════════════════════════════════╣
║ Token Savings Applied:      {savings['savings_percent_applied']:>8.1f}%                          ║
╠═══════════════════════════════════════════════════════════════════╣
║ Monthly Baseline Tokens:    {savings['monthly_baseline_tokens']:>18,.0f}              ║
║ Monthly Token Savings:      {savings['monthly_token_savings']:>18,.0f}              ║
╠═══════════════════════════════════════════════════════════════════╣
║ Monthly Cost Savings:       ${savings['monthly_cost_savings']:>17,.2f}              ║
║ Annual Cost Savings:        ${savings['annual_cost_savings']:>17,.2f}              ║
╚═══════════════════════════════════════════════════════════════════╝
"""


# ============================================================================
# HEBBIA-STYLE SCHEMA GENERATORS
# ============================================================================
# These generators create synthetic data approximating three schemas publicly
# described by Hebbia in their blog posts:
# 1. MatrixStripProfileRow - LevFin/equity comps grid rows with citations
# 2. ISDCitation - Line-level citation objects from ISD
# 3. AgentTurnContext - Agent 2.0 multi-agent message history
#
# References:
# - https://www.hebbia.com/blog/3-ways-leveraged-finance-teams-use-hebbia-to-cut-time-from-diligence-to-deal
# - https://www.hebbia.com/blog/inside-hebbias-deeper-research-agent
# - https://www.hebbia.com/blog/divide-and-conquer-hebbias-multi-agent-redesign

import random
from datetime import datetime, timedelta
import uuid

# Default scale configuration
DEFAULT_SCALE_CONFIG = {
    "matrix_rows": 1000,
    "isd_citations": 5000,
    "context_messages": 100,
}

# Realistic ticker data with actual revenue ranges (in billions)
COMPANY_DATA = [
    ("NVDA", "NVIDIA Corporation", "Semiconductors", 60.9),
    ("AAPL", "Apple Inc.", "Technology Hardware", 383.3),
    ("MSFT", "Microsoft Corporation", "Software", 211.9),
    ("GOOGL", "Alphabet Inc.", "Internet Services", 307.4),
    ("AMZN", "Amazon.com Inc.", "E-Commerce", 574.8),
    ("META", "Meta Platforms Inc.", "Social Media", 134.9),
    ("TSLA", "Tesla Inc.", "Electric Vehicles", 96.8),
    ("AMD", "Advanced Micro Devices", "Semiconductors", 22.7),
    ("INTC", "Intel Corporation", "Semiconductors", 54.2),
    ("CRM", "Salesforce Inc.", "Enterprise Software", 34.9),
    ("ORCL", "Oracle Corporation", "Enterprise Software", 52.9),
    ("ADBE", "Adobe Inc.", "Software", 19.4),
    ("NFLX", "Netflix Inc.", "Streaming", 33.7),
    ("PYPL", "PayPal Holdings", "Fintech", 29.8),
    ("UBER", "Uber Technologies", "Transportation", 37.3),
]

FISCAL_PERIODS = [
    (2024, "Q1"),
    (2024, "Q2"),
    (2024, "Q3"),
    (2024, "Q4"),
    (2025, "Q1"),
    (2025, "Q2"),
    (2025, "Q3"),
    (2025, "Q4"),
]

# Document types aligned with LevFin/credit workflows
DOC_TYPES_LEVFIN = ["10-K", "10-Q", "Credit Agreement", "PIB", "Presentation"]
DOC_TYPES_ISD = ["10-K", "10-Q", "Credit Agreement", "Presentation", "Indenture"]

# Section paths for different document types
SECTION_PATHS_10K = [
    ["Item 7", "MD&A", "Results of Operations"],
    ["Item 7", "MD&A", "Liquidity and Capital Resources"],
    ["Item 1A", "Risk Factors", "Competition"],
    ["Item 1A", "Risk Factors", "Regulatory"],
    ["Item 8", "Financial Statements", "Revenue Recognition"],
]

SECTION_PATHS_CREDIT = [
    ["Section 6", "Negative Covenants", "Indebtedness"],
    ["Section 6", "Negative Covenants", "Liens"],
    ["Section 6", "Negative Covenants", "Restricted Payments"],
    ["Section 7", "Events of Default"],
    ["Article I", "Definitions", "Consolidated EBITDA"],
    ["Article I", "Definitions", "Net Leverage Ratio"],
]

SECTION_PATHS_PRESENTATION = [
    ["Financial Highlights", "Q3 2025 Results"],
    ["Business Update", "Growth Initiatives"],
    ["Guidance", "FY2025 Outlook"],
]

# Agent 2.0 components (from Hebbia's multi-agent redesign blog)
AGENT_NAMES = [
    "Orchestrator",
    "ReadMatrixAgent",
    "SearchISDAgent",
    "OutputAgent",
    "DecomposerAgent",
    "ValidatorAgent",
    "CitationAgent",
    "SynthesizerAgent",
]

TOOL_NAMES = [
    "read_matrix",
    "search_isd",
    "fetch_document",
    "extract_metrics",
    "validate_citation",
    "synthesize_answer",
    "summarize_deck",
]

# Labels for ISD citations
CITATION_LABELS = [
    "covenant",
    "risk",
    "guidance",
    "leverage",
    "liquidity",
    "margin",
    "growth",
]


def generate_matrix_rows(
    num_companies: int = None,
    num_metrics: int = 8,
    quarters: int = 4,
    scale_cfg: Dict = None,
) -> List[Dict[str, Any]]:
    """
    Generate MatrixStripProfileRow objects for LevFin/equity comps.

    Models Hebbia's Matrix grid with strip-profile rows containing:
    - Company identifiers (ticker, name, sector)
    - Fiscal period (e.g., Q2 2025, FY2024)
    - Financial metrics (revenue, EBITDA, margins, leverage) in realistic magnitudes
    - Source citations with doc_id, section_path, page
    - Row metadata with agent attribution and confidence scores

    Args:
        num_companies: Number of companies (defaults to scale_cfg["matrix_rows"] / quarters)
        num_metrics: Metrics per row
        quarters: Number of quarters per company
        scale_cfg: Scale configuration dict

    Returns:
        List of MatrixStripProfileRow dicts
    """
    cfg = scale_cfg or DEFAULT_SCALE_CONFIG
    if num_companies is None:
        num_companies = max(10, cfg["matrix_rows"] // quarters)

    rows = []
    row_id = 0

    # Use company data with realistic revenue, cycling if needed
    companies = COMPANY_DATA * ((num_companies // len(COMPANY_DATA)) + 1)

    for company_idx in range(num_companies):
        ticker, company_name, sector, annual_revenue_b = companies[
            company_idx % len(companies)
        ]

        # Quarterly revenue is ~25% of annual, with variation
        base_quarterly_revenue = (annual_revenue_b * 1_000_000_000) / 4

        # Realistic margins by sector
        if sector == "Software" or sector == "Enterprise Software":
            base_ebitda_margin = random.uniform(0.25, 0.40)
            base_gross_margin = random.uniform(0.70, 0.85)
        elif sector == "Semiconductors":
            base_ebitda_margin = random.uniform(0.35, 0.55)
            base_gross_margin = random.uniform(0.55, 0.75)
        else:
            base_ebitda_margin = random.uniform(0.15, 0.30)
            base_gross_margin = random.uniform(0.35, 0.55)

        base_net_debt = (
            base_quarterly_revenue * 4 * random.uniform(0, 0.4)
        )  # 0-40% of annual revenue

        for q_idx in range(min(quarters, len(FISCAL_PERIODS))):
            fy, fq = FISCAL_PERIODS[q_idx]

            # Quarterly variation
            growth_factor = 1 + random.uniform(-0.05, 0.12)
            revenue = base_quarterly_revenue * growth_factor
            ebitda = revenue * base_ebitda_margin

            # Determine doc type based on quarter
            if fq == "Q4":
                doc_type = "10-K"
                section_path = random.choice(SECTION_PATHS_10K)
            else:
                doc_type = "10-Q"
                section_path = random.choice(SECTION_PATHS_10K)

            row = {
                "row_id": f"row_{row_id:05d}",
                "company_name": company_name,
                "ticker": ticker,
                "sector": sector,
                "fiscal_year": fy,
                "fiscal_quarter": fq,
                "currency": "USD",
                "metrics": {
                    "revenue_mm": round(revenue / 1_000_000, 1),  # In millions
                    "ebitda_mm": round(ebitda / 1_000_000, 1),
                    "ebitda_margin": round(
                        base_ebitda_margin + random.uniform(-0.02, 0.02), 3
                    ),
                    "gross_margin": round(
                        base_gross_margin + random.uniform(-0.02, 0.02), 3
                    ),
                    "net_debt_mm": round(
                        base_net_debt / 1_000_000 * (1 + random.uniform(-0.1, 0.1)), 1
                    ),
                    "net_leverage": (
                        round(base_net_debt / (revenue * 4 * base_ebitda_margin), 2)
                        if ebitda > 0
                        else 0
                    ),
                    "yoy_growth": round(random.uniform(-0.05, 0.35), 3),
                    "seq_growth": round(random.uniform(-0.03, 0.15), 3),
                },
                "source": {
                    "doc_id": f"{doc_type.replace('-', '')}_{fy}_{ticker}",
                    "doc_type": doc_type,
                    "section_path": section_path,
                    "page": random.randint(25, 120),
                    "citation_ids": [
                        f"cit_{ticker.lower()}_{fq.lower()}_{fy}_{i}"
                        for i in range(random.randint(1, 3))
                    ],
                },
                "row_metadata": {
                    "created_by_agent": "MatrixOutputAgent",
                    "created_at": (
                        datetime.now() - timedelta(days=random.randint(0, 30))
                    ).isoformat()
                    + "Z",
                    "confidence": round(random.uniform(0.70, 0.99), 2),
                },
            }
            rows.append(row)
            row_id += 1

            if len(rows) >= cfg["matrix_rows"]:
                return rows

    return rows


def flatten_matrix_rows(rows):
    flat = []
    for r in rows:
        out = {
            k: v for k, v in r.items() if k not in ("metrics", "source", "row_metadata")
        }
        for k, v in r["metrics"].items():
            out[f"m_{k}"] = v
        # Option A: keep only doc_id + page
        out["source_doc_id"] = r["source"]["doc_id"]
        out["source_page"] = r["source"]["page"]
        flat.append(out)
    return flat


def generate_isd_citations(
    num_docs: int = None, citations_per_doc: int = None, scale_cfg: Dict = None
) -> List[Dict[str, Any]]:
    """
    Generate ISDCitation objects for line-level document citations.

    Models Hebbia's ISD output with:
    - Citation identifiers
    - Document metadata (10-K, Credit Agreement, Presentation)
    - Location (page, char_start, char_end, section_path)
    - Snippet text in legal/financial style
    - Relevance/confidence scores (0.60-0.98 range)
    - Labels from covenant, risk, guidance, leverage categories

    Args:
        num_docs: Number of source documents
        citations_per_doc: Citations per document
        scale_cfg: Scale configuration dict

    Returns:
        List of ISDCitation dicts
    """
    cfg = scale_cfg or DEFAULT_SCALE_CONFIG
    target_citations = cfg["isd_citations"]

    if num_docs is None:
        num_docs = 350  # Realistic for "Deeper" research runs
    if citations_per_doc is None:
        citations_per_doc = max(1, target_citations // num_docs)

    citations = []
    citation_id = 0

    # Generate documents with realistic types
    docs = []
    for i in range(num_docs):
        ticker, company_name, sector, _ = random.choice(COMPANY_DATA)
        fy, fq = random.choice(FISCAL_PERIODS)
        doc_type = random.choice(DOC_TYPES_ISD)

        if doc_type in ["10-K", "10-Q"]:
            title = f"{company_name} Form {doc_type} FY{fy}"
            section_paths = SECTION_PATHS_10K
        elif doc_type == "Credit Agreement":
            title = f"{company_name} Senior Secured Credit Agreement"
            section_paths = SECTION_PATHS_CREDIT
        else:
            title = f"{company_name} {doc_type} {fq} {fy}"
            section_paths = SECTION_PATHS_PRESENTATION

        docs.append(
            {
                "doc_id": f"{doc_type.replace(' ', '_').replace('-', '')}_{fy}_{ticker}_{i:03d}",
                "doc_type": doc_type,
                "title": title,
                "source_system": random.choice(["EDGAR", "S&P Capital IQ", "internal"]),
                "symbol": ticker,
                "fiscal_year": fy,
                "fiscal_quarter": fq,
                "section_paths": section_paths,
            }
        )

    # Legal/financial snippet templates (grammatically correct, realistic style)
    snippet_templates_10k = [
        "Total revenue increased {pct}% year-over-year to ${value} billion, driven primarily by {factor}.",
        "The Company recorded EBITDA of ${value} million, representing a margin of {margin}%, compared to {prev_margin}% in the prior period.",
        "Net leverage ratio as of the balance sheet date was {ratio}x, compared to {prev_ratio}x in the prior quarter.",
        "Management anticipates revenue growth of {growth}% to {growth_high}% in the upcoming fiscal year, subject to market conditions.",
        "Gross margin improved {delta} basis points to {margin}%, reflecting operational efficiencies and favorable product mix.",
    ]

    snippet_templates_credit = [
        "The Borrower shall not permit the Net Leverage Ratio to exceed {max_ratio}:1.00 as of the last day of any fiscal quarter.",
        "Consolidated EBITDA means, for any period, Consolidated Net Income plus interest expense, taxes, depreciation and amortization.",
        "The Borrower shall maintain a minimum Interest Coverage Ratio of not less than {min_ratio}:1.00.",
        "Indebtedness incurred pursuant to this Section shall not exceed ${value} million in aggregate principal amount.",
        "Each Restricted Payment shall be permitted only if no Default or Event of Default exists before or after giving effect thereto.",
    ]

    snippet_templates_presentation = [
        "Q{quarter} revenue of ${value} billion exceeded guidance midpoint by {delta}%, demonstrating strong execution.",
        "We expect {metric} to reach ${value} billion by year-end, representing {growth}% growth versus prior year.",
        "Operating margin expanded {delta} basis points year-over-year, driven by {factor}.",
        "Free cash flow of ${value} million reflects disciplined capital allocation and working capital improvements.",
    ]

    for doc in docs:
        # Select appropriate templates based on doc type
        if doc["doc_type"] in ["10-K", "10-Q"]:
            templates = snippet_templates_10k
            section_paths = SECTION_PATHS_10K
        elif doc["doc_type"] == "Credit Agreement":
            templates = snippet_templates_credit
            section_paths = SECTION_PATHS_CREDIT
        else:
            templates = snippet_templates_presentation
            section_paths = SECTION_PATHS_PRESENTATION

        for _ in range(citations_per_doc):
            template = random.choice(templates)

            # Fill in template with realistic values
            snippet = template.format(
                pct=random.randint(5, 45),
                value=round(random.uniform(1.5, 85.0), 1),
                factor=random.choice(
                    [
                        "data center demand",
                        "cloud adoption",
                        "AI infrastructure",
                        "enterprise expansion",
                        "cost optimization",
                    ]
                ),
                margin=round(random.uniform(18, 52), 1),
                prev_margin=round(random.uniform(15, 48), 1),
                ratio=round(random.uniform(1.5, 4.5), 1),
                prev_ratio=round(random.uniform(1.8, 5.0), 1),
                max_ratio=round(random.uniform(4.0, 6.5), 1),
                min_ratio=round(random.uniform(2.0, 3.5), 1),
                growth=random.randint(8, 35),
                growth_high=random.randint(36, 50),
                delta=random.randint(50, 350),
                metric=random.choice(["revenue", "EBITDA", "operating income"]),
                quarter=random.randint(1, 4),
            )

            page = random.randint(15, 180)
            char_start = page * 2800 + random.randint(0, 2000)

            citation = {
                "citation_id": f"cit_{citation_id:06d}",
                "doc_id": doc["doc_id"],
                "doc_type": doc["doc_type"],
                "title": doc["title"],
                "source_system": doc["source_system"],
                "symbol": doc["symbol"],
                "fiscal_year": doc["fiscal_year"],
                "location": {
                    "page": page,
                    "section_path": random.choice(section_paths),
                    "char_start": char_start,
                    "char_end": char_start + len(snippet),
                },
                "snippet": snippet,
                "scores": {
                    "retrieval_score": round(random.uniform(0.60, 0.98), 3),
                    "relevance_score": round(random.uniform(0.60, 0.98), 3),
                },
                "labels": random.sample(CITATION_LABELS, k=random.randint(1, 3)),
            }
            citations.append(citation)
            citation_id += 1

            if len(citations) >= target_citations:
                return citations

    return citations


def generate_context_history(
    messages_per_turn: int = None, scale_cfg: Dict = None
) -> Dict[str, Any]:
    """
    Generate an AgentTurnContext for Agent 2.0 multi-agent workflows.

    Models Hebbia's Agent 2.0 context (from "Divide and Conquer" blog) with:
    - Turn/session identifiers (UUID format)
    - Product and entry point
    - Message history mixing user, tool, and agent roles
    - Tool calls with realistic tool_args (e.g., {"issuer": "NVDA", "period": "Q2 2025"})
    - Metrics (end_to_end_latency_ms, model_family, tokens_input/output)
    - Trace info (datadog_trace_id, maximizer_request_id)

    Args:
        messages_per_turn: Number of messages in the turn
        scale_cfg: Scale configuration dict

    Returns:
        AgentTurnContext dict
    """
    cfg = scale_cfg or DEFAULT_SCALE_CONFIG
    if messages_per_turn is None:
        messages_per_turn = cfg["context_messages"]

    turn_id = f"urn:uuid1:{uuid.uuid1()}"
    session_id = f"urn:uuid:{uuid.uuid4()}"

    # Sample issuers for tool_args
    sample_tickers = ["NVDA", "AAPL", "MSFT", "GOOGL", "AMZN"]
    sample_periods = ["Q1 2025", "Q2 2025", "Q3 2025", "FY2024"]

    # Tool args templates for different tools
    def get_tool_args(tool_name: str) -> Dict:
        ticker = random.choice(sample_tickers)
        period = random.choice(sample_periods)
        if tool_name == "read_matrix":
            return {
                "matrix_id": f"matrix_tech_comps_{period.replace(' ', '_').lower()}",
                "columns": random.sample(
                    [
                        "ticker",
                        "revenue_mm",
                        "ebitda_margin",
                        "net_leverage",
                        "gross_margin",
                    ],
                    k=3,
                ),
                "filter": f"fiscal_period = '{period}'",
            }
        elif tool_name == "search_isd":
            return {
                "query": random.choice(
                    [
                        "covenant analysis",
                        "EBITDA definition",
                        "leverage restrictions",
                        "risk factors",
                    ]
                ),
                "issuer": ticker,
                "doc_types": random.sample(["10-K", "Credit Agreement", "10-Q"], k=2),
                "limit": random.choice([25, 50, 100]),
            }
        elif tool_name == "fetch_document":
            return {
                "doc_id": f"10K_{random.randint(2024, 2025)}_{ticker}",
                "sections": ["Item 7", "Item 1A"],
            }
        elif tool_name == "extract_metrics":
            return {
                "issuer": ticker,
                "period": period,
                "metrics": ["revenue", "ebitda", "net_debt"],
            }
        elif tool_name == "summarize_deck":
            return {
                "doc_id": f"presentation_{ticker}_{period.replace(' ', '_')}",
                "max_tokens": 500,
            }
        else:
            return {"query": f"analysis for {ticker}", "period": period}

    messages = []
    base_time = datetime.now() - timedelta(minutes=5)

    # Start with user message
    messages.append(
        {
            "message_id": "msg_0001",
            "timestamp": base_time.isoformat() + "Z",
            "agent": "User",
            "role": "user",
            "content": "Which of the big tech giants has the highest EBITDA margin and how does it compare to the peer median? Include citations.",
            "tool_name": None,
            "tool_args": None,
            "visible_to": ["Orchestrator", "ReadMatrixAgent", "OutputAgent"],
        }
    )

    # Orchestrator decomposition
    messages.append(
        {
            "message_id": "msg_0002",
            "timestamp": (base_time + timedelta(milliseconds=500)).isoformat() + "Z",
            "agent": "Orchestrator",
            "role": "agent",
            "content": "Decomposing task: [1] fetch latest filings via ReadMatrixAgent, [2] compute EBITDA margins, [3] compare to peer median, [4] synthesize answer with citations via OutputAgent.",
            "tool_name": None,
            "tool_args": None,
            "visible_to": ["ReadMatrixAgent", "SearchISDAgent"],
        }
    )

    # Generate mixed message types
    msg_id = 3
    for i in range(min(messages_per_turn - 5, 95)):
        time_offset = timedelta(milliseconds=500 + i * 100)
        agent = random.choice(AGENT_NAMES)
        roll = random.random()

        if roll < 0.25:  # 25% tool calls (agent invoking a tool)
            tool = random.choice(TOOL_NAMES)
            messages.append(
                {
                    "message_id": f"msg_{msg_id:04d}",
                    "timestamp": (base_time + time_offset).isoformat() + "Z",
                    "agent": agent,
                    "role": "agent",
                    "content": None,
                    "tool_name": tool,
                    "tool_args": get_tool_args(tool),
                    "visible_to": [agent],
                }
            )
        elif roll < 0.50:  # 25% tool responses
            tool = random.choice(TOOL_NAMES)
            result_count = random.randint(5, 150)
            messages.append(
                {
                    "message_id": f"msg_{msg_id:04d}",
                    "timestamp": (base_time + time_offset).isoformat() + "Z",
                    "agent": agent,
                    "role": "tool",
                    "content": f"Retrieved {result_count} results from {tool}. Top result confidence: {random.uniform(0.85, 0.98):.2f}",
                    "tool_name": tool,
                    "tool_args": None,
                    "visible_to": ["Orchestrator", "OutputAgent"],
                }
            )
        else:  # 50% agent reasoning messages
            reasoning_types = [
                f"Analyzing {random.choice(sample_tickers)} financials for {random.choice(sample_periods)}",
                f"Cross-referencing {random.randint(3, 12)} citations across documents",
                f"Computing peer median from {random.randint(8, 20)} issuers",
                f"Validating covenant compliance: {random.choice(['PASS', 'PASS', 'REVIEW'])}",
                f"Synthesizing findings from {random.randint(2, 5)} sub-agents",
            ]
            messages.append(
                {
                    "message_id": f"msg_{msg_id:04d}",
                    "timestamp": (base_time + time_offset).isoformat() + "Z",
                    "agent": agent,
                    "role": "agent",
                    "content": random.choice(reasoning_types),
                    "tool_name": None,
                    "tool_args": None,
                    "visible_to": random.sample(AGENT_NAMES, k=random.randint(1, 3)),
                }
            )
        msg_id += 1

    # Final OutputAgent response
    messages.append(
        {
            "message_id": f"msg_{msg_id:04d}",
            "timestamp": (base_time + timedelta(seconds=4)).isoformat() + "Z",
            "agent": "OutputAgent",
            "role": "agent",
            "content": "NVIDIA has the highest EBITDA margin at 46.2%, which is 12.3 percentage points above the peer median of 33.9%. [cit_000142][cit_000287] The margin improvement is primarily driven by data center revenue growth and operational efficiencies in manufacturing.",
            "tool_name": None,
            "tool_args": None,
            "visible_to": ["User"],
        }
    )

    # Build full context with metrics and trace
    total_input_tokens = random.randint(8000, 45000)
    context = {
        "turn_id": turn_id,
        "session_id": session_id,
        "product": random.choice(["matrix", "chat", "research"]),
        "entry_point": "matrix_agent_v2",
        "message_history": messages,
        "metrics": {
            "end_to_end_latency_ms": random.randint(2500, 7500),
            "max_model_latency_ms": random.randint(400, 1800),
            "model_family": random.choice(
                ["gpt-4o", "claude-3.5-sonnet", "gemini-1.5-pro"]
            ),
            "tokens_input": total_input_tokens,
            "tokens_output": random.randint(500, 2500),
        },
        "trace": {
            "datadog_trace_id": f"{random.randint(10**15, 10**16):x}",
            "maximizer_request_id": f"mxr_{random.randint(10**8, 10**9)}",
            "partitions": random.sample(range(1, 16), k=random.randint(1, 3)),
        },
    }

    return context


def assemble_json_payload(
    question: str,
    matrix_rows: List[Dict],
    citations: List[Dict],
    context: Dict,
    system_prompt: str = "You are a financial analyst assistant for Hebbia.",
) -> str:
    """
    Assemble a complete JSON payload for an end-to-end research turn.

    Args:
        question: The user's question
        matrix_rows: List of MatrixStripProfileRow dicts
        citations: List of ISDCitation dicts
        context: AgentTurnContext dict
        system_prompt: System prompt

    Returns:
        Complete JSON-formatted prompt string
    """
    payload = {
        "matrix_data": matrix_rows,
        "citations": citations,
        "agent_context": context,
    }

    json_context = json.dumps(payload, indent=2, ensure_ascii=False)

    return f"""{system_prompt}

Context data:
```json
{json_context}
```

Question: {question}

Provide a detailed answer with citations."""


def assemble_toon_payload(
    question: str,
    matrix_rows: List[Dict],
    citations: List[Dict],
    context: Dict,
    system_prompt: str = "You are a financial analyst assistant for Hebbia.",
) -> str:
    """
    Assemble a complete TOON payload for an end-to-end research turn.

    Args:
        question: The user's question
        matrix_rows: List of MatrixStripProfileRow dicts
        citations: List of ISDCitation dicts
        context: AgentTurnContext dict
        system_prompt: System prompt

    Returns:
        Complete TOON-formatted prompt string
    """
    payload = {
        "matrix_data": matrix_rows,
        "citations": citations,
        "agent_context": context,
    }

    toon_context = encode_toon(payload)

    return f"""{system_prompt}

Context data:
```toon
{toon_context}
```

Question: {question}

Provide a detailed answer with citations."""


def compare_hebbia_payloads(
    question: str,
    matrix_rows: List[Dict],
    citations: List[Dict],
    context: Dict,
    model: str = "gpt-4o",
) -> Dict[str, Any]:
    """
    Compare JSON vs TOON for a complete Hebbia-style payload.

    Returns breakdown by component and totals.
    """
    # Component-level comparison
    matrix_cmp = compare_formats(matrix_rows, model)
    citations_cmp = compare_formats(citations, model)
    context_cmp = compare_formats(context, model)

    # Full payload comparison
    json_payload = assemble_json_payload(question, matrix_rows, citations, context)
    toon_payload = assemble_toon_payload(question, matrix_rows, citations, context)

    json_total = count_tokens(json_payload, model)
    toon_total = count_tokens(toon_payload, model)

    return {
        "matrix": {
            "json_tokens": matrix_cmp.json_tokens,
            "toon_tokens": matrix_cmp.toon_tokens,
            "savings_percent": matrix_cmp.savings_percent,
        },
        "citations": {
            "json_tokens": citations_cmp.json_tokens,
            "toon_tokens": citations_cmp.toon_tokens,
            "savings_percent": citations_cmp.savings_percent,
        },
        "context": {
            "json_tokens": context_cmp.json_tokens,
            "toon_tokens": context_cmp.toon_tokens,
            "savings_percent": context_cmp.savings_percent,
        },
        "total": {
            "json_tokens": json_total,
            "toon_tokens": toon_total,
            "savings_tokens": json_total - toon_total,
            "savings_percent": (
                ((json_total - toon_total) / json_total * 100) if json_total > 0 else 0
            ),
        },
        "json_payload": json_payload,
        "toon_payload": toon_payload,
    }


if __name__ == "__main__":
    # Quick self-test
    test_data = {
        "users": [
            {"id": 1, "name": "Alice", "role": "admin"},
            {"id": 2, "name": "Bob", "role": "user"},
            {"id": 3, "name": "Charlie", "role": "user"},
        ]
    }

    result = compare_formats(test_data)
    print(format_comparison_table(result))
    print("\nJSON:")
    print(result.json_text)
    print("\nTOON:")
    print(result.toon_text)

    # Test new generators
    print("\n" + "=" * 70)
    print("TESTING HEBBIA-STYLE GENERATORS")
    print("=" * 70)

    # Test with small scale
    small_cfg = {"matrix_rows": 10, "isd_citations": 20, "context_messages": 10}

    print("\n1. Matrix Rows:")
    rows = generate_matrix_rows(scale_cfg=small_cfg)
    print(f"   Generated {len(rows)} rows")
    print(f"   Sample: {rows[0]['ticker']} - ${rows[0]['metrics']['revenue']:,}")

    print("\n2. ISD Citations:")
    cites = generate_isd_citations(scale_cfg=small_cfg)
    print(f"   Generated {len(cites)} citations")
    print(f"   Sample: {cites[0]['citation_id']} - {cites[0]['snippet'][:50]}...")

    print("\n3. Agent Context:")
    ctx = generate_context_history(scale_cfg=small_cfg)
    print(f"   Generated context with {len(ctx['message_history'])} messages")
    print(f"   Turn ID: {ctx['turn_id'][:40]}...")
