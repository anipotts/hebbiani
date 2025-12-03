# Analysis: Personalization Points & Unified Architecture

## 📋 Analysis Summary

### What Needs Personalization in Async Version

The async version (`consensus_eval_async.py`) requires customization in these areas:

1. **Lines 27-38: Experiment Configuration**
   - `API_MODEL_AGENT` / `API_MODEL_EVAL`: Models to use
   - `N`: Bootstrap runs (default: 10)
   - `TEMPERATURES`: Temperature values
   - `MAX_CONCURRENT_REQUESTS`: Concurrency limit
   - `PERM_TEST_ITER`: Statistical test iterations

2. **Lines 356-366: Agent Definitions**
   - Agent names, models, and prompts
   - Currently compares "baseline_H0" vs "candidate_HA"

3. **Lines 369-385: Questions**
   - Question IDs, text, and evaluation criteria
   - Currently has 3 sample financial questions

4. **Lines 387-393: Document Context** ⚠️ **CRITICAL**
   - Currently a placeholder string
   - Must be replaced with actual document text for grounded Q&A
   - Can load from file or paste directly

5. **Lines 119-134: Agent Prompt Template**
   - Includes document context handling
   - Can be customized for different document types

### Key Differences: Sync vs Async

| Aspect | Sync | Async |
|--------|------|-------|
| **Execution** | Sequential | Parallel with semaphore |
| **Speed** | ~1.5s per call | ~0.2s per call (8x concurrency) |
| **Use Case** | Small experiments, debugging | Production runs |
| **Document Context** | Not supported | Supported |
| **Leaderboard** | No | Yes |
| **Default N** | 50 | 10 |

## 🏗️ Unified Architecture Solution

### Structure

```
experiments/
├── config.py                          # ⭐ SINGLE SOURCE OF TRUTH
│   ├── get_experiment_config()       # Experiment parameters
│   ├── get_agents()                  # Agent definitions
│   └── get_questions()               # Question definitions
│
├── consensus_core.py                  # Shared core logic
│   ├── Data structures (Question, AgentConfig, EvalRecord)
│   ├── Evaluation logic (extract_score_from_logprobs)
│   ├── Statistical tests (permutation_test, compute_significance)
│   └── Utilities (build_prompts, save_results)
│
├── consensus_eval_sync_refactored.py  # Sync executor
└── consensus_eval_async_refactored.py # Async executor
```

### Benefits

1. **Single Configuration Point**: All customization in `config.py`
2. **DRY Principle**: Shared logic in `consensus_core.py`
3. **Easy Customization**: Clear functions to override
4. **Type Safety**: Dataclasses for all structures
5. **Documentation**: Comprehensive README with examples

### Migration Path

**Old approach:**
- Edit config scattered across file
- Duplicate code between sync/async
- Hard to maintain consistency

**New approach:**
- Edit `config.py` only
- Both scripts use same config
- Easy to add new executors (e.g., batched, distributed)

## 🎯 For GitHub/Recruiters

### What This Demonstrates

1. **Software Engineering**
   - Clean architecture with separation of concerns
   - DRY principles and code reuse
   - Type hints and dataclasses for maintainability

2. **ML/Research Engineering**
   - Faithful reproduction of research methodology
   - Statistical rigor (permutation tests, bootstrap sampling)
   - Uncertainty quantification via logprobs

3. **Production Readiness**
   - Async/concurrent execution for scale
   - Error handling and progress tracking
   - Configurable and extensible design

4. **Documentation**
   - Clear README with examples
   - Inline documentation
   - Example configurations for different domains

### Key Files to Review

1. **`config.py`**: Shows how easy it is to customize experiments
2. **`consensus_core.py`**: Demonstrates clean abstractions
3. **`README.md`**: Comprehensive user guide
4. **`examples/config_example.py`**: Real-world use cases

### Talking Points

> "I implemented Hebbia's consensus evaluation framework with a clean, modular architecture. The framework separates configuration from execution, allowing easy experimentation while maintaining statistical rigor. I built both synchronous and asynchronous executors, with shared core logic to ensure consistency. The design makes it trivial to compare different agents, models, or evaluation criteria."

## 📝 Next Steps

1. **Test the refactored code** with your existing experiments
2. **Customize `config.py`** with your specific use case
3. **Add domain-specific examples** (legal, medical, etc.)
4. **Consider enhancements**:
   - Visualization tools
   - Cost tracking
   - Retry logic for rate limits
   - Distributed execution

