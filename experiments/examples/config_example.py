"""
Example Configurations

Copy and modify these examples in your config.py file.
"""

from consensus_core import AgentConfig, Question


# ============================================================================
# EXAMPLE 1: Financial Q&A Evaluation
# ============================================================================

def get_agents_financial():
    """Compare concise vs detailed financial analysis agents."""
    agent_concise = AgentConfig(
        name="concise_analyst",
        model="gpt-4o",
        prompt="""You are a financial analyst. Provide concise, factual answers 
        based only on the provided document. Focus on key numbers and trends.""",
    )

    agent_detailed = AgentConfig(
        name="detailed_analyst",
        model="gpt-4o",
        prompt="""You are a senior financial analyst. Provide comprehensive, 
        detailed analysis with context and implications. Explain your reasoning.""",
    )

    return agent_concise, agent_detailed


def get_questions_financial():
    """Financial earnings call questions."""
    return [
        Question(
            id="revenue_drivers",
            text="What were the main revenue drivers this quarter?",
            criterion="factual_accuracy",
        ),
        Question(
            id="guidance",
            text="What is management's revenue guidance for next quarter?",
            criterion="verbatim_precision",
        ),
        Question(
            id="risks",
            text="What risks did management identify?",
            criterion="completeness",
        ),
    ]


# ============================================================================
# EXAMPLE 2: Legal Document Analysis
# ============================================================================

def get_agents_legal():
    """Compare different legal analysis styles."""
    agent_literal = AgentConfig(
        name="literal_reader",
        model="gpt-4o",
        prompt="""You are a legal document reader. Extract exact text and 
        clauses. Do not interpret or summarize.""",
    )

    agent_interpretive = AgentConfig(
        name="interpretive_analyst",
        model="gpt-4o",
        prompt="""You are a legal analyst. Extract information and explain 
        its implications and context.""",
    )

    return agent_literal, agent_interpretive


def get_questions_legal():
    """Legal document questions."""
    return [
        Question(
            id="termination_clauses",
            text="What are the termination conditions in this contract?",
            criterion="completeness",
        ),
        Question(
            id="liability",
            text="What are the liability limitations?",
            criterion="logical_coherence",
        ),
    ]


# ============================================================================
# EXAMPLE 3: Research Paper Summarization
# ============================================================================

def get_agents_research():
    """Compare different summarization approaches."""
    agent_abstract = AgentConfig(
        name="abstract_summarizer",
        model="gpt-4o",
        prompt="""You are a research assistant. Provide high-level summaries 
        focusing on main contributions and conclusions.""",
    )

    agent_technical = AgentConfig(
        name="technical_summarizer",
        model="gpt-4o",
        prompt="""You are a technical research assistant. Provide detailed 
        summaries including methodology, results, and technical details.""",
    )

    return agent_abstract, agent_technical


def get_questions_research():
    """Research paper questions."""
    return [
        Question(
            id="main_contribution",
            text="What is the main contribution of this paper?",
            criterion="insightfulness",
        ),
        Question(
            id="methodology",
            text="What methodology did the authors use?",
            criterion="factual_accuracy",
        ),
        Question(
            id="results",
            text="What were the key results?",
            criterion="completeness",
        ),
    ]


# ============================================================================
# EXAMPLE 4: Multi-Model Comparison
# ============================================================================

def get_agents_multimodel():
    """Compare different models with same prompt."""
    agent_gpt4o = AgentConfig(
        name="gpt4o",
        model="gpt-4o",
        prompt="You are a helpful assistant. Provide accurate, detailed answers.",
    )

    agent_gpt4turbo = AgentConfig(
        name="gpt4turbo",
        model="gpt-4-turbo",
        prompt="You are a helpful assistant. Provide accurate, detailed answers.",
    )

    return agent_gpt4o, agent_gpt4turbo

