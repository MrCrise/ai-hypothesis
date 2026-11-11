from typing import TypedDict

from langgraph.graph import END, START, StateGraph


class PipelineState(TypedDict):
    topic_id: str
    query: str
    sub_questions: list[dict]
    papers_found: int
    hypotheses: list[dict]


async def plan_node(state: PipelineState) -> dict:
    return {"sub_questions": []}


async def search_filter_node(state: PipelineState) -> dict:
    return {"papers_found": 0}


async def ingest_node(state: PipelineState) -> dict:
    return {}


async def overview_gaps_node(state: PipelineState) -> dict:
    return {}


async def generate_hypotheses_node(state: PipelineState) -> dict:
    return {"hypotheses": []}


async def novelty_ranking_node(state: PipelineState) -> dict:
    return {}


def build_pipeline_graph() -> StateGraph:
    builder = StateGraph(PipelineState)
    builder.add_node("plan", plan_node)
    builder.add_node("search_filter", search_filter_node)
    builder.add_node("ingest", ingest_node)
    builder.add_node("overview_gaps", overview_gaps_node)
    builder.add_node("generate_hypotheses", generate_hypotheses_node)
    builder.add_node("novelty_ranking", novelty_ranking_node)

    builder.add_edge(START, "plan")
    builder.add_edge("plan", "search_filter")
    builder.add_edge("search_filter", "ingest")
    builder.add_edge("ingest", "overview_gaps")
    builder.add_edge("overview_gaps", "generate_hypotheses")
    builder.add_edge("generate_hypotheses", "novelty_ranking")
    builder.add_edge("novelty_ranking", END)
    return builder
