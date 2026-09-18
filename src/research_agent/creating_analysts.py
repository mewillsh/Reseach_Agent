from research_agent.utils.nodes import create_analysts, human_feedback
from research_agent.utils.edges import should_continue
from langgraph.graph import START, END, StateGraph
from research_agent.utils.states import GenerateAnalystsState
from dotenv import load_dotenv

load_dotenv()

# creating our graph
# GenerateAnalystsState state
# nodes create_analysts
# edges no need / simple

builder = StateGraph(GenerateAnalystsState)

builder.add_node("create_analysts", create_analysts)
builder.add_node("human_feedback", human_feedback)

builder.add_edge(START, "create_analysts")
builder.add_edge("create_analysts", "human_feedback")
builder.add_conditional_edges("human_feedback", should_continue)

graph = builder.compile()