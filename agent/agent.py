from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from agent.state import AgentState
from agent.nodes import (
    query_understanding_agent,
    orchestrator_agent,
    context_retrieval_agent,
    data_retrieval_agent,
    analysis_computation_agent,
    visualization_specification_agent,
    visualization_rendering_agent,
    insight_generation_agent
)
from agent.tools import ALL_TOOLS


def route_after_orchestrator(state: AgentState) -> str:
    """Route after orchestrator decision"""
    next_action = state.get('next_action')
    
    if next_action == 'ask_clarification':
        return 'end'
    elif next_action == 'retrieve_data':
        return 'context_retrieval'
    else:
        return 'end'


def route_after_data_retrieval(state: AgentState) -> str:
    """Route to tools if there are tool calls, otherwise to analysis"""
    last_message = state['messages'][-1]
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        return 'tools'
    return 'analysis'


def route_after_tools(state: AgentState) -> str:
    """After tool execution, go to analysis"""
    return 'analysis'


def route_after_analysis(state: AgentState) -> str:
    """After analysis, check if visualization is needed"""
    if state.get('requires_visualization'):
        return 'visualization_spec'
    return 'insight_generation'


def route_after_viz_spec(state: AgentState) -> str:
    """After viz spec, render charts then generate insights"""
    viz_specs = state.get('visualization_specs', [])
    if viz_specs:
        return 'visualization_rendering'
    return 'insight_generation'


def route_after_viz_rendering(state: AgentState) -> str:
    """After rendering, always go to insight generation"""
    return 'insight_generation'


def route_after_insights(state: AgentState) -> str:
    """Final routing - always end after insights"""
    return 'end'


# Create tool node
tool_node = ToolNode(ALL_TOOLS)

# Build the graph
builder = StateGraph(AgentState)

# Add all nodes
builder.add_node('query_understanding', query_understanding_agent)
builder.add_node('orchestrator', orchestrator_agent)
builder.add_node('context_retrieval', context_retrieval_agent)
builder.add_node('data_retrieval', data_retrieval_agent)
builder.add_node('tools', tool_node)
builder.add_node('analysis', analysis_computation_agent)
builder.add_node('visualization_spec', visualization_specification_agent)
builder.add_node('visualization_rendering', visualization_rendering_agent)
builder.add_node('insight_generation', insight_generation_agent)

# Define edges
builder.add_edge(START, 'query_understanding')
builder.add_edge('query_understanding', 'orchestrator')

# Orchestrator routing
builder.add_conditional_edges(
    'orchestrator',
    route_after_orchestrator,
    {
        'context_retrieval': 'context_retrieval',
        'end': END
    }
)

# Context retrieval always goes to data retrieval
builder.add_edge('context_retrieval', 'data_retrieval')

# Data retrieval routing (to tools or analysis)
builder.add_conditional_edges(
    'data_retrieval',
    route_after_data_retrieval,
    {
        'tools': 'tools',
        'analysis': 'analysis'
    }
)

# Tools always go to analysis
builder.add_conditional_edges(
    'tools',
    route_after_tools,
    {
        'analysis': 'analysis'
    }
)

# Analysis routing (to viz spec or insights)
builder.add_conditional_edges(
    'analysis',
    route_after_analysis,
    {
        'visualization_spec': 'visualization_spec',
        'insight_generation': 'insight_generation'
    }
)

# Visualization spec routing
builder.add_conditional_edges(
    'visualization_spec',
    route_after_viz_spec,
    {
        'visualization_rendering': 'visualization_rendering',
        'insight_generation': 'insight_generation'
    }
)

# Visualization rendering routing
builder.add_conditional_edges(
    'visualization_rendering',
    route_after_viz_rendering,
    {
        'insight_generation': 'insight_generation'
    }
)

# Insight generation routing
builder.add_conditional_edges(
    'insight_generation',
    route_after_insights,
    {
        'end': END
    }
)

# Compile the graph
graph = builder.compile()