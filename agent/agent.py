from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from agent.state import AgentState
from agent.nodes import (
    query_understanding_agent,
    orchestrator_agent,
    context_retrieval_agent,
    sql_generation_agent,
    data_retrieval_agent,
    should_retry_sql,  
    analysis_computation_agent,
    visualization_specification_agent,
    visualization_rendering_agent,
    insight_generation_agent
)
from agent.tools import ALL_TOOLS
from enum import Enum

class RouteDecision(Enum):
    END = 'end'
    CONTEXT_RETRIEVAL = 'context_retrieval'
    SQL_GENERATION = 'sql_generation'
    DATA_RETRIEVAL = 'data_retrieval'
    TOOLS = 'tools'
    ANALYSIS = 'analysis'
    VIZ_SPEC = 'visualization_spec'
    VIZ_RENDER = 'visualization_rendering'
    INSIGHTS = 'insight_generation'


def route_after_orchestrator(state: AgentState) -> str:
    next_action = state.get('next_action')
    
    route_map = {
        'ask_clarification': RouteDecision.END,
        'retrieve_data': RouteDecision.CONTEXT_RETRIEVAL,
        'end': RouteDecision.END
    }
    
    route = route_map.get(next_action)
    
    if route is None:
        print(f"ERROR: Invalid next_action '{next_action}'. Valid: {list(route_map.keys())}")
        return RouteDecision.END.value
    
    return route.value


def route_after_context_retrieval(state: AgentState) -> str:
    """Route to SQL generation after context retrieval"""
    return RouteDecision.SQL_GENERATION.value


def route_after_sql_generation(state: AgentState) -> str:
    """Route to data retrieval after SQL generation"""
    generated_sql = state.get('generated_sql')
    
    if generated_sql:
        return RouteDecision.DATA_RETRIEVAL.value
    else:
        return RouteDecision.INSIGHTS.value


def route_after_data_retrieval(state: AgentState) -> str:
    """
    Route after data retrieval to either:
    - sql_generation (if retry needed)
    - tools (if tool calls present)
    - analysis (if data retrieved successfully)
    """

    needs_retry = state.get('needs_sql_retry', False)
    
    if needs_retry:
        return RouteDecision.SQL_GENERATION.value
    

    messages = state.get('messages', [])
    
    if not messages:
        print("WARNING: No messages, routing to analysis")
        return RouteDecision.ANALYSIS.value
    
    last_message = messages[-1]
    
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        return RouteDecision.TOOLS.value
    
    return RouteDecision.ANALYSIS.value


def route_after_tools(state: AgentState) -> str:
    """Route after tools execute to analysis"""
    return RouteDecision.ANALYSIS.value


def route_after_analysis(state: AgentState) -> str:
    analysis_results = state.get('analysis_results', {})
    raw_data = analysis_results.get('raw_data', [])
    
    has_visualizable_data = False
    for result in raw_data:
        if result.get('success') and result.get('data'):
            if len(result['data']) > 1:
                has_visualizable_data = True
                break
    
    if has_visualizable_data:
        return RouteDecision.VIZ_SPEC.value
    
    return RouteDecision.INSIGHTS.value


def route_after_viz_spec(state: AgentState) -> str:
    """Route after visualization spec to rendering or insights"""
    viz_specs = state.get('visualization_specs', [])
    
    if viz_specs and len(viz_specs) > 0:
        return RouteDecision.VIZ_RENDER.value
    
    print("WARNING: No viz specs, skipping rendering")
    return RouteDecision.INSIGHTS.value


def route_after_viz_rendering(state: AgentState) -> str:
    """Route after visualization rendering to insights"""
    return RouteDecision.INSIGHTS.value


def route_after_insights(state: AgentState) -> str:
    """Route after insights generation to end"""
    return RouteDecision.END.value


tool_node = ToolNode(ALL_TOOLS)

builder = StateGraph(AgentState)

builder.add_node('query_understanding', query_understanding_agent)
builder.add_node('orchestrator', orchestrator_agent)
builder.add_node('context_retrieval', context_retrieval_agent)
builder.add_node('sql_generation', sql_generation_agent)
builder.add_node('data_retrieval', data_retrieval_agent)
builder.add_node('tools', tool_node)
builder.add_node('analysis', analysis_computation_agent)
builder.add_node('visualization_spec', visualization_specification_agent)
builder.add_node('visualization_rendering', visualization_rendering_agent)
builder.add_node('insight_generation', insight_generation_agent)

builder.add_edge(START, 'query_understanding')
builder.add_edge('query_understanding', 'orchestrator')

builder.add_conditional_edges(
    'orchestrator',
    route_after_orchestrator,
    {
        RouteDecision.CONTEXT_RETRIEVAL.value: 'context_retrieval',
        RouteDecision.END.value: END
    }
)

builder.add_conditional_edges(
    'context_retrieval',
    route_after_context_retrieval,
    {
        RouteDecision.SQL_GENERATION.value: 'sql_generation'
    }
)

builder.add_conditional_edges(
    'sql_generation',
    route_after_sql_generation,
    {
        RouteDecision.DATA_RETRIEVAL.value: 'data_retrieval',
        RouteDecision.INSIGHTS.value: 'insight_generation'
    }
)

builder.add_conditional_edges(
    'data_retrieval',
    route_after_data_retrieval,
    {
        RouteDecision.SQL_GENERATION.value: 'sql_generation',  
        RouteDecision.TOOLS.value: 'tools',
        RouteDecision.ANALYSIS.value: 'analysis'
    }
)

builder.add_conditional_edges(
    'tools',
    route_after_tools,
    {
        RouteDecision.ANALYSIS.value: 'analysis'
    }
)

builder.add_conditional_edges(
    'analysis',
    route_after_analysis,
    {
        RouteDecision.VIZ_SPEC.value: 'visualization_spec',
        RouteDecision.INSIGHTS.value: 'insight_generation'
    }
)

builder.add_conditional_edges(
    'visualization_spec',
    route_after_viz_spec,
    {
        RouteDecision.VIZ_RENDER.value: 'visualization_rendering',
        RouteDecision.INSIGHTS.value: 'insight_generation'
    }
)

builder.add_conditional_edges(
    'visualization_rendering',
    route_after_viz_rendering,
    {
        RouteDecision.INSIGHTS.value: 'insight_generation'
    }
)

builder.add_conditional_edges(
    'insight_generation',
    route_after_insights,
    {
        RouteDecision.END.value: END
    }
)

graph = builder.compile()