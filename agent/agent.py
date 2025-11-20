from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from agent.state import AgentState
from agent.nodes import (
    query_simplification_agent,
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
    QUERY_UNDERSTANDING = 'query_understanding'  # NEW: Route to understanding
    CONTEXT_RETRIEVAL = 'context_retrieval'
    SQL_GENERATION = 'sql_generation'
    DATA_RETRIEVAL = 'data_retrieval'
    TOOLS = 'tools'
    ANALYSIS = 'analysis'
    VIZ_SPEC = 'visualization_spec'
    VIZ_RENDER = 'visualization_rendering'
    INSIGHTS = 'insight_generation'


# NEW: Route after query simplification
def route_after_simplification(state: AgentState) -> str:
    """
    Route after query simplification based on whether query is conversational
    """
    is_conversational = state.get('is_conversational', False)
    
    if is_conversational:
        print("[Route] Conversational query detected, ending workflow")
        return RouteDecision.END.value
    else:
        print("[Route] Analytics query, proceeding to query understanding")
        return RouteDecision.QUERY_UNDERSTANDING.value


def route_after_orchestrator(state: AgentState) -> str:
    next_action = state.get('next_action')
    
    route_map = {
        'ask_clarification': RouteDecision.END,
        'retrieve_memory': RouteDecision.CONTEXT_RETRIEVAL,
        'skip_to_sql': RouteDecision.SQL_GENERATION,
        'end': RouteDecision.END
    }
    
    route = route_map.get(next_action)
    
    if route is None:
        print(f"ERROR: Invalid next_action '{next_action}'. Valid: {list(route_map.keys())}")
        return RouteDecision.END.value
    
    return route.value


def route_after_context_retrieval(state: AgentState) -> str:
    """Route after context retrieval based on needs_database flag"""
    needs_database = state.get('needs_database', True)
    
    if needs_database:
        return RouteDecision.SQL_GENERATION.value
    else:
        print("Memory-only query detected, skipping SQL generation")
        return RouteDecision.INSIGHTS.value


def route_after_sql_generation(state: AgentState) -> str:
    """Route to data retrieval after SQL generation"""
    generated_sql = state.get('generated_sql')
    
    if generated_sql:
        return RouteDecision.DATA_RETRIEVAL.value
    else:
        return RouteDecision.INSIGHTS.value


def route_after_data_retrieval(state: AgentState) -> str:
    """Route after data retrieval to either retry, tools, or analysis"""
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
    """Route after analysis based on whether data is visualizable"""
    analysis_results = state.get('analysis_results', {})
    raw_data = analysis_results.get('raw_data', [])
    
    has_visualizable_data = False
    
    for result in raw_data:
        if result.get('success') and result.get('data'):
            data = result['data']
            
            if not data or len(data) == 0:
                continue
            
            first_row = data[0]
            
            # Count numeric columns
            numeric_cols = 0
            non_metric_cols = ['model_id', 'execution_id', 'model_name', 
                              'algorithm', 'use_case', 'version', 'trained_date',
                              'created_at', 'model_type', 'description']
            
            for key, value in first_row.items():
                if key.lower() not in non_metric_cols:
                    try:
                        if isinstance(value, (int, float)) and value is not None:
                            numeric_cols += 1
                        elif isinstance(value, str):
                            float(value)
                            numeric_cols += 1
                    except (ValueError, TypeError):
                        continue
            
            is_multi_row_with_metrics = len(data) >= 2 and numeric_cols >= 1
            is_single_row_multi_metrics = len(data) == 1 and numeric_cols >= 2
            
            if is_multi_row_with_metrics or is_single_row_multi_metrics:
                has_visualizable_data = True
                print(f"[Route] ✓ Visualizable: {len(data)} rows, {numeric_cols} metrics")
                break
            else:
                print(f"[Route] Not visualizable: {len(data)} rows, {numeric_cols} metrics")
    
    if has_visualizable_data:
        print(f"[Route] Routing to visualization_spec")
        return RouteDecision.VIZ_SPEC.value
    
    print(f"[Route] No visualizable data, routing to insights")
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


# Create tool node
tool_node = ToolNode(ALL_TOOLS)

# Build graph
builder = StateGraph(AgentState)

# Add all nodes
builder.add_node('query_simplification', query_simplification_agent)
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

# Start with query simplification
builder.add_edge(START, 'query_simplification')

# NEW: Conditional edge after simplification
builder.add_conditional_edges(
    'query_simplification',
    route_after_simplification,
    {
        RouteDecision.QUERY_UNDERSTANDING.value: 'query_understanding',
        RouteDecision.END.value: END  # Exit early for conversational queries
    }
)

# Continue with normal flow
builder.add_edge('query_understanding', 'orchestrator')

builder.add_conditional_edges(
    'orchestrator',
    route_after_orchestrator,
    {
        RouteDecision.CONTEXT_RETRIEVAL.value: 'context_retrieval',
        RouteDecision.SQL_GENERATION.value: 'sql_generation',
        RouteDecision.END.value: END
    }
)

builder.add_conditional_edges(
    'context_retrieval',
    route_after_context_retrieval,
    {
        RouteDecision.SQL_GENERATION.value: 'sql_generation',
        RouteDecision.INSIGHTS.value: 'insight_generation'
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

# Compile graph
graph = builder.compile()