from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from agent.state import AgentState
from agent.nodes import query_understanding_agent, orchestrator_agent, data_retrieval_agent
from agent.tools import ALL_TOOLS

def route_after_orchestrator(state: AgentState) -> str:
    next_action = state.get('next_action')
    
    if next_action == 'ask_clarification':
        return 'end'
    elif next_action == 'retrieve_data':
        return 'data_retrieval'
    else:
        return 'end'

def route_after_data_retrieval(state: AgentState) -> str:
    """Route to tools if there are tool calls, otherwise end"""
    last_message = state['messages'][-1]
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        return 'tools'
    return 'end'

# Create tool node
tool_node = ToolNode(ALL_TOOLS)

# Build the graph
builder = StateGraph(AgentState)

builder.add_node('query_understanding', query_understanding_agent)
builder.add_node('orchestrator', orchestrator_agent)
builder.add_node('data_retrieval', data_retrieval_agent)
builder.add_node('tools', tool_node)

# Define edges
builder.add_edge(START, 'query_understanding')
builder.add_edge('query_understanding', 'orchestrator')
builder.add_conditional_edges(
    'orchestrator',
    route_after_orchestrator,
    {
        'data_retrieval': 'data_retrieval',
        'end': END
    }
)
builder.add_conditional_edges(
    'data_retrieval',
    route_after_data_retrieval,
    {
        'tools': 'tools',
        'end': END
    }
)
builder.add_edge('tools', END)

graph = builder.compile()