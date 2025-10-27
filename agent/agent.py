from langgraph.graph import StateGraph, START, END
from agent.state import AgentState
from agent.nodes import query_understanding_agent, orchestrator_agent

def route_after_orchestrator(state: AgentState) -> str:
    next_action = state.get('next_action')
    
    if next_action == 'ask_clarification':
        return 'end'
    elif next_action == 'retrieve_data':
        return 'end'
    else:
        return 'end'

builder = StateGraph(AgentState)

builder.add_node('query_understanding', query_understanding_agent)
builder.add_node('orchestrator', orchestrator_agent)

builder.add_edge(START, 'query_understanding')
builder.add_edge('query_understanding', 'orchestrator')
builder.add_conditional_edges(
    'orchestrator',
    route_after_orchestrator,
    {
        'end': END
    }
)

graph = builder.compile()