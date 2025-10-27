import os
from dotenv import load_dotenv
from agent.agent import graph
from langchain_core.messages import AIMessage

load_dotenv()

def print_state_info(state):
    print("\n" + "="*70)
    print("PARSED QUERY INFORMATION")
    print("="*70)
    
    if state.get('use_case'):
        print(f"Use Case: {state['use_case']}")
    
    if state.get('models_requested'):
        print(f"Models Requested: {', '.join(state['models_requested'])}")
    
    if state.get('comparison_type'):
        print(f"Comparison Type: {state['comparison_type']}")
    
    if state.get('time_range'):
        print(f"Time Range: {state['time_range']}")
    
    if state.get('metrics_requested'):
        print(f"Metrics: {', '.join(state['metrics_requested'])}")
    
    if state.get('entities_requested'):
        print(f"Entities: {', '.join(state['entities_requested'])}")
    
    if state.get('execution_path'):
        print(f"Execution Path: {' → '.join(state['execution_path'])}")
    
    print(f"Next Action: {state.get('next_action', 'N/A')}")
    print("="*70 + "\n")

def main():
    print("Pharma Model Results Interpreter")
    print("Type 'quit' to exit\n")
    
    conversation_state = {
        "messages": [],
        "user_query": None,
        "parsed_intent": None,
        "use_case": None,
        "models_requested": None,
        "comparison_type": None,
        "time_range": None,
        "metrics_requested": None,
        "entities_requested": None,
        "needs_clarification": False,
        "clarification_question": None,
        "loop_count": 0,
        "next_action": None,
        "execution_path": []
    }
    
    while True:
        user_input = input("You: ").strip()
        
        if not user_input:
            continue
        
        if user_input.lower() in ['quit', 'exit']:
            print("Goodbye!")
            break
        
        conversation_state["user_query"] = user_input
        conversation_state["execution_path"] = []
        
        print("\nAgent: ", end="", flush=True)
        
        try:
            final_state = None
            for event in graph.stream(conversation_state):
                for value in event.values():
                    final_state = value
            
            if final_state:
                conversation_state.update(final_state)
                
                agent_messages = [msg for msg in final_state.get('messages', []) if isinstance(msg, AIMessage)]
                if agent_messages:
                    print(agent_messages[-1].content)
                
                print_state_info(final_state)
                
                conversation_state["loop_count"] = final_state.get("loop_count", 0)
        
        except Exception as e:
            print(f"\nError: {str(e)}")
            print("Please try again with a different query.\n")
        
        print()

if __name__ == "__main__":
    main()