import os
import json
from dotenv import load_dotenv
from agent.agent import graph
from langchain_core.messages import AIMessage, ToolMessage

load_dotenv()

def print_state_info(state):
    print("\n" + "="*80)
    print("PARSED QUERY INFORMATION")
    print("="*80)
    
    if state.get('use_case'):
        print(f"Use Case: {state['use_case']}")
    
    if state.get('models_requested'):
        print(f"Models: {', '.join(state['models_requested'])}")
    
    if state.get('comparison_type'):
        print(f"Comparison: {state['comparison_type']}")
    
    if state.get('time_range'):
        print(f"Time Range: {state['time_range']}")
    
    if state.get('metrics_requested'):
        print(f"Metrics: {', '.join(state['metrics_requested'])}")
    
    if state.get('entities_requested'):
        print(f"Entities: {', '.join(state['entities_requested'])}")
    
    if state.get('execution_path'):
        print(f"Path: {' → '.join(state['execution_path'])}")
    
    print(f"Next: {state.get('next_action', 'N/A')}")
    print("="*80 + "\n")

def print_tool_results(messages):
    tool_messages = [msg for msg in messages if isinstance(msg, ToolMessage)]
    
    if not tool_messages:
        return
    
    print("\n" + "="*80)
    print("RETRIEVED DATA")
    print("="*80)
    
    for i, tool_msg in enumerate(tool_messages, 1):
        try:
            result = json.loads(tool_msg.content) if isinstance(tool_msg.content, str) else tool_msg.content
            
            print(f"\nResult #{i}:")
            
            if isinstance(result, dict) and 'error' in result:
                print(f"Error: {result['error']}")
                continue
            
            if isinstance(result, dict):
                if 'models' in result and isinstance(result['models'], list):
                    print(f"Found {len(result['models'])} model(s)")
                    for model in result['models'][:3]:
                        print(f"  - {model.get('model_name', 'Unknown')} ({model.get('model_type', 'N/A')})")
                
                if 'comparison' in result:
                    print("Comparison Results:")
                    comp = result['comparison']
                    for metric, values in list(comp.items())[:5]:
                        if isinstance(values, dict):
                            print(f"  {metric}:")
                            print(f"    Ensemble: {values.get('ensemble_value', 'N/A')}")
                            print(f"    Base Avg: {values.get('base_average', 'N/A')}")
                            if values.get('improvement_vs_average') is not None:
                                print(f"    Improvement: {values['improvement_vs_average']:.2f}%")
                
                if 'top_features' in result:
                    print(f"Top Features ({len(result['top_features'])}):")
                    for feat in result['top_features'][:5]:
                        print(f"  {feat.get('rank')}. {feat.get('feature_name')} - {feat.get('importance_score', 0):.4f}")
                
                if 'drift_details' in result:
                    print(f"Drift Detection:")
                    print(f"  With drift: {result.get('models_with_drift', 0)}/{result.get('total_models_checked', 0)}")
                
                if 'top_predictions' in result:
                    print(f"Top Predictions ({len(result['top_predictions'])}):")
                    for pred in result['top_predictions'][:5]:
                        print(f"  {pred.get('entity_id')}: {pred.get('prediction_value', 'N/A')}")
                
                if 'total_results' in result:
                    print(f"Total: {result['total_results']}")
                if 'total_models' in result:
                    print(f"Total: {result['total_models']}")
            
            print()
            
        except Exception as e:
            print(f"Parse error: {e}")
            print(f"Raw: {tool_msg.content[:200]}...")
    
    print("="*80 + "\n")

def main():
    print("="*80)
    print("PHARMA MODEL RESULTS INTERPRETER")
    print("="*80)
    print("\nType 'quit' to exit\n")
    
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
        "retrieved_data": None,
        "tool_calls": None,
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
        
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("\nGoodbye!\n")
            break
        
        conversation_state["user_query"] = user_input
        conversation_state["execution_path"] = []
        conversation_state["next_action"] = None
        
        print("\nAgent: ", end="", flush=True)
        
        try:
            final_state = None
            all_messages = []
            
            for event in graph.stream(conversation_state):
                for node_name, value in event.items():
                    if 'messages' in value:
                        all_messages.extend(value['messages'])
                    final_state = value
            
            if final_state:
                conversation_state.update(final_state)
                all_messages = final_state.get('messages', [])
                
                agent_messages = [msg for msg in all_messages if isinstance(msg, AIMessage)]
                for msg in agent_messages:
                    if msg.content and not msg.tool_calls:
                        print(msg.content)
                
                tool_messages = [msg for msg in all_messages if isinstance(msg, ToolMessage)]
                if tool_messages:
                    print_tool_results(all_messages)
                
                print_state_info(final_state)
                conversation_state["loop_count"] = final_state.get("loop_count", 0)
        
        except Exception as e:
            print(f"\nError: {str(e)}")
            print(f"Type: {type(e).__name__}")
            import traceback
            traceback.print_exc()
            print("\nTry again.\n")
        
        print()

if __name__ == "__main__":
    main()