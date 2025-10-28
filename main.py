import os
import json
from dotenv import load_dotenv
from agent.agent import graph
from langchain_core.messages import AIMessage, ToolMessage, HumanMessage

load_dotenv()

def print_section(title, content=""):
    print(f"\n{'='*80}")
    print(f"{title}")
    print(f"{'='*80}")
    if content:
        print(content)

def print_node_output(node_name, output_data):
    print(f"\n[DEBUG] Node: {node_name}")
    print(f"{'─'*80}")
    
    if isinstance(output_data, dict):
        for key, value in output_data.items():
            if key in ['messages', 'context_documents', 'rendered_charts', 'visualization_specs']:
                if value:
                    print(f"  {key}: {len(value) if isinstance(value, list) else 'present'}")
            elif key == 'parsed_intent':
                print(f"  parsed_intent: {json.dumps(value, indent=2)}")
            elif key not in ['execution_path']:
                if value is not None and value != [] and value != {}:
                    if isinstance(value, (str, int, float, bool)):
                        print(f"  {key}: {value}")
                    elif isinstance(value, list):
                        print(f"  {key}: {len(value)} items")
                    elif isinstance(value, dict):
                        print(f"  {key}: {list(value.keys())}")
    print(f"{'─'*80}")

def format_assistant_response(state):
    messages = state.get('messages', [])
    
    ai_messages = []
    tool_results = []
    
    for msg in messages:
        if isinstance(msg, AIMessage) and msg.content and not msg.tool_calls:
            ai_messages.append(msg.content)
        elif isinstance(msg, ToolMessage):
            try:
                result = json.loads(msg.content) if isinstance(msg.content, str) else msg.content
                tool_results.append(result)
            except:
                pass
    
    response_parts = []
    
    if state.get('needs_clarification'):
        clarification = state.get('clarification_question', ai_messages[-1] if ai_messages else "Please provide more details.")
        return clarification
    
    if tool_results:
        for result in tool_results:
            if isinstance(result, dict):
                if 'error' in result:
                    response_parts.append(f"❌ {result['error']}")
                
                if 'ensemble_name' in result:
                    response_parts.append(f"📊 Analyzing: {result['ensemble_name']}")
                
                if 'comparison' in result:
                    comp = result['comparison']
                    response_parts.append("\n🔍 Performance Comparison:")
                    for metric, values in list(comp.items())[:5]:
                        if isinstance(values, dict):
                            ensemble_val = values.get('ensemble_value', 0)
                            base_avg = values.get('base_average', 0)
                            improvement = values.get('improvement_vs_average', 0)
                            response_parts.append(
                                f"  • {metric}: Ensemble {ensemble_val:.4f} vs Base Avg {base_avg:.4f} "
                                f"({improvement:+.2f}% improvement)"
                            )
                
                if 'models' in result and isinstance(result['models'], list):
                    response_parts.append(f"\n📋 Found {len(result['models'])} model(s):")
                    for model in result['models'][:5]:
                        response_parts.append(f"  • {model.get('model_name')} ({model.get('model_type')})")
                
                if 'top_features' in result:
                    response_parts.append(f"\n🎯 Top Features:")
                    for feat in result['top_features'][:10]:
                        response_parts.append(
                            f"  {feat.get('rank')}. {feat.get('feature_name')}: "
                            f"{feat.get('importance_score', 0):.4f}"
                        )
                
                if 'drift_details' in result:
                    drift_count = result.get('models_with_drift', 0)
                    total = result.get('total_models_checked', 0)
                    response_parts.append(f"\n⚠️  Drift Detection: {drift_count}/{total} models showing drift")
                    
                    for detail in result['drift_details'][:3]:
                        if detail.get('current_drift_detected'):
                            response_parts.append(
                                f"  • {detail['model_name']}: "
                                f"Drift Score {detail.get('current_drift_score', 'N/A')}"
                            )
    
    if state.get('analysis_results'):
        analysis = state['analysis_results']
        computed = analysis.get('computed_metrics', {})
        
        if computed:
            response_parts.append("\n📈 Analysis Results:")
            for metric, values in computed.items():
                if isinstance(values, dict):
                    response_parts.append(f"  • {metric}:")
                    for k, v in values.items():
                        if isinstance(v, float):
                            response_parts.append(f"    - {k}: {v:.4f}")
    
    if state.get('final_insights'):
        response_parts.append(f"\n💡 Insights:\n{state['final_insights']}")
    
    if state.get('rendered_charts'):
        charts = state['rendered_charts']
        response_parts.append(f"\n📊 {len(charts)} visualization(s) generated")
        for chart in charts:
            response_parts.append(f"  • {chart.get('title')}")
    
    if not response_parts and ai_messages:
        return ai_messages[-1]
    
    return "\n".join(response_parts) if response_parts else "Processing complete."

def main():
    print_section("PHARMA MODEL INSIGHTS CHATBOT", 
                  "Ask questions about your model results.\nType 'help' for examples, 'quit' to exit.\n")
    
    conversation_history = []
    
    while True:
        try:
            user_input = input("\n🗣️  You: ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("\n👋 Goodbye!\n")
                break
            
            if user_input.lower() == 'help':
                print("\n📚 Example Queries:")
                examples = [
                    "Compare Random Forest vs XGBoost",
                    "Show ensemble vs base model performance",
                    "What are top features for NRx model?",
                    "Has the model drifted?",
                    "Compare version 1.0 to 2.0",
                    "Show me drift detection results",
                    "What models do we have for HCP engagement?"
                ]
                for ex in examples:
                    print(f"  • {ex}")
                continue
            
            conversation_history.append(HumanMessage(content=user_input))
            
            state = {
                "messages": conversation_history.copy(),
                "user_query": user_input,
                "parsed_intent": None,
                "use_case": None,
                "models_requested": None,
                "comparison_type": None,
                "time_range": None,
                "metrics_requested": None,
                "entities_requested": None,
                "requires_visualization": False,
                "context_documents": None,
                "retrieved_data": None,
                "tool_calls": None,
                "analysis_results": None,
                "visualization_specs": None,
                "rendered_charts": None,
                "final_insights": None,
                "needs_clarification": False,
                "clarification_question": None,
                "loop_count": 0,
                "next_action": None,
                "execution_path": []
            }
            
            print_section("PROCESSING")
            
            final_state = None
            
            for event in graph.stream(state):
                for node_name, node_output in event.items():
                    print(f"→ {node_name}")
                    print_node_output(node_name, node_output)
                    final_state = node_output
            
            if final_state:
                response = format_assistant_response(final_state)
                print_section("ASSISTANT RESPONSE")
                print(f"\n🤖 Assistant:\n{response}\n")
                
                if final_state.get('messages'):
                    conversation_history = final_state['messages']
                
                if final_state.get('rendered_charts'):
                    for chart_data in final_state['rendered_charts']:
                        try:
                            chart_data['figure'].show()
                        except:
                            pass
            else:
                print("\n⚠️  No response generated. Try rephrasing your query.\n")
        
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!\n")
            break
        except Exception as e:
            print(f"\n❌ Error: {str(e)}")
            print("Try rephrasing your query or type 'help' for examples.\n")

if __name__ == "__main__":
    main()