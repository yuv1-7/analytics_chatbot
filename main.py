import os
import json
from dotenv import load_dotenv
from agent.agent import graph
from langchain_core.messages import AIMessage, ToolMessage

load_dotenv()


def print_state_info(state):
    """Print parsed query information and execution path"""
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
    
    if state.get('requires_visualization'):
        print(f"Visualization Required: Yes")
    
    if state.get('execution_path'):
        print(f"Path: {' → '.join(state['execution_path'])}")
    
    print(f"Next: {state.get('next_action', 'N/A')}")
    print("="*80 + "\n")


def print_tool_results(messages):
    """Print data retrieved from tools"""
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


def print_analysis_results(state):
    """Print analysis and computation results"""
    analysis_results = state.get('analysis_results')
    
    if not analysis_results:
        return
    
    print("\n" + "="*80)
    print("ANALYSIS RESULTS")
    print("="*80)
    
    computed_metrics = analysis_results.get('computed_metrics', {})
    if computed_metrics:
        print("\nComputed Metrics:")
        for metric_name, values in computed_metrics.items():
            print(f"  {metric_name}:")
            for key, val in values.items():
                if isinstance(val, float):
                    print(f"    {key}: {val:.4f}")
                else:
                    print(f"    {key}: {val}")
    
    trends = analysis_results.get('trends', [])
    if trends:
        print(f"\nTrends Identified: {len(trends)}")
        for trend in trends[:3]:
            print(f"  - {trend}")
    
    anomalies = analysis_results.get('anomalies', [])
    if anomalies:
        print(f"\nAnomalies Detected: {len(anomalies)}")
        for anomaly in anomalies[:3]:
            print(f"  - {anomaly}")
    
    print("="*80 + "\n")


def display_visualizations(state):
    """Display rendered charts"""
    rendered_charts = state.get('rendered_charts', [])
    
    if not rendered_charts:
        return
    
    print("\n" + "="*80)
    print("VISUALIZATIONS")
    print("="*80)
    
    for i, chart_data in enumerate(rendered_charts, 1):
        title = chart_data.get('title', f'Chart {i}')
        chart_type = chart_data.get('type', 'unknown')
        figure = chart_data.get('figure')
        
        print(f"\n{i}. {title} ({chart_type})")
        
        if figure:
            try:
                # Display the chart
                figure.show()
                print(f"   ✓ Chart displayed successfully")
            except Exception as e:
                print(f"   ✗ Failed to display: {e}")
        else:
            print(f"   ✗ No figure data available")
    
    print("\n" + "="*80 + "\n")


def print_final_insights(state):
    """Print the final narrative insights"""
    insights = state.get('final_insights')
    
    if not insights:
        return
    
    print("\n" + "="*80)
    print("INSIGHTS & RECOMMENDATIONS")
    print("="*80)
    print(f"\n{insights}\n")
    print("="*80 + "\n")


def main():
    print("="*80)
    print("PHARMA MODEL RESULTS INTERPRETER")
    print("="*80)
    print("\nType 'quit' to exit")
    print("Type 'help' for example queries\n")
    
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
    
    while True:
        user_input = input("You: ").strip()
        
        if not user_input:
            continue
        
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("\nGoodbye!\n")
            break
        
        if user_input.lower() == 'help':
            print("\nExample Queries:")
            print("  - Compare Random Forest vs XGBoost for NRx forecasting")
            print("  - Show me ensemble vs base model performance")
            print("  - What are the top features for the NRx model?")
            print("  - Has the model drifted recently?")
            print("  - Display prediction trends over time")
            print("  - Compare version 1.0 to version 2.0 of the HCP model\n")
            continue
        
        # Reset state for new query
        conversation_state["user_query"] = user_input
        conversation_state["execution_path"] = []
        conversation_state["next_action"] = None
        conversation_state["requires_visualization"] = False
        conversation_state["analysis_results"] = None
        conversation_state["rendered_charts"] = None
        conversation_state["final_insights"] = None
        
        print("\n" + "="*80)
        print("PROCESSING QUERY")
        print("="*80 + "\n")
        
        try:
            final_state = None
            all_messages = []
            
            # Stream through the graph
            for event in graph.stream(conversation_state):
                for node_name, value in event.items():
                    print(f"→ Executing: {node_name}")
                    if 'messages' in value:
                        all_messages.extend(value['messages'])
                    final_state = value
            
            if final_state:
                conversation_state.update(final_state)
                all_messages = final_state.get('messages', [])
                
                # Print clarification messages if any
                agent_messages = [msg for msg in all_messages if isinstance(msg, AIMessage)]
                for msg in agent_messages:
                    if msg.content and not msg.tool_calls:
                        if final_state.get('needs_clarification'):
                            print("\nAgent: " + msg.content)
                
                # Only print detailed results if not asking for clarification
                if not final_state.get('needs_clarification'):
                    # Print tool results
                    tool_messages = [msg for msg in all_messages if isinstance(msg, ToolMessage)]
                    if tool_messages:
                        print_tool_results(all_messages)
                    
                    # Print analysis results
                    print_analysis_results(final_state)
                    
                    # Display visualizations
                    display_visualizations(final_state)
                    
                    # Print final insights
                    print_final_insights(final_state)
                
                # Print state info
                print_state_info(final_state)
                
                # Update loop count for conversation continuity
                conversation_state["loop_count"] = final_state.get("loop_count", 0)
        
        except Exception as e:
            print(f"\n{'='*80}")
            print("ERROR")
            print("="*80)
            print(f"Error: {str(e)}")
            print(f"Type: {type(e).__name__}")
            import traceback
            traceback.print_exc()
            print(f"\n{'='*80}\n")
            print("Try rephrasing your query or type 'help' for examples.\n")
        
        print()


if __name__ == "__main__":
    main()