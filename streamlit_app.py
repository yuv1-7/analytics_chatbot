import os
import json
import streamlit as st
from dotenv import load_dotenv
from agent.agent import graph
from langchain_core.messages import AIMessage, ToolMessage, HumanMessage
from core.database import initialize_connection_pool, close_connection_pool
import plotly.graph_objects as go

load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Pharma Model Results Interpreter",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem 0;
        border-bottom: 3px solid #1f77b4;
        margin-bottom: 2rem;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .user-message {
        background-color: #e3f2fd;
        border-left: 4px solid #2196f3;
    }
    .agent-message {
        background-color: #f5f5f5;
        border-left: 4px solid #4caf50;
    }
    .info-box {
        background-color: #fff3cd;
        border: 1px solid #ffc107;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #28a745;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
    .metric-card {
        background-color: #ffffff;
        border: 1px solid #ddd;
        border-radius: 0.5rem;
        padding: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []

if 'conversation_state' not in st.session_state:
    st.session_state.conversation_state = {
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
        "generated_sql": None,
        "sql_purpose": None,
        "expected_columns": None,
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

if 'db_initialized' not in st.session_state:
    try:
        initialize_connection_pool()
        st.session_state.db_initialized = True
    except Exception as e:
        st.error(f"Failed to initialize database: {e}")
        st.session_state.db_initialized = False


def extract_tool_results(messages):
    """Extract data from tool messages"""
    results = []
    for msg in messages:
        if isinstance(msg, ToolMessage):
            try:
                result = json.loads(msg.content) if isinstance(msg.content, str) else msg.content
                results.append(result)
            except:
                continue
    return results


def display_parsed_intent(state):
    """Display parsed query information in sidebar"""
    with st.sidebar:
        st.subheader("🎯 Query Analysis")
        
        if state.get('use_case'):
            st.markdown(f"**Use Case:** {state['use_case']}")
        
        if state.get('models_requested'):
            st.markdown(f"**Models:** {', '.join(state['models_requested'])}")
        
        if state.get('comparison_type'):
            st.markdown(f"**Comparison:** {state['comparison_type']}")
        
        if state.get('time_range'):
            st.markdown(f"**Time Range:** {state['time_range']}")
        
        if state.get('metrics_requested'):
            st.markdown(f"**Metrics:** {', '.join(state['metrics_requested'])}")
        
        if state.get('requires_visualization'):
            st.markdown("**Visualization:** ✓ Required")
        
        if state.get('execution_path'):
            st.markdown("---")
            st.markdown("**Execution Path:**")
            for i, step in enumerate(state['execution_path'], 1):
                st.markdown(f"{i}. {step}")


def display_data_results(tool_results):
    """Display retrieved data in expandable sections"""
    if not tool_results:
        return
    
    st.subheader("📊 Retrieved Data")
    
    for i, result in enumerate(tool_results, 1):
        with st.expander(f"Query Result #{i}", expanded=(i == 1)):
            if not result.get('success'):
                st.error(f"❌ Error: {result.get('error', 'Unknown error')}")
                continue
            
            data = result.get('data', [])
            row_count = result.get('row_count', 0)
            columns = result.get('columns', [])
            truncated = result.get('truncated', False)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Rows Retrieved", row_count)
            with col2:
                st.metric("Columns", len(columns))
            with col3:
                if truncated:
                    st.warning("⚠️ Truncated")
                else:
                    st.success("✓ Complete")
            
            if data:
                import pandas as pd
                df = pd.DataFrame(data)
                st.dataframe(df, use_container_width=True, height=300)


def display_analysis_results(state):
    """Display analysis and computed metrics"""
    analysis_results = state.get('analysis_results')
    
    if not analysis_results:
        return
    
    st.subheader("🔍 Analysis Results")
    
    computed_metrics = analysis_results.get('computed_metrics', {})
    if computed_metrics:
        st.markdown("**Computed Metrics:**")
        
        cols = st.columns(min(len(computed_metrics), 3))
        for idx, (metric_name, values) in enumerate(computed_metrics.items()):
            with cols[idx % 3]:
                with st.container():
                    st.markdown(f"**{metric_name}**")
                    for key, val in values.items():
                        if isinstance(val, float):
                            st.markdown(f"- {key}: `{val:.4f}`")
                        else:
                            st.markdown(f"- {key}: `{val}`")
    
    trends = analysis_results.get('trends', [])
    if trends:
        with st.expander(f"📈 Trends Identified ({len(trends)})"):
            for trend in trends:
                st.markdown(f"- {trend}")
    
    anomalies = analysis_results.get('anomalies', [])
    if anomalies:
        with st.expander(f"⚠️ Anomalies Detected ({len(anomalies)})"):
            for anomaly in anomalies:
                st.markdown(f"- {anomaly}")


def display_visualizations(state):
    """Display rendered charts"""
    rendered_charts = state.get('rendered_charts', [])
    
    if not rendered_charts:
        return
    
    st.subheader("📈 Visualizations")
    
    for i, chart_data in enumerate(rendered_charts):
        title = chart_data.get('title', f'Chart {i+1}')
        figure = chart_data.get('figure')
        
        if figure:
            st.plotly_chart(figure, use_container_width=True, key=f"chart_{i}")
        else:
            st.warning(f"No figure data available for: {title}")


def process_query(user_input):
    """Process user query through the agent graph"""
    # Reset state for new query
    st.session_state.conversation_state["user_query"] = user_input
    st.session_state.conversation_state["execution_path"] = []
    st.session_state.conversation_state["next_action"] = None
    st.session_state.conversation_state["requires_visualization"] = False
    st.session_state.conversation_state["analysis_results"] = None
    st.session_state.conversation_state["rendered_charts"] = None
    st.session_state.conversation_state["final_insights"] = None
    st.session_state.conversation_state["generated_sql"] = None
    st.session_state.conversation_state["sql_purpose"] = None
    
    with st.spinner("Processing your query..."):
        try:
            final_state = None
            all_messages = []
            
            # Progress container
            progress_container = st.container()
            
            with progress_container:
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                steps = []
                for event in graph.stream(st.session_state.conversation_state):
                    for node_name, value in event.items():
                        steps.append(node_name)
                        status_text.text(f"Executing: {node_name}")
                        progress_bar.progress(len(steps) / 10)  # Approximate progress
                        
                        if 'messages' in value:
                            all_messages.extend(value['messages'])
                        final_state = value
                
                progress_bar.progress(1.0)
                status_text.text("✓ Processing complete")
            
            if final_state:
                st.session_state.conversation_state.update(final_state)
                all_messages = final_state.get('messages', [])
                
                # Display parsed intent in sidebar
                display_parsed_intent(final_state)
                
                # Check for clarification requests
                if final_state.get('needs_clarification'):
                    agent_messages = [msg for msg in all_messages if isinstance(msg, AIMessage)]
                    for msg in agent_messages:
                        if msg.content and not msg.tool_calls:
                            return msg.content, final_state
                
                # Display SQL query if generated
                if final_state.get('generated_sql'):
                    with st.expander("🔧 Generated SQL Query", expanded=False):
                        st.markdown(f"**Purpose:** {final_state.get('sql_purpose', 'N/A')}")
                        st.code(final_state['generated_sql'], language='sql')
                
                # Display retrieved data
                tool_results = extract_tool_results(all_messages)
                if tool_results:
                    display_data_results(tool_results)
                
                # Display analysis results
                display_analysis_results(final_state)
                
                # Display visualizations
                display_visualizations(final_state)
                
                # Return final insights
                insights = final_state.get('final_insights', 'Analysis complete.')
                return insights, final_state
            
            return "Query processed but no results generated.", None
        
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            return None, None


# Main UI
st.markdown('<div class="main-header">🔬 Pharma Model Results Interpreter</div>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.title("💡 Help & Examples")
    
    st.markdown("### Example Queries:")
    examples = [
        "Compare Random Forest vs XGBoost for NRx forecasting",
        "Show me ensemble vs base model performance",
        "What are the top features for the NRx model?",
        "Has the model drifted recently?",
        "Display prediction trends over time",
        "Compare version 1.0 to version 2.0 of the HCP model",
        "Find all models using XGBoost algorithm",
        "Show drift detection for last month"
    ]
    
    for example in examples:
        if st.button(example, key=f"ex_{example[:20]}", use_container_width=True):
            st.session_state.current_input = example
    
    st.markdown("---")
    
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.session_state.conversation_state["loop_count"] = 0
        st.rerun()
    
    st.markdown("---")
    
    # Database status
    if st.session_state.db_initialized:
        st.success("✓ Database Connected")
    else:
        st.error("❌ Database Not Connected")

# Chat interface
chat_container = st.container()

with chat_container:
    # Display chat history
    for message in st.session_state.messages:
        role = message["role"]
        content = message["content"]
        
        if role == "user":
            st.markdown(f'<div class="chat-message user-message"><strong>You:</strong><br>{content}</div>', 
                       unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="chat-message agent-message"><strong>Agent:</strong><br>{content}</div>', 
                       unsafe_allow_html=True)

# Input area
input_container = st.container()

with input_container:
    col1, col2 = st.columns([6, 1])
    
    with col1:
        user_input = st.text_input(
            "Ask a question about your models:",
            key="user_input",
            placeholder="e.g., Compare Random Forest vs XGBoost performance...",
            label_visibility="collapsed",
            value=st.session_state.get('current_input', '')
        )
    
    with col2:
        submit_button = st.button("Send", type="primary", use_container_width=True)

# Handle example button clicks
if 'current_input' in st.session_state and st.session_state.current_input:
    user_input = st.session_state.current_input
    submit_button = True
    del st.session_state.current_input

# Process input
if submit_button and user_input:
    if not st.session_state.db_initialized:
        st.error("Database connection not initialized. Please check your configuration.")
    else:
        # Add user message to history
        st.session_state.messages.append({"role": "user", "content": user_input})
        
        # Process query
        response, final_state = process_query(user_input)
        
        # Add agent response to history
        if response:
            st.session_state.messages.append({"role": "agent", "content": response})
        
        # Update loop count
        if final_state:
            st.session_state.conversation_state["loop_count"] = final_state.get("loop_count", 0)
        
        st.rerun()

# Footer
st.markdown("---")
st.markdown(
    '<div style="text-align: center; color: #666; padding: 1rem;">Powered by LangGraph & Google Gemini</div>',
    unsafe_allow_html=True
)