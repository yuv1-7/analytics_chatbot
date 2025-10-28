import streamlit as st
import os
import json
from dotenv import load_dotenv
from agent.agent import graph
from langchain_core.messages import AIMessage, ToolMessage, HumanMessage
from core.database import initialize_connection_pool, close_connection_pool
import plotly.graph_objects as go
from datetime import datetime

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Pharma Analytics Assistant",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Simple and Clean CSS
st.markdown("""
<style>
    /* Hide sidebar */
    [data-testid="stSidebar"] {
        display: none;
    }
    
    /* Main container */
    .main {
        background-color: #1a1a1a;
        padding: 2rem 4rem;
    }
    
    /* Header */
    .app-header {
        background-color: #e8e8e8;
        padding: 2rem;
        border-radius: 12px;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .app-title {
        font-size: 2.5rem;
        font-weight: 600;
        color: #7c7ce8;
        margin: 0;
    }
    
    .app-subtitle {
        font-size: 1rem;
        color: #666;
        margin-top: 0.5rem;
    }
    
    /* Section headers */
    .section-header {
        font-size: 1.5rem;
        color: #ffffff;
        font-weight: 500;
        margin: 2rem 0 1rem 0;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    /* Chat messages */
    .chat-message {
        padding: 1.2rem;
        margin-bottom: 1rem;
        border-radius: 8px;
        background-color: #d0e8f8;
        color: #000;
    }
    
    .message-header {
        font-weight: 600;
        margin-bottom: 0.5rem;
        font-size: 0.95rem;
    }
    
    .message-content {
        line-height: 1.6;
        font-size: 0.95rem;
    }
    
    /* Status indicator */
    .status-badge {
        display: inline-block;
        padding: 0.5rem 1rem;
        background-color: #d4edda;
        color: #155724;
        border-radius: 6px;
        font-size: 0.9rem;
        margin-bottom: 1rem;
    }
    
    /* Input section */
    .input-section {
        background-color: #2a2a2a;
        padding: 1.5rem;
        border-radius: 8px;
        margin-top: 2rem;
    }
    
    .stTextInput>div>div>input {
        background-color: #000000;
        border: 2px solid #ddd;
        border-radius: 6px;
        padding: 0.75rem;
        font-size: 0.95rem;
    }
    
    .stTextInput>div>div>input:focus {
        border-color: #7c7ce8;
        box-shadow: none;
    }
    
    /* Buttons */
    .stButton>button {
        background-color: #7c7ce8;
        color: white;
        border: none;
        border-radius: 6px;
        padding: 0.75rem 2rem;
        font-weight: 500;
        transition: background-color 0.2s;
    }
    
    .stButton>button:hover {
        background-color: #6565d8;
    }
    
    /* Expanders */
    .streamlit-expanderHeader {
        background-color: #2a2a2a;
        color: #ffffff;
        border-radius: 6px;
        font-weight: 500;
    }
    
    /* Metrics */
    [data-testid="stMetricValue"] {
        color: #ffffff;
    }
    
    [data-testid="stMetricLabel"] {
        color: #cccccc;
    }
    
    /* Code blocks */
    .stCodeBlock {
        background-color: #2a2a2a;
        border-radius: 6px;
    }
    
    /* Progress bar */
    .stProgress > div > div > div {
        background-color: #7c7ce8;
    }
    
    /* Remove default padding */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
def initialize_session_state():
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
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
            st.session_state.db_initialized = False
            st.session_state.db_error = str(e)

def format_tool_results(messages):
    """Format tool results for display"""
    tool_messages = [msg for msg in messages if isinstance(msg, ToolMessage)]
    
    if not tool_messages:
        return None
    
    results_data = []
    for tool_msg in tool_messages:
        try:
            result = json.loads(tool_msg.content) if isinstance(tool_msg.content, str) else tool_msg.content
            
            if result.get('success'):
                data = result.get('data', [])
                results_data.append({
                    'row_count': result.get('row_count', 0),
                    'columns': result.get('columns', []),
                    'data': data[:5],
                    'total_rows': len(data),
                    'truncated': result.get('truncated', False)
                })
        except Exception as e:
            st.error(f"Error parsing tool results: {e}")
    
    return results_data if results_data else None

def display_data_table(results_data):
    """Display retrieved data in tables"""
    if not results_data:
        return
    
    with st.expander("📄 Retrieved Data", expanded=False):
        for i, result in enumerate(results_data, 1):
            st.write(f"**Result Set {i}**")
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Rows", result['row_count'])
            col2.metric("Columns", len(result['columns']))
            col3.metric("Showing", f"{min(5, result['total_rows'])}/{result['total_rows']}")
            
            if result['data']:
                import pandas as pd
                df = pd.DataFrame(result['data'])
                st.dataframe(df, use_container_width=True)

def display_analysis_metrics(state):
    """Display analysis metrics"""
    analysis_results = state.get('analysis_results')
    
    if not analysis_results:
        return
    
    computed_metrics = analysis_results.get('computed_metrics', {})
    
    if computed_metrics:
        with st.expander("📈 Computed Metrics", expanded=True):
            for metric_name, values in computed_metrics.items():
                st.write(f"**{metric_name}**")
                cols = st.columns(len(values))
                for idx, (key, val) in enumerate(values.items()):
                    if isinstance(val, float):
                        cols[idx].metric(key, f"{val:.4f}")
                    else:
                        cols[idx].metric(key, str(val))

def display_visualizations(state):
    """Display rendered charts"""
    rendered_charts = state.get('rendered_charts', [])
    
    if not rendered_charts:
        return
    
    st.markdown('<div class="section-header">📊 Visualizations</div>', unsafe_allow_html=True)
    
    for i, chart_data in enumerate(rendered_charts):
        title = chart_data.get('title', f'Chart {i+1}')
        figure = chart_data.get('figure')
        
        if figure:
            st.plotly_chart(figure, use_container_width=True, key=f"chart_{i}")
        else:
            st.warning(f"No figure data available for {title}")

def process_query(user_input):
    """Process user query through the agent graph"""
    st.session_state.conversation_state["user_query"] = user_input
    st.session_state.conversation_state["execution_path"] = []
    st.session_state.conversation_state["next_action"] = None
    
    try:
        final_state = None
        all_messages = []
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        steps = []
        for event in graph.stream(st.session_state.conversation_state):
            for node_name, value in event.items():
                steps.append(node_name)
                status_text.text(f"Processing: {node_name}")
                progress_bar.progress(min(len(steps) / 10, 1.0))
                
                if 'messages' in value:
                    all_messages.extend(value['messages'])
                final_state = value
        
        progress_bar.empty()
        status_text.empty()
        
        if final_state:
            st.session_state.conversation_state.update(final_state)
            
            assistant_response = {
                'type': 'assistant',
                'content': '',
                'state': final_state,
                'visualizations': final_state.get('rendered_charts', []),
                'insights': final_state.get('final_insights', ''),
                'tool_results': format_tool_results(all_messages)
            }
            
            if final_state.get('needs_clarification'):
                assistant_response['content'] = final_state.get('clarification_question', 'Could you please provide more details?')
            else:
                assistant_response['content'] = final_state.get('final_insights', 'Analysis complete.')
            
            return assistant_response
        
    except Exception as e:
        return {
            'type': 'error',
            'content': f"Error processing query: {str(e)}"
        }

def display_chat_message(message):
    """Display a chat message"""
    if message['type'] == 'user':
        st.markdown(f"""
        <div class="chat-message">
            <div class="message-header">👤 You</div>
            <div class="message-content">{message['content']}</div>
        </div>
        """, unsafe_allow_html=True)
    
    elif message['type'] == 'assistant':
        st.markdown(f"""
        <div class="chat-message">
            <div class="message-header">🤖 Assistant</div>
            <div class="message-content">{message['content']}</div>
        </div>
        """, unsafe_allow_html=True)
        
        if message.get('tool_results'):
            display_data_table(message['tool_results'])
        
        if message.get('state'):
            display_analysis_metrics(message['state'])
        
        if message.get('visualizations'):
            display_visualizations(message['state'])
    
    elif message['type'] == 'error':
        st.error(f"⚠️ {message['content']}")

def main():
    initialize_session_state()
    
    # Header
    st.markdown("""
    <div class="app-header">
        <div class="app-title">🏥 Pharma Analytics Assistant</div>
        <div class="app-subtitle">AI-Powered Insights for Healthcare Analytics</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Database status
    if st.session_state.db_initialized:
        st.markdown('<div class="status-badge">● Database Connected</div>', unsafe_allow_html=True)
    else:
        st.error(f"❌ Database Error: {st.session_state.get('db_error', 'Unknown error')}")
    
    # Conversation section
    st.markdown('<div class="section-header">💬 Conversation</div>', unsafe_allow_html=True)
    
    if not st.session_state.chat_history:
        st.info("""
        👋 **Welcome to Pharma Analytics Assistant!**
        
        I can help you analyze pharmaceutical data, compare models, detect drift, and generate insights.
        
        **What I can do:**
        - ✨ Compare ML Models
        - 📊 Feature Analysis  
        - 🎯 Drift Detection
        - 📈 Generate Visualizations
        - 🔍 SQL Generation
        
        Type your question below to get started!
        """)
    else:
        for message in st.session_state.chat_history:
            display_chat_message(message)
    
    # Input section
    st.markdown("---")
    
    col1, col2 = st.columns([5, 1])
    
    with col1:
        user_input = st.text_input(
            "Ask a question",
            placeholder="e.g., Compare Random Forest vs XGBoost performance...",
            label_visibility="collapsed"
        )
    
    with col2:
        send_button = st.button("Send", use_container_width=True, type="primary")
    
    if send_button and user_input:
        st.session_state.chat_history.append({
            'type': 'user',
            'content': user_input
        })
        
        with st.spinner("Processing..."):
            response = process_query(user_input)
            st.session_state.chat_history.append(response)
        
        st.rerun()
    
    # Clear chat
    if st.session_state.chat_history:
        st.markdown("---")
        if st.button("🗑️ Clear Chat", use_container_width=False):
            st.session_state.chat_history = []
            st.session_state.conversation_state["messages"] = []
            st.session_state.conversation_state["loop_count"] = 0
            st.rerun()

if __name__ == "__main__":
    try:
        main()
    finally:
        if st.session_state.get('db_initialized'):
            close_connection_pool()