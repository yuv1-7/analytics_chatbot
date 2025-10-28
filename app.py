import streamlit as st
import os
import json
from dotenv import load_dotenv
from agent.agent import graph
from langchain_core.messages import AIMessage, ToolMessage, HumanMessage
from core.database import initialize_connection_pool, close_connection_pool
import plotly.graph_objects as go

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Pharma Analytics Assistant",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling with improved visibility
st.markdown("""
<style>
    /* Main container */
    .main {
        background-color: #f8f9fa;
    }
    
    /* Header styling */
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1a1a1a;
        text-align: center;
        margin-bottom: 2rem;
        padding: 1rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    /* Chat message containers */
    .chat-message {
        padding: 1.25rem;
        border-radius: 0.75rem;
        margin-bottom: 1rem;
        display: flex;
        flex-direction: column;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        line-height: 1.6;
    }
    
    .user-message {
        background-color: #e3f2fd;
        border-left: 5px solid #1976d2;
        color: #1a1a1a;
    }
    
    .user-message strong {
        color: #1565c0;
        font-size: 1rem;
        font-weight: 600;
    }
    
    .assistant-message {
        background-color: #ffffff;
        border-left: 5px solid #43a047;
        color: #2c3e50;
    }
    
    .assistant-message strong {
        color: #2e7d32;
        font-size: 1rem;
        font-weight: 600;
    }
    
    /* Message content text */
    .message-content {
        color: #2c3e50;
        font-size: 0.95rem;
        margin-top: 0.5rem;
        white-space: pre-wrap;
        word-wrap: break-word;
    }
    
    /* Info boxes with better contrast */
    .info-box {
        background-color: #e3f2fd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1976d2;
        margin: 1rem 0;
        color: #1a1a1a;
        font-weight: 500;
    }
    
    .error-box {
        background-color: #ffebee;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #d32f2f;
        margin: 1rem 0;
        color: #1a1a1a;
        font-weight: 500;
    }
    
    .success-box {
        background-color: #e8f5e9;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #43a047;
        margin: 1rem 0;
        color: #1a1a1a;
        font-weight: 500;
    }
    
    /* Sidebar styling */
    .css-1d391kg, [data-testid="stSidebar"] {
        background-color: #ffffff;
    }
    
    /* Sidebar text */
    .css-1d391kg p, [data-testid="stSidebar"] p {
        color: #2c3e50;
        font-size: 0.9rem;
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background-color: #f5f5f5;
        color: #1a1a1a;
        font-weight: 600;
        font-size: 0.95rem;
    }
    
    /* Button styling */
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 0.5rem;
        padding: 0.6rem 1rem;
        border: none;
        font-weight: 600;
        font-size: 0.95rem;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        background: linear-gradient(135deg, #5568d3 0%, #5f3d85 100%);
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
    }
    
    /* Input field styling */
    .stTextInput>div>div>input {
        background-color: #ffffff;
        color: #1a1a1a;
        border: 2px solid #e0e0e0;
        border-radius: 0.5rem;
        padding: 0.75rem;
        font-size: 0.95rem;
    }
    
    .stTextInput>div>div>input:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 2px rgba(102, 126, 234, 0.2);
    }
    
    /* Metric cards */
    .metric-card {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 6px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
        border: 1px solid #e0e0e0;
    }
    
    /* Data table styling */
    .dataframe {
        font-size: 0.85rem;
        color: #2c3e50;
    }
    
    /* Code blocks */
    code {
        background-color: #f5f5f5;
        color: #d32f2f;
        padding: 0.2rem 0.4rem;
        border-radius: 0.25rem;
        font-size: 0.9rem;
    }
    
    /* Headers in content */
    h1, h2, h3, h4, h5, h6 {
        color: #1a1a1a;
        font-weight: 600;
    }
    
    /* Markdown text */
    .markdown-text-container {
        color: #2c3e50;
        font-size: 0.95rem;
    }
    
    /* Selectbox */
    .stSelectbox>div>div {
        background-color: #ffffff;
        color: #1a1a1a;
    }
    
    /* Progress bar */
    .stProgress > div > div > div > div {
        background-color: #667eea;
    }
    
    /* Footer */
    .footer-text {
        color: #6c757d;
        font-size: 0.85rem;
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
                    'data': data[:5],  # First 5 rows
                    'total_rows': len(data),
                    'truncated': result.get('truncated', False)
                })
        except Exception as e:
            st.error(f"Error parsing tool results: {e}")
    
    return results_data if results_data else None

def display_parsed_intent(state):
    """Display parsed query information in sidebar"""
    if not state.get('parsed_intent'):
        return
    
    with st.sidebar.expander("📊 Query Analysis", expanded=True):
        if state.get('use_case'):
            st.markdown(f"**Use Case:** `{state['use_case']}`")
        
        if state.get('models_requested'):
            st.markdown(f"**Models:** `{', '.join(state['models_requested'])}`")
        
        if state.get('comparison_type'):
            st.markdown(f"**Comparison:** `{state['comparison_type']}`")
        
        if state.get('metrics_requested'):
            st.markdown(f"**Metrics:** `{', '.join(state['metrics_requested'])}`")
        
        if state.get('requires_visualization'):
            st.markdown("**Visualization:** ✅ Required")
        
        if state.get('execution_path'):
            path_str = ' → '.join(state['execution_path'])
            st.markdown(f"**Execution Path:**")
            st.code(path_str, language=None)

def display_sql_query(state):
    """Display generated SQL query"""
    if state.get('generated_sql'):
        with st.sidebar.expander("🔍 Generated SQL", expanded=False):
            st.markdown(f"**Purpose:** {state.get('sql_purpose', 'N/A')}")
            st.code(state['generated_sql'], language='sql')

def display_data_table(results_data):
    """Display retrieved data in tables"""
    if not results_data:
        return
    
    with st.expander("📄 Retrieved Data", expanded=False):
        for i, result in enumerate(results_data, 1):
            st.markdown(f"### Result Set {i}")
            st.info(f"Retrieved {result['row_count']} rows")
            
            if result['data']:
                import pandas as pd
                df = pd.DataFrame(result['data'])
                st.dataframe(df, use_container_width=True, height=300)
                
                if result['total_rows'] > 5:
                    st.caption(f"📊 Showing first 5 of {result['total_rows']} total rows")

def display_analysis_metrics(state):
    """Display analysis metrics"""
    analysis_results = state.get('analysis_results')
    
    if not analysis_results:
        return
    
    computed_metrics = analysis_results.get('computed_metrics', {})
    
    if computed_metrics:
        with st.expander("📈 Computed Metrics", expanded=True):
            num_metrics = len(computed_metrics)
            cols = st.columns(min(num_metrics, 3))
            
            for idx, (metric_name, values) in enumerate(computed_metrics.items()):
                with cols[idx % 3]:
                    st.markdown(f"### {metric_name}")
                    for key, val in values.items():
                        if isinstance(val, float):
                            st.metric(label=key, value=f"{val:.4f}")
                        else:
                            st.metric(label=key, value=str(val))

def display_visualizations(state):
    """Display rendered charts"""
    rendered_charts = state.get('rendered_charts', [])
    
    if not rendered_charts:
        return
    
    st.markdown("## 📊 Visualizations")
    
    for i, chart_data in enumerate(rendered_charts):
        title = chart_data.get('title', f'Chart {i+1}')
        figure = chart_data.get('figure')
        
        st.markdown(f"### {title}")
        if figure:
            st.plotly_chart(figure, use_container_width=True, key=f"chart_{i}")
        else:
            st.warning(f"⚠️ No figure data available for {title}")

def process_query(user_input):
    """Process user query through the agent graph"""
    # Update conversation state
    st.session_state.conversation_state["user_query"] = user_input
    st.session_state.conversation_state["execution_path"] = []
    st.session_state.conversation_state["next_action"] = None
    
    try:
        final_state = None
        all_messages = []
        
        # Create a progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Stream through the graph
        steps = []
        for event in graph.stream(st.session_state.conversation_state):
            for node_name, value in event.items():
                steps.append(node_name)
                status_text.markdown(f"**⚙️ Executing:** `{node_name}`")
                progress_bar.progress(min(len(steps) / 10, 1.0))  # Approximate progress
                
                if 'messages' in value:
                    all_messages.extend(value['messages'])
                final_state = value
        
        progress_bar.progress(100)
        status_text.empty()
        progress_bar.empty()
        
        if final_state:
            st.session_state.conversation_state.update(final_state)
            
            # Extract assistant response
            assistant_response = {
                'type': 'assistant',
                'content': '',
                'state': final_state,
                'visualizations': final_state.get('rendered_charts', []),
                'insights': final_state.get('final_insights', ''),
                'tool_results': format_tool_results(all_messages)
            }
            
            # Get insights or clarification
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
        <div class="chat-message user-message">
            <strong>👤 You</strong>
            <div class="message-content">{message['content']}</div>
        </div>
        """, unsafe_allow_html=True)
    
    elif message['type'] == 'assistant':
        st.markdown(f"""
        <div class="chat-message assistant-message">
            <strong>🤖 Assistant</strong>
            <div class="message-content">{message['content']}</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Display additional information
        if message.get('tool_results'):
            display_data_table(message['tool_results'])
        
        if message.get('state'):
            display_analysis_metrics(message['state'])
        
        if message.get('visualizations'):
            display_visualizations(message['state'])
    
    elif message['type'] == 'error':
        st.markdown(f"""
        <div class="error-box">
            <strong>⚠️ Error</strong><br>
            {message['content']}
        </div>
        """, unsafe_allow_html=True)

def main():
    initialize_session_state()
    
    # Header
    st.markdown('<div class="main-header">🏥 Pharma Analytics Assistant</div>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("# 🔬 Analytics Control")
        st.markdown("---")
        
        # Database status
        if st.session_state.db_initialized:
            st.markdown('<div class="success-box">✅ Database Connected</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="error-box">❌ Database Error<br><small>{st.session_state.get("db_error", "Unknown error")}</small></div>', unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Quick actions
        st.markdown("### 🚀 Quick Start")
        
        example_queries = [
            "Compare Random Forest vs XGBoost for NRx forecasting",
            "Show ensemble vs base model performance",
            "Top features for NRx model",
            "Check for model drift",
            "Compare version 1.0 to 2.0"
        ]
        
        selected_example = st.selectbox(
            "Select an example query:",
            ["Choose an example..."] + example_queries,
            key="example_query"
        )
        
        if selected_example != "Choose an example..." and st.button("▶️ Run Example", use_container_width=True):
            st.session_state.chat_history.append({
                'type': 'user',
                'content': selected_example
            })
            
            with st.spinner("🔄 Processing..."):
                response = process_query(selected_example)
                st.session_state.chat_history.append(response)
            
            st.rerun()
        
        st.markdown("---")
        
        # Display parsed intent if available
        if st.session_state.chat_history:
            last_response = st.session_state.chat_history[-1]
            if last_response.get('type') == 'assistant' and last_response.get('state'):
                display_parsed_intent(last_response['state'])
                display_sql_query(last_response['state'])
        
        st.markdown("---")
        
        # Clear chat button
        if st.button("🗑️ Clear Chat History", use_container_width=True):
            st.session_state.chat_history = []
            st.session_state.conversation_state["messages"] = []
            st.session_state.conversation_state["loop_count"] = 0
            st.rerun()
        
        # Help section
        with st.expander("ℹ️ Help & Guide"):
            st.markdown("""
            **Available Use Cases:**
            - 🔮 NRx Forecasting
            - 👥 HCP Engagement
            - 📊 Feature Importance Analysis
            - 🎯 Model Drift Detection
            - 💬 Messaging Optimization
            
            **Query Examples:**
            - Compare models for [use case]
            - Show me [metric] for [model]
            - What are the top features?
            - Has the model drifted?
            - Compare version X to Y
            """)
    
    # Main chat area
    st.markdown("## 💬 Conversation")
    chat_container = st.container()
    
    with chat_container:
        if not st.session_state.chat_history:
            st.markdown("""
            <div class="info-box">
                <strong>👋 Welcome!</strong><br>
                I'm your Pharma Analytics Assistant. Ask me about model performance, feature importance, 
                drift detection, or ensemble comparisons. Select an example from the sidebar to get started!
            </div>
            """, unsafe_allow_html=True)
        else:
            # Display chat history
            for message in st.session_state.chat_history:
                display_chat_message(message)
    
    # Chat input at the bottom
    st.markdown("---")
    st.markdown("### ✍️ Ask a Question")
    
    col1, col2 = st.columns([5, 1])
    
    with col1:
        user_input = st.text_input(
            "Type your question here...",
            placeholder="e.g., Compare Random Forest vs XGBoost performance...",
            key="user_input",
            label_visibility="collapsed"
        )
    
    with col2:
        send_button = st.button("📤 Send", use_container_width=True, type="primary")
    
    # Process input
    if send_button and user_input:
        # Add user message to history
        st.session_state.chat_history.append({
            'type': 'user',
            'content': user_input
        })
        
        # Process query
        with st.spinner("🔄 Processing your query..."):
            response = process_query(user_input)
            st.session_state.chat_history.append(response)
        
        # Clear input and rerun
        st.rerun()
    
    # Footer
    st.markdown("---")
    st.markdown(
        '<div class="footer-text" style="text-align: center;">'
        '⚡ Powered by LangGraph & Google Gemini | 🏥 Pharma Analytics Platform'
        '</div>',
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    try:
        main()
    finally:
        # Cleanup on app shutdown
        if st.session_state.get('db_initialized'):
            close_connection_pool()