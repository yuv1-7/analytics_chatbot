import streamlit as st
import os
import json
from dotenv import load_dotenv
from agent.agent import graph
from langchain_core.messages import AIMessage, ToolMessage, HumanMessage
from core.database import initialize_connection_pool, close_connection_pool
import plotly.graph_objects as go
from datetime import datetime
import pandas as pd
import time
import uuid
from core.session_cleanup import cleanup_old_sessions

load_dotenv()

st.set_page_config(
    page_title="Pharma Analytics Assistant",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main {
        background-color: #1a1a1a;
        padding: 2rem 4rem;
    }
    
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
    
    .section-header {
        font-size: 1.5rem;
        color: #ffffff;
        font-weight: 500;
        margin: 2rem 0 1rem 0;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
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
    
    .status-badge {
        display: inline-block;
        padding: 0.5rem 1rem;
        background-color: #d4edda;
        color: #155724;
        border-radius: 6px;
        font-size: 0.9rem;
        margin-bottom: 1rem;
    }
    
    .context-badge {
        display: inline-block;
        padding: 0.5rem 1rem;
        background-color: #fff3cd;
        color: #856404;
        border-radius: 6px;
        font-size: 0.9rem;
        margin-bottom: 1rem;
        margin-left: 1rem;
    }
    
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
    
    .streamlit-expanderHeader {
        background-color: #2a2a2a;
        color: #ffffff;
        border-radius: 6px;
        font-weight: 500;
    }
    
    [data-testid="stMetricValue"] {
        color: #ffffff;
    }
    
    [data-testid="stMetricLabel"] {
        color: #cccccc;
    }
    
    .stCodeBlock {
        background-color: #2a2a2a;
        border-radius: 6px;
    }
    
    .stProgress > div > div > div {
        background-color: #7c7ce8;
    }
    
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }
    
    .stTextArea textarea {
        background-color: #2a2a2a;
        color: #ffffff;
        border: 2px solid #555;
        border-radius: 6px;
    }
    
    .stTextArea textarea:focus {
        border-color: #7c7ce8;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #2a2a2a;
    }
    
    [data-testid="stSidebar"] .element-container {
        color: #ffffff;
    }
</style>
""", unsafe_allow_html=True)


def initialize_session_state():
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    # Initialize personalized business context
    if 'personalized_context' not in st.session_state:
        st.session_state.personalized_context = ""
    
    if 'context_last_updated' not in st.session_state:
        st.session_state.context_last_updated = None
    
    # NEW: Session ID generation (MUST be before conversation_state)
    if 'session_id' not in st.session_state:
        st.session_state.session_id = f"session_{uuid.uuid4().hex[:12]}"
        print(f"✓ New session created: {st.session_state.session_id}")
    
    # NEW: Turn counter
    if 'turn_number' not in st.session_state:
        st.session_state.turn_number = 0
    
    # NEW: User ID (can be None until login)
    if 'user_id' not in st.session_state:
        st.session_state.user_id = None
    
    # NOW initialize conversation_state (after session_id and user_id exist)
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
            "sql_retry_count": 0,
            "needs_sql_retry": False,
            "sql_error_feedback": None,
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
            "execution_path": [],
            "conversation_context": {},
            "mentioned_models": [],
            "mentioned_model_ids": [],
            "last_query_summary": None,
            "current_topic": None,
            "viz_strategy": None,
            "viz_reasoning": None,
            "viz_warnings": None,
            "clarification_attempts": 0,
            "personalized_business_context": "",
            "user_id": st.session_state.user_id,  # Now safe to reference
            # NEW: Memory fields
            "session_id": st.session_state.session_id,  # Now safe to reference
            "turn_number": 0,
            "needs_memory": False,
            "needs_database": True,
            "conversation_summaries": None,
            "summary_generated": False
        }
    
    if 'db_initialized' not in st.session_state:
        try:
            initialize_connection_pool()
            st.session_state.db_initialized = True
        except Exception as e:
            st.session_state.db_initialized = False
            st.session_state.db_error = str(e)
    
    # NEW: Run cleanup on first initialization
    if 'cleanup_done' not in st.session_state:
        try:
            print("Running session cleanup (30+ days old)...")
            result = cleanup_old_sessions(days_old=30)
            
            if result['success']:
                print(f"✓ Cleanup complete: {result['sessions_deleted']} sessions deleted")
            else:
                print(f"⚠ Cleanup failed: {result.get('error')}")
            
            st.session_state.cleanup_done = True
        except Exception as e:
            print(f"⚠ Cleanup error (non-critical): {e}")
            st.session_state.cleanup_done = True


def initialize_log_dataframe():
    if 'query_log' not in st.session_state:
        st.session_state.query_log = pd.DataFrame(columns=[
            'timestamp',
            'query',
            'response',
            'status',
            'error_type',
            'execution_path',
            'processing_time',
            'num_visualizations',
            'num_tool_results',
            'needs_clarification',
            'has_personalized_context'
        ])


def format_tool_results(messages):
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


def display_chat_message(message):
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
        
        if message.get('visualizations'):
            st.markdown("---")
            for i, chart_data in enumerate(message['visualizations'], 1):
                title = chart_data.get('title', f'Chart {i}')
                explanation = chart_data.get('explanation', '')
                figure = chart_data.get('figure')
                
                if explanation:
                    st.markdown(f"**{explanation}**")
                
                if figure:
                    st.plotly_chart(figure, use_container_width=True, key=f"chart_{id(message)}_{i}")
                else:
                    st.warning(f"⚠️ Chart {i}: {title} - No figure data available")
            st.markdown("---")
        
        if message.get('viz_warnings'):
            with st.expander("⚠️ Visualization Notes", expanded=False):
                for warning in message['viz_warnings']:
                    st.warning(warning)
        
        if message.get('tool_results'):
            with st.expander("📄 Retrieved Data", expanded=False):
                display_data_table(message['tool_results'])
        
        if message.get('state') and message['state'].get('analysis_results'):
            with st.expander("📊 Analysis Metrics", expanded=False):
                display_analysis_metrics(message['state'])
    
    elif message['type'] == 'error':
        st.error(f"⚠️ {message['content']}")


def display_data_table(results_data):
    if not results_data:
        return
    
    for i, result in enumerate(results_data, 1):
        st.write(f"**Result Set {i}**")
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Rows", result['row_count'])
        col2.metric("Columns", len(result['columns']))
        col3.metric("Showing", f"{min(5, result['total_rows'])}/{result['total_rows']}")
        
        if result['data']:
            df = pd.DataFrame(result['data'])
            st.dataframe(df, use_container_width=True)


def display_analysis_metrics(state):
    analysis_results = state.get('analysis_results')
    
    if not analysis_results:
        return
    
    computed_metrics = analysis_results.get('computed_metrics', {})
    
    if computed_metrics:
        for metric_name, values in computed_metrics.items():
            st.write(f"**{metric_name}**")
            if isinstance(values, dict):
                cols = st.columns(len(values))
                for idx, (key, val) in enumerate(values.items()):
                    if isinstance(val, float):
                        cols[idx].metric(key, f"{val:.4f}")
                    else:
                        cols[idx].metric(key, str(val))
            else:
                st.write(values)


def extract_final_insights(final_state, all_messages):
    if final_state.get('final_insights'):
        return final_state['final_insights']
    
    ai_messages = [msg for msg in all_messages if isinstance(msg, AIMessage)]
    if ai_messages:
        last_ai_msg = ai_messages[-1]
        if hasattr(last_ai_msg, 'content') and last_ai_msg.content:
            if not ("Generated SQL query" in last_ai_msg.content or 
                    "execute_sql_query" in str(last_ai_msg.content)):
                return last_ai_msg.content
    
    analysis_results = final_state.get('analysis_results')
    if analysis_results:
        summary_parts = []
        
        if analysis_results.get('computed_metrics'):
            summary_parts.append("**Key Metrics:**")
            for metric, values in list(analysis_results['computed_metrics'].items())[:3]:
                if isinstance(values, dict):
                    summary_parts.append(f"- {metric}: {values}")
                else:
                    summary_parts.append(f"- {metric}: {values}")
        
        if analysis_results.get('patterns'):
            summary_parts.append("\n**Patterns Identified:**")
            for pattern in analysis_results['patterns'][:3]:
                summary_parts.append(f"- {pattern}")
        
        if summary_parts:
            return "\n".join(summary_parts)
    
    return "Analysis completed."


def process_query(user_input):
    start_time = time.time()
    
    # Increment turn number
    st.session_state.turn_number += 1
    
    # Add personalized context to conversation state
    st.session_state.conversation_state["user_query"] = user_input
    st.session_state.conversation_state["personalized_business_context"] = st.session_state.personalized_context
    st.session_state.conversation_state["user_id"] = st.session_state.get('user_id')
    st.session_state.conversation_state["session_id"] = st.session_state.session_id
    st.session_state.conversation_state["execution_path"] = []
    st.session_state.conversation_state["rendered_charts"] = None
    st.session_state.conversation_state["visualization_specs"] = None
    st.session_state.conversation_state["viz_strategy"] = None
    st.session_state.conversation_state["viz_reasoning"] = None
    st.session_state.conversation_state["viz_warnings"] = None

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
            final_insights = extract_final_insights(final_state, all_messages)

            simplified_query = final_state.get('simplified_query', user_input)
            print(f"\n[Process Query] Original: {user_input}")
            print(f"[Process Query] Simplified: {simplified_query}\n")
            
            rendered_charts = final_state.get('rendered_charts')
            if rendered_charts is None:
                rendered_charts = []
            
            viz_strategy = final_state.get('viz_strategy')
            viz_reasoning = final_state.get('viz_reasoning')
            viz_warnings = final_state.get('viz_warnings', [])

            assistant_response = {
                'type': 'assistant',
                'content': final_insights,
                'state': final_state,
                'visualizations': rendered_charts,
                'insights': final_insights,
                'tool_results': format_tool_results(all_messages),
                'viz_strategy': viz_strategy,
                'viz_reasoning': viz_reasoning,
                'viz_warnings': viz_warnings,
                'simplified_query': simplified_query
            }

            if final_state.get('needs_clarification'):
                assistant_response['content'] = final_state.get(
                    'clarification_question', 
                    'Could you please provide more details?'
                )

            try:
                from core.memory_manager import get_memory_manager
                memory_manager = get_memory_manager()
                
                memory_manager.store_insight(
                    session_id=st.session_state.session_id,
                    turn_number=st.session_state.turn_number,
                    user_query=user_input,  # Original query
                    simplified_query=simplified_query,  # Simplified query
                    insight_text=final_insights
                )
                print(f"✓ Stored turn {st.session_state.turn_number} in memory with both queries")
            except Exception as mem_error:
                print(f"⚠ Warning: Failed to store in memory: {mem_error}")


            log_entry = {
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'query': user_input,
                'simplified_query': simplified_query,
                'response': assistant_response['content'],
                'status': 'success',
                'error_type': None,
                'execution_path': " → ".join(final_state.get('execution_path', [])),
                'processing_time': round(time.time() - start_time, 2),
                'num_visualizations': len(rendered_charts),
                'num_tool_results': len(format_tool_results(all_messages) or []),
                'needs_clarification': final_state.get('needs_clarification', False),
                'has_personalized_context': bool(st.session_state.personalized_context)
            }

            st.session_state.query_log.loc[len(st.session_state.query_log)] = log_entry

            return assistant_response

    except Exception as e:
        import traceback
        error_details = traceback.format_exc()

        log_entry = {
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'query': user_input,
            'simplified_query': None,
            'response': None,
            'status': 'error',
            'error_type': str(type(e).__name__),
            'execution_path': None,
            'processing_time': round(time.time() - start_time, 2),
            'num_visualizations': 0,
            'num_tool_results': 0,
            'needs_clarification': False,
            'has_personalized_context': bool(st.session_state.personalized_context)
        }

        st.session_state.query_log.loc[len(st.session_state.query_log)] = log_entry

        return {
            'type': 'error',
            'content': f"Error processing query: {str(e)}\n\nDetails:\n{error_details}"
        }

def show_login_screen():
    """Display login screen for user identification"""
    st.markdown("""
    <div class="app-header">
        <div class="app-title">🏥 Pharma Analytics Assistant</div>
        <div class="app-subtitle">AI-Powered Insights for Healthcare Analytics</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown("### 👤 User Identification")
    st.info("Enter a unique identifier to access your personalized analytics workspace.")
    
    user_id_input = st.text_input(
        "Your User ID (email, username, or code)",
        placeholder="e.g., john.doe@pharma.com or user123",
        help="This ID will be used to store and retrieve your personalized business context"
    )
    
    col1, col2 = st.columns([1, 3])
    with col1:
        if st.button("Continue", use_container_width=True, type="primary", disabled=not user_id_input):
            if user_id_input and user_id_input.strip():
                st.session_state.user_id = user_id_input.strip()
                st.success(f"✓ Logged in as: {st.session_state.user_id}")
                st.rerun()
            else:
                st.error("Please enter a valid User ID")
    
    st.markdown("---")
    st.markdown("""
    **Note:** 
    - No password required (authentication coming soon)
    - Your context is stored securely and retrievable using your ID
    - You can change your personalized context anytime in the sidebar
    """)

def main():
    initialize_session_state()
    initialize_log_dataframe()

    if 'user_id' not in st.session_state or not st.session_state.user_id:
        show_login_screen()
        return
    
    # Sidebar for personalized context
    # Sidebar for personalized context and user info
    with st.sidebar:
        # User info at top
        st.markdown(f"### 👤 User: `{st.session_state.user_id}`")
        if st.button("🚪 Logout", use_container_width=True):
            # Clear user session
            st.session_state.user_id = None
            st.session_state.personalized_context = ""
            st.session_state.context_last_updated = None
            st.rerun()
        
        st.markdown("---")
        
        st.markdown("### 📝 Personalized Business Context")
        st.markdown("Add your own business context to customize responses:")
        
        # Show current context summary
        try:
            from core.user_context_manager import get_user_context_manager
            context_manager = get_user_context_manager()
            
            summary = context_manager.get_user_context_summary(st.session_state.user_id)
            
            if summary.get('has_context'):
                st.success(f"✅ Active Context: {summary['chunk_count']} chunks")
                
                # Show category breakdown
                if summary.get('categories'):
                    st.markdown("**Categories:**")
                    for cat, count in summary['categories'].items():
                        st.markdown(f"- {cat}: {count} chunk(s)")
                
                # Show last updated
                if summary.get('last_updated'):
                    st.info(f"Last updated: {summary['last_updated'][:19]}")
                
                # Show total size
                if summary.get('total_chars'):
                    st.info(f"Total size: {summary['total_chars']} characters")
            else:
                st.warning("⚠ No personalized context set")
        except Exception as e:
            st.error(f"Error loading context: {e}")
        
        st.markdown("---")
        
        # Text area for context input
        new_context = st.text_area(
            "Enter your business context (free-form text)",
            value=st.session_state.get('personalized_context', ''),
            height=300,
            placeholder="""Example:
- Our company focuses on oncology products
- We launched Product X in Q3 2024
- Our main competitors are Company A and Company B
- Target markets: US Northeast, California
- Key KPIs: NRx growth, market share, HCP engagement""",
            help="Context will be automatically chunked and categorized"
        )
        
        # Save/Update button
        col1, col2 = st.columns(2)
        with col1:
            if st.button("💾 Save Context", use_container_width=True):
                if new_context and new_context.strip():
                    with st.spinner("Saving context..."):
                        try:
                            from core.user_context_manager import get_user_context_manager
                            context_manager = get_user_context_manager()
                            
                            result = context_manager.store_user_context(
                                user_id=st.session_state.user_id,
                                context_text=new_context
                            )
                            
                            if result['success']:
                                st.session_state.personalized_context = new_context
                                st.session_state.context_last_updated = result['timestamp']
                                st.success(f"✅ Saved {result['chunks_created']} chunks!")
                                st.info(f"Categories: {', '.join(result['categories'])}")
                                st.rerun()
                            else:
                                st.error(f"Failed: {result.get('error')}")
                        except Exception as e:
                            st.error(f"Error: {e}")
                else:
                    st.warning("Please enter some context first")
        
        with col2:
            if st.button("🗑️ Clear Context", use_container_width=True):
                try:
                    from core.user_context_manager import get_user_context_manager
                    context_manager = get_user_context_manager()
                    
                    result = context_manager.delete_user_context(st.session_state.user_id)
                    
                    if result['success']:
                        st.session_state.personalized_context = ""
                        st.session_state.context_last_updated = None
                        st.success("✅ Context cleared!")
                        st.rerun()
                    else:
                        st.error(f"Failed: {result.get('error')}")
                except Exception as e:
                    st.error(f"Error: {e}")
        
        # Version history expander
        st.markdown("---")
        with st.expander("📜 View Version History"):
            try:
                from core.user_context_manager import get_user_context_manager
                context_manager = get_user_context_manager()
                
                versions = context_manager.get_version_history(st.session_state.user_id)
                
                if versions:
                    # Show only active versions by default
                    active_versions = [v for v in versions if not v['is_archived']]
                    
                    if active_versions:
                        st.markdown("**Current Version:**")
                        for v in active_versions:
                            status = "🟢" if not v['is_archived'] else "🔴"
                            st.markdown(f"{status} Chunk {v['chunk_index']} v{v['version']} ({v['category']})")
                            st.caption(f"Updated: {v['updated_at'][:19]}")
                            st.caption(f"Preview: {v['chunk_preview']}")
                            st.markdown("---")
                    
                    # Show archived if requested
                    archived_versions = [v for v in versions if v['is_archived']]
                    if archived_versions:
                        show_archived = st.checkbox("Show archived versions")
                        if show_archived:
                            st.markdown("**Archived Versions:**")
                            for v in archived_versions[:10]:  # Show max 10 archived
                                st.markdown(f"🔴 Chunk {v['chunk_index']} v{v['version']} ({v['category']})")
                                st.caption(f"Archived: {v['updated_at'][:19]}")
                                st.caption(f"Preview: {v['chunk_preview']}")
                else:
                    st.info("No version history available")
            except Exception as e:
                st.error(f"Error loading versions: {e}")
        
        st.markdown("---")
        st.markdown("### 💡 Context Tips")
        st.markdown("""
        - Context is auto-chunked and categorized
        - Top 3 relevant chunks used per query
        - Updates create new versions (max 10/chunk)
        - Categories: competitors, products, markets, KPIs, HCP targeting, campaigns
        """)

    # Main content area
    st.markdown("""
    <div class="app-header">
        <div class="app-title">🏥 Pharma Analytics Assistant</div>
        <div class="app-subtitle">AI-Powered Insights for Healthcare Analytics</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Status badges
    status_col1, status_col2 = st.columns([1, 2])
    with status_col1:
        if st.session_state.db_initialized:
            st.markdown('<div class="status-badge">● Database Connected</div>', unsafe_allow_html=True)
        else:
            st.error(f"❌ Database Error: {st.session_state.get('db_error', 'Unknown error')}")
    
    with status_col2:
        if st.session_state.personalized_context:
            context_preview = st.session_state.personalized_context[:50] + "..." if len(st.session_state.personalized_context) > 50 else st.session_state.personalized_context
            st.markdown(f'<div class="context-badge">📝 Custom Context Active: {len(st.session_state.personalized_context)} chars</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="section-header">💬 Conversation</div>', unsafe_allow_html=True)
    
    if not st.session_state.chat_history:
        welcome_message = """
        👋 **Welcome to Pharma Analytics Assistant!**
        
        I can help you analyze pharmaceutical data, compare models, detect drift, and generate insights.
        
        **What I can do:**
        - ✨ Compare ML Models
        - 📊 Feature Analysis  
        - 🎯 Drift Detection
        - 📈 Generate Visualizations
        """
        
        if st.session_state.personalized_context:
            welcome_message += f"\n\n📝 **I see you've added personalized context!** I'll incorporate your business context into all my responses."
        else:
            welcome_message += f"\n\n💡 **Tip:** Add your personalized business context in the sidebar to get more customized insights!"
        
        welcome_message += "\n\nType your question below to get started!"
        
        st.info(welcome_message)
    else:
        for message in st.session_state.chat_history:
            display_chat_message(message)
    
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
    
    if st.session_state.chat_history:
        st.markdown("---")
        col_clear, col_log = st.columns([1, 3])

        with col_clear:
            if st.button("🗑️ Clear Chat", use_container_width=True):
                st.session_state.chat_history = []
                st.session_state.conversation_state = {
                    "messages": [], "user_query": None, "parsed_intent": None, "use_case": None,
                    "models_requested": None, "comparison_type": None, "time_range": None,
                    "metrics_requested": None, "entities_requested": None, "requires_visualization": False,
                    "context_documents": None, "generated_sql": None, "sql_purpose": None,
                    "expected_columns": None, "retrieved_data": None, "tool_calls": None,
                    "analysis_results": None, "visualization_specs": None, "rendered_charts": None,
                    "final_insights": None, "needs_clarification": False, "clarification_question": None,
                    "loop_count": 0, "next_action": None, "execution_path": [],
                    "conversation_context": {}, "mentioned_models": [], "mentioned_model_ids": [],
                    "last_query_summary": None, "current_topic": None, "viz_strategy": None,
                    "viz_reasoning": None, "viz_warnings": None,
                    "personalized_business_context": st.session_state.personalized_context,
                    "personalized_business_context": st.session_state.personalized_context,
                    "user_id": st.session_state.get('user_id'),
                    "session_id": st.session_state.session_id
                }
                st.rerun()

        with col_log:
            if 'query_log' in st.session_state and not st.session_state.query_log.empty:
                with st.expander("📜 Query Log History", expanded=False):
                    st.dataframe(st.session_state.query_log, use_container_width=True)
                    st.download_button(
                        label="⬇️ Download Log as CSV",
                        data=st.session_state.query_log.to_csv(index=False).encode('utf-8'),
                        file_name=f"pharma_query_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime='text/csv'
                    )


if __name__ == "__main__":
    try:
        main()
    finally:
        if st.session_state.get('db_initialized'):
            close_connection_pool()