import os
import json
import re
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from agent.state import AgentState
from agent.tools import ALL_TOOLS, execute_sql_query
from core.schema_context import SCHEMA_CONTEXT
from core.placeholders_filler import fill_template_placeholders
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from agent.drift_query_helper import (
    detect_drift_query_intent,
    should_use_drift_sql,
    get_drift_sql_query
)

llm = ChatOpenAI(
    model='gpt-4.1-mini',
    api_key=os.getenv("gpt_api_key"),
    stream_usage=True,
    temperature=0.7
)

llm_with_tools = llm.bind_tools(ALL_TOOLS)

def get_personalized_context_section(state: AgentState) -> str:
    """
    Extract and format personalized business context for inclusion in prompts.
    Now retrieves from user-specific chunked context in Pinecone.
    
    Args:
        state: Current agent state
    
    Returns:
        Formatted context section or empty string
    """
    user_id = state.get('user_id')
    user_query = state.get('user_query', '')
    
    if not user_id:
        return ""
    
    try:
        from core.user_context_manager import get_user_context_manager
        
        context_manager = get_user_context_manager()
        
        # Retrieve top 3 semantically relevant chunks
        chunks = context_manager.retrieve_user_context(
            user_id=user_id,
            query=user_query,
            top_k=3
        )
        
        if not chunks:
            return ""
        
        # Format chunks
        context_parts = []
        context_parts.append("### PERSONALIZED BUSINESS CONTEXT (CRITICAL - USER PROVIDED):")
        context_parts.append("The user has provided the following specific business context that MUST be considered:")
        context_parts.append("")
        
        for i, chunk in enumerate(chunks, 1):
            category = chunk['category'].upper()
            text = chunk['chunk_text']
            relevance = chunk['relevance']
            
            context_parts.append(f"**Context Chunk {i} ({category}) [Relevance: {relevance:.2f}]:**")
            context_parts.append(text)
            context_parts.append("")
        
        context_parts.append("IMPORTANT: This personalized context takes precedence over generic knowledge.")
        context_parts.append("Use this context to:")
        context_parts.append("- Customize your analysis and recommendations")
        context_parts.append("- Reference specific products, competitors, or markets mentioned")
        context_parts.append("- Align your insights with the user's business priorities")
        context_parts.append("- Provide relevant examples from their context")
        
        return "\n".join(context_parts)
        
    except Exception as e:
        print(f"Warning: Failed to retrieve user context: {e}")
        return ""

class ParsedIntent(BaseModel):
    use_case: Optional[str] = Field(
        default="all_use_cases",
        description="One of: NRx_forecasting, HCP_engagement, feature_importance_analysis, model_drift_detection, messaging_optimization"
    )
    models_requested: Optional[List[str]] = Field(
        default=None,
        description="List of model names or types mentioned (e.g., ['Random Forest', 'XGBoost', 'ensemble'])"
    )
    comparison_type: Optional[str] = Field(
        default=None,
        description="Type of comparison: performance, predictions, feature_importance, drift, ensemble_vs_base"
    )
    time_range: Optional[Dict[str, str]] = Field(
        default=None,
        description="Time period mentioned (e.g., {'period': 'last_month', 'start': '2024-09', 'end': '2024-10'})"
    )
    metrics_requested: Optional[List[str]] = Field(
        default=None,
        description="Metrics mentioned (e.g., ['RMSE', 'AUC', 'accuracy'])"
    )
    entities_requested: Optional[List[str]] = Field(
        default=None,
        description="Specific entities mentioned (e.g., HCP IDs, territory codes)"
    )
    needs_clarification: bool = Field(
        default=False,
        description="True if query is ambiguous or missing critical information"
    )
    clarification_question: Optional[str] = Field(
        default=None,
        description="Question to ask user for clarification"
    )
    requires_visualization: bool = Field(
        default=False,
        description="True if query implies need for charts/graphs"
    )
    references_previous_context: bool = Field(
        default=False,
        description="True if query refers to previous results (e.g., 'these', 'them', 'those models')"
    )
    resolved_models: Optional[List[str]] = Field(
        default=None,
        description="Model names after resolving references from conversation context"
    )
    references_specific_turn: bool = Field(  # NEW
        default=False,
        description="True if query references specific past turn (e.g., 'turn 3', 'previous analysis')"
    )
    referenced_turn_number: Optional[int] = Field(  # NEW
        default=None,
        description="Turn number referenced (e.g., 3 from 'turn 3')"
    )


class SQLQuerySpec(BaseModel):
    sql_query: str = Field(
        description="The SQL SELECT query to execute"
    )
    query_purpose: str = Field(
        description="What this query retrieves (for documentation)"
    )
    expected_columns: List[str] = Field(
        description="Expected column names in result"
    )


def extract_model_names_from_text(text: str) -> List[str]:
    model_patterns = [
        r'model[_\s]id[:\s]+([a-f0-9\-]{36})',
        r'model[_\s]name[:\s]+([^\n,]+)',
        r'\b(RF_\w+|XGB_\w+|LGB_\w+|ENS_\w+)\b',
    ]
    
    models = []
    for pattern in model_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        models.extend([m.strip() for m in matches])
    
    return list(set(models))


def detect_turn_reference(query: str) -> tuple[bool, Optional[int]]:
    """
    Detect if query references a specific past turn
    
    Returns:
        (has_reference, turn_number)
    """
    query_lower = query.lower()
    
    # Pattern 1: "turn X"
    turn_match = re.search(r'\bturn\s+(\d+)\b', query_lower)
    if turn_match:
        return True, int(turn_match.group(1))
    
    # Pattern 2: Temporal keywords
    temporal_keywords = ['previous', 'earlier', 'last time', 'before', 'first', 'initial']
    if any(kw in query_lower for kw in temporal_keywords):
        return True, None  # Has reference but no specific turn number
    
    return False, None



def query_simplification_agent(state: AgentState) -> AgentState:
    """
    Simplifies user queries using database schema context.
    Passes both original and simplified queries forward.
    """
    from langchain_google_genai import ChatGoogleGenerativeAI
    import os
    
    user_query = state['user_query']
    
    # Track execution
    execution_path = state.get('execution_path', [])
    execution_path.append('query_simplification')
    
    # Get database schema context
    try:
            # Get path to schema_context.py
            schema_file_path = os.path.join(
                os.path.dirname(__file__),
                '..',
                'core',
                'schema_context.py'
            )
            
            # Read file content
            with open(schema_file_path, 'r', encoding='utf-8') as f:
                schema_file_content = f.read()
            
            # Limit to avoid token limits (adjust as needed)
            schema_context = schema_file_content[:15000]
            
            print(f"✓ Loaded schema ({len(schema_context)} chars)")
            
    except Exception as e:
        print(f"⚠ Warning: Could not load schema: {e}")
        schema_context = """
        **Database Schema:**
        - models: model_id, model_name, algorithm, use_case, trained_date
        - executions: execution_id, model_id, execution_date, status
        - metrics: metric_id, execution_id, metric_name, metric_value
        - drift_results: drift_id, model_id, drift_score, drift_type
        """

    
    # Simplification prompt
    simplification_prompt = f"""You are a query simplification expert for a pharmaceutical ML analytics database.

**Database Schema:**
{schema_context}

**User Query:**
{user_query}

**Your Task:**
Simplify the query to be SHORT, PRECISE, and TO THE POINT. Focus on:

1. **Expand abbreviations** (e.g., "RMSE" stays "RMSE", "R²" stays "R²", "RF" → "Random Forest")
2. **Remove filler words** (e.g., "Can you please", "I want to", "Show me")
3. **Keep it concise** - DO NOT add explanations or extra details
4. **Standardize names** to match schema (e.g., "random forest" → "Random Forest")

**Rules:**
- Keep the query SHORT (max 20 words if possible)
- Remove redundant phrases
- Keep all essential filters, metrics, and entities
- Do NOT expand into explanations or specifications
- Output ONLY the simplified query

**Examples:**
- Original: "Can you please show me the RMSE values for all RF models trained in Q3?"
  Simplified: "RMSE for Random Forest models in Q3"

- Original: "Compare the top 5 models from last quarter across all use cases"
  Simplified: "Top 5 models last quarter, all use cases, RMSE R² execution count"

- Original: "What's the average accuracy of my ML models?"
  Simplified: "Average accuracy all models"

**Simplified Query:**"""

    try:
        response = llm.invoke(simplification_prompt)
        simplified_query = response.content.strip()
        
        print(f"\n{'='*60}")
        print(f"[Query Simplification Node]")
        print(f"Original: {user_query}")
        print(f"Simplified: {simplified_query}")
        print(f"{'='*60}\n")
        
    except Exception as e:
        print(f"Error in query simplification: {e}")
        simplified_query = user_query  
    
    return {
        'user_query': user_query,  # ← CRITICAL: Keep original query
        'simplified_query': simplified_query,
        'execution_path': execution_path,
        'messages': []  # Initialize empty messages list for downstream nodes
    }





def query_understanding_agent(state: AgentState) -> dict:
    """
    FIXED VERSION - Better intent classification for "tell me about" queries
    """
    query = state.get('simplified_query') or state.get('user_query', '')
    messages = state.get('messages', [])
    conversation_context = state.get('conversation_context', {})
    mentioned_models = state.get('mentioned_models', [])
    current_topic = state.get('current_topic')
    clarification_attempts = state.get('clarification_attempts', 0)
    last_query_summary = state.get('last_query_summary')
    final_insights = state.get('final_insights')
    
    # Check for explicit turn reference
    has_turn_ref, turn_number = detect_turn_reference(query)
    
    personalized_context_section = get_personalized_context_section(state)
    context_parts = []
    
    if mentioned_models:
        context_parts.append(f"Previously mentioned models: {', '.join(mentioned_models[:5])}")
    
    if current_topic:
        context_parts.append(f"Current topic: {current_topic}")
    
    if last_query_summary:
        context_parts.append(f"Previous query: {last_query_summary}")
    
    if final_insights:
        context_parts.append(f"Last insight summary: {final_insights[:500]}...")
    
    recent_content = []
    for msg in messages[-3:]:
        if isinstance(msg, AIMessage) and hasattr(msg, 'content') and msg.content:
            if any(keyword in msg.content.lower() for keyword in ['model', 'rmse', 'auc', 'ensemble', 'nrx', 'hcp']):
                recent_content.append(msg.content[:200])
    
    if recent_content:
        context_parts.append(f"Recent discussion: {' | '.join(recent_content)}")
    
    context_summary = "\n".join(context_parts) if context_parts else "No prior context"
    
    # === ENHANCED SYSTEM PROMPT ===
    system_msg = f"""You are a query parser for pharma commercial analytics. Parse user queries to extract:

USE CASES:
- NRx_forecasting: Predicting new prescriptions
- HCP_engagement: HCP response to marketing
- feature_importance_analysis: Understanding key drivers
- model_drift_detection: Detecting model performance changes over time
- messaging_optimization: Next-best-action for HCP targeting

MODELS (fuzzy matching - user may say "random forest", "RF", "forest", etc.):
- Base: Random Forest, XGBoost, LightGBM, Logistic Regression, SVM, Neural Network, Decision Tree
- Ensembles: stacking, boosting, bagging, meta-learner

COMPARISON TYPES:
- performance: Compare model metrics (DEFAULT for model-related queries)
- predictions: Compare actual predictions
- feature_importance: Compare which features matter most
- drift: Detect changes over time
- ensemble_vs_base: Why ensemble performs better/worse

METRICS: RMSE, MAE, R2, AUC-ROC, Accuracy, Precision, Recall, F1, TRx, NRx

CONVERSATION CONTEXT (USE THIS TO AVOID ASKING FOR INFO ALREADY GIVEN):
{context_summary}
{personalized_context_section}

CRITICAL QUERY CLASSIFICATION RULES:

1. **"Tell me about" / "List" / "Show me" queries:**
   - "tell me about all models for NRx" → comparison_type="performance" (NOT list!)
   - "show me models for HCP" → comparison_type="performance"
   - "what models do we have" → comparison_type="performance"
   - "what models have we ran" → comparison_type="performance"
   - These are asking to SEE model information, which requires comparison analysis

2. **"Compare" queries:**
   - "compare Random Forest vs XGBoost" → comparison_type="performance"
   - "compare all models" → comparison_type="performance"

3. **Check conversation context FIRST** before asking for clarification
4. Current clarification attempts: {clarification_attempts}
5. If attempts >= 3, DO NOT ask for clarification. Make reasonable assumptions.

6. **Reference resolution:**
   - "these models", "them", "those" → resolve from mentioned_models list
   - "the ensemble" → look for ensemble in context
   - "that model" → use last mentioned model

7. **Default to performance** for model-related queries without explicit comparison type

8. **Be GENEROUS with assumptions** when context exists

Examples:
- "tell me about all models for NRx forecasting" → use_case=NRx_forecasting, comparison_type=performance
- "Compare Random Forest vs XGBoost for NRx" → use_case=NRx_forecasting, models=['Random Forest', 'XGBoost'], comparison_type=performance
- "show performance" (with context use_case=NRx_forecasting) → use_case=NRx_forecasting, comparison_type=performance
- "which models" → comparison_type=performance (show all models)

Extract all relevant information. BE GENEROUS with assumptions when context exists.
You MUST include all fields defined in the structured model.
If you cannot infer a field, set it to null.
NEVER omit fields.
"""
    
    structured_llm = llm.with_structured_output(ParsedIntent, method="function_calling")
    
    context_messages = [SystemMessage(content=system_msg)]
    
    if messages:
        context_messages.extend(messages[-6:])
    
    context_messages.append(HumanMessage(content=query))

    # FIRST: Get result from LLM
    result = structured_llm.invoke(context_messages)

    # THEN: drift detection
    result_dict = result.dict()
    result_dict = detect_drift_query_intent(query, result_dict)

    # Update result object safely
    result.comparison_type = result_dict.get("comparison_type", result.comparison_type)

    if result_dict.get("requires_drift_analysis"):
        result.comparison_type = "drift"

    
    execution_path = state.get('execution_path', [])
    execution_path.append('query_understanding')
    
    final_models = result.models_requested or []
    new_mentioned_models = []
    
    # Handle pronoun resolution
    if result.references_previous_context:
        if result.resolved_models:
            final_models = result.resolved_models
        elif mentioned_models:
            final_models = mentioned_models
            result.resolved_models = mentioned_models
            result.needs_clarification = False
    
    if clarification_attempts >= 3:
        result.needs_clarification = False
        
        if not result.use_case and current_topic:
            result.use_case = current_topic
        
        if not final_models and mentioned_models:
            final_models = mentioned_models
    
    if final_models:
        new_mentioned_models.extend(final_models)
    
    # Extract models from recent AI messages
    last_ai_messages = [msg for msg in messages[-6:] if isinstance(msg, AIMessage)]
    for msg in last_ai_messages:
        if hasattr(msg, 'content') and msg.content:
            extracted = extract_model_names_from_text(msg.content)
            new_mentioned_models.extend(extracted)
    
    # Determine if we need memory or database
    needs_memory = result.references_specific_turn or (result.references_previous_context and not final_models)
    needs_database = bool(result.use_case or final_models or result.comparison_type)
    
    if has_turn_ref:
        needs_memory = True
        result.references_specific_turn = True
        result.referenced_turn_number = turn_number
    
    summary_parts = []
    if result.use_case:
        summary_parts.append(f"use_case: {result.use_case}")
    if final_models:
        summary_parts.append(f"models: {', '.join(final_models[:3])}")
    if result.comparison_type:
        summary_parts.append(f"comparison: {result.comparison_type}")
    
    return {
        "messages": [HumanMessage(content=query)],
        "parsed_intent": result.dict(),
        "use_case": result.use_case,
        "models_requested": final_models,
        "comparison_type": result.comparison_type,
        "time_range": result.time_range,
        "metrics_requested": result.metrics_requested,
        "entities_requested": result.entities_requested,
        "needs_clarification": result.needs_clarification,
        "clarification_question": result.clarification_question,
        "requires_visualization": result.requires_visualization,
        "execution_path": execution_path,
        "mentioned_models": new_mentioned_models if new_mentioned_models else None,
        "current_topic": result.use_case if result.use_case else state.get('current_topic'),
        "last_query_summary": " | ".join(summary_parts) if summary_parts else None,
        "clarification_attempts": clarification_attempts,
        "needs_memory": needs_memory,
        "needs_database": needs_database
    }


def context_retrieval_agent(state: AgentState) -> dict:
    execution_path = state.get('execution_path', [])
    execution_path.append('context_retrieval')
    
    needs_memory = state.get('needs_memory', False)
    needs_database = state.get('needs_database', True)
    session_id = state.get('session_id', 'default_session')
    turn_number = state.get('turn_number', 0)
    user_query = state.get('user_query', '')
    
    try:
        from core.vector_retriever import get_vector_retriever
        
        retriever = get_vector_retriever()
        
        # === 1. CONDITIONALLY retrieve conversation memory (LAZY) ===
        conversation_chunks = []
        needs_clarification = False
        clarification_question = ""
        
        if needs_memory and turn_number > 0:
            print(f"[Memory] Lazy search for session {session_id}, turn {turn_number}")
            
            # Build filters for specific turn lookup
            filters = {}
            referenced_turn = state.get('parsed_intent', {}).get('referenced_turn_number')
            if referenced_turn:
                filters["turn_number"] = {"$eq": referenced_turn}
        
            chunks, needs_clarify, clarify_msg = retriever.search_conversation_memory(
                query=user_query,
                session_id=session_id,
                current_turn=turn_number,
                filters=filters if filters else None,
                top_k=3  # Last 3 turns
            )
            
            conversation_chunks = chunks
            needs_clarification = needs_clarify
            clarification_question = clarify_msg
            
            if needs_clarification:
                print(f"[Memory] ⚠ Clarification needed: {clarification_question}")
            elif conversation_chunks:
                print(f"[Memory] ✓ Retrieved {len(conversation_chunks)} chunks")

                for chunk in conversation_chunks:
                    chunk_type = "full insight" if not chunk.get('is_partial') else "partial chunk"
                    print(f"  - Turn {chunk['turn']}: {chunk_type} (relevance: {chunk['relevance']:.2f})")
            else:
                print(f"[Memory] No relevant memory found")
        
        # === 2. CONDITIONALLY retrieve domain knowledge ===
        domain_docs = []
        if needs_database:
            print("[Domain] Retrieving domain knowledge")
            
            use_case = state.get('use_case')
            comparison_type = state.get('comparison_type')
            
            domain_docs = retriever.search_with_context(
                query=user_query,
                use_case=use_case,
                comparison_type=comparison_type,
                n_results=5,
                namespace="domain_knowledge"
            )
            
            print(f"[Domain] Retrieved {len(domain_docs)} documents")
        
        # === 3. Format context documents ===
        context_docs = []
        
        # Add conversation chunks FIRST (most relevant for follow-ups)
        for chunk in conversation_chunks:
            chunk_label = "Full Insight" if not chunk.get('is_partial') else f"Chunk {chunk.get('chunk_index', 0)+1}/{chunk.get('total_chunks', 1)}"
            
            context_docs.append({
                'doc_id': f"turn_{chunk['turn']}_{chunk.get('chunk_index', 0)}",
                'category': 'conversation_memory',
                'title': f"Turn {chunk['turn']} - {chunk_label}",
                'content': chunk['insight_chunk'],
                'relevance_score': chunk['relevance'],
                'source': 'conversation_memory',
                'turn_number': chunk['turn'],
                'user_query': chunk.get('user_query', ''),
                'is_partial': chunk.get('is_partial', True)
            })
        
        # Add domain knowledge
        for doc in domain_docs:
            context_docs.append({
                'doc_id': doc['doc_id'],
                'category': doc['metadata'].get('category', 'unknown'),
                'title': doc['metadata'].get('title', 'Untitled'),
                'content': doc['content'],
                'relevance_score': 1.0 - doc['distance'],
                'source': 'domain_knowledge'
            })
        
        # Add personalized context
        personalized_context = state.get('personalized_business_context', '')
        if personalized_context and personalized_context.strip():
            context_docs.insert(0, {
                'doc_id': 'PERSONALIZED_CONTEXT',
                'category': 'personalized',
                'title': 'User Personalized Business Context',
                'content': personalized_context.strip(),
                'relevance_score': 1.0,
                'keywords': ['personalized', 'user_context'],
                'source': 'user_input'
            })
        
        print(f"[Context] Total: {len(conversation_chunks)} memory + {len(domain_docs)} domain docs")
        
        # If clarification needed, set flag
        if needs_clarification:
            return {
                "context_documents": context_docs,
                "conversation_summaries": conversation_chunks,
                "execution_path": execution_path,
                "needs_clarification": True,
                "clarification_question": clarification_question
            }
        
        return {
            "context_documents": context_docs,
            "conversation_summaries": conversation_chunks,
            "execution_path": execution_path
        }
    
    except Exception as e:
        print(f"Context retrieval failed: {e}")
        import traceback
        traceback.print_exc()
        
        # On failure, ask user for clarification
        return {
            "context_documents": [],
            "conversation_summaries": [],
            "execution_path": execution_path,
            "needs_clarification": True,
            "clarification_question": "I'm having trouble accessing conversation history. Could you provide more details or specify which turn you're referring to?"
        }
    

def is_drift_query(user_query: str) -> bool:
    drift_keywords = [
        "drift", "concept drift", "data drift", "performance drift",
        "prediction drift", "detect drift", "drift score",
        "model drifting", "drift detection"
    ]
    return any(k in user_query.lower() for k in drift_keywords)



def sql_generation_agent(state: AgentState) -> dict:
    """
    ENHANCED VERSION - Uses semantic schema retrieval instead of full schema injection
    
    Generates SQL with:
    - Minimal core schema (always present)
    - Semantically retrieved detailed schema (only relevant tables)
    - Auto-expanded JOIN dependencies
    - Graceful retry with expanded retrieval
    """
    execution_path = state.get('execution_path', [])
    execution_path.append('sql_generation')
    
    use_case = state.get('use_case')
    models_requested = state.get('models_requested', [])
    comparison_type = state.get('comparison_type')
    metrics_requested = state.get('metrics_requested', [])
    time_range = state.get('time_range')
    context_docs = state.get('context_documents', [])
    user_query = state.get('simplified_query') or state.get('user_query', '')
    
    

    if should_use_drift_sql(state):
        print("[SQL] Detected DRIFT query - using specialized SQL")
        
        try:
            drift_sql = get_drift_sql_query(state)
            
            return {
                "generated_sql": drift_sql,
                "sql_purpose": f"Drift detection analysis for {use_case or 'all models'}",
                "expected_columns": [
                    "model_name", "drift_type", "drift_score", 
                    "is_significant", "detected_at"
                ],
                "execution_path": execution_path,
                "messages": [AIMessage(content=f"Generated drift detection SQL query")]
            }
        except Exception as e:
            print(f"[SQL] Drift SQL generation failed: {e}")



    personalized_context_section = get_personalized_context_section(state)
    
    # Get retry information
    retry_count = state.get('sql_retry_count', 0)
    previous_sql = state.get('generated_sql')
    previous_error = state.get('sql_error_feedback')
    
    # === RETRIEVE RELEVANT SCHEMA (Hybrid Approach) ===
    from core.schema_retreiver import retrieve_relevant_schema, format_schema_summary
    
    print(f"\n[SQL Generation] Retrieving schema (attempt {retry_count + 1})...")
    
    schema_context = retrieve_relevant_schema(
        query=user_query,
        use_case=use_case,
        comparison_type=comparison_type,
        models_requested=models_requested,
        retry_count=retry_count,
        top_k=3  # Will be auto-adjusted based on retry_count
    )
    
    print(f"[SQL Generation] Schema context size: {len(schema_context)} characters")
    
    # Build retry context if this is a retry attempt
    retry_context = ""
    if retry_count > 0 and previous_sql:
        retry_context = f"""
⚠️ RETRY ATTEMPT {retry_count}/3:
The previous query returned 0 rows. Review and modify the query.

Previous SQL that failed:
```sql
{previous_sql}
```

Issue: {previous_error}

CRITICAL FIXES NEEDED:
1. **Use LEFT JOINs** instead of INNER JOINs
2. **Relax ILIKE filters** - Use broader patterns (e.g., '%random%' not '%random forest%')
3. **Remove unnecessary filters** - Start with fewer constraints
4. **Verify column names** - Double-check against schema provided
5. **Use aggregation with FILTER** - For metrics, use AVG() with FILTER clause
6. **Check GROUP BY** - Ensure all non-aggregated columns are in GROUP BY

More schema context has been provided below to help with retry.
"""
    
    # === BUILD ENHANCED PROMPT ===
    prompt = f"""
You are a SQL expert specializing in pharma commercial analytics databases. 
Generate a **valid PostgreSQL SELECT query** that answers the user's question.

USER QUERY: {user_query}
context_docs:{context_docs}

PARSED INTENT:
- Use Case: {use_case}
- Models Requested: {models_requested}
- Comparison Type: {comparison_type}
- Metrics: {metrics_requested}
- Time Range: {time_range}

{retry_context}

{schema_context}

{personalized_context_section}

QUERY GENERATION GUIDELINES:

1. **Follow the core schema rules** (LEFT JOIN, is_active filter, etc.)

2. **Use the detailed schema provided above** for column names and relationships

3. **For performance metrics queries**:
   - MUST use AVG() with FILTER clause to create WIDE format
   - Example: AVG(pm.metric_value) FILTER (WHERE pm.metric_name = 'rmse' AND pm.data_split = 'test') as avg_test_rmse
   - Include multiple metrics as separate FILTER aggregations
   - GROUP BY model attributes

4. **For model comparisons**:
   - Use latest_model_executions view (not model_executions directly)
   - Filter by is_active = true
   - Include execution_count to verify data exists
   - Use HAVING COUNT(DISTINCT lme.execution_id) > 0

5. **Validation checklist**:
   □ Starts with SELECT
   □ Uses LEFT JOIN (not INNER JOIN)
   □ Filters by is_active = true
   □ Uses proper aggregation for metrics (AVG with FILTER)
   □ Has GROUP BY for aggregated queries
   □ Has LIMIT clause
   □ Uses ILIKE for case-insensitive matching
   □ Handles NULLs with NULLS LAST in ORDER BY

{"🔴 RETRY MODE: Broader filters, verify column names against schema" if retry_count > 0 else ""}

Generate the SQL query now:
"""

    try:
        structured_llm = llm.with_structured_output(SQLQuerySpec, method="function_calling")
        
        try:
            result = structured_llm.invoke(prompt)
        except Exception as e:
            print(f"[SQL Generation] Structured output failed, applying fallback parser: {e}")
            # Fallback: Extract SQL from response
            import re
            sql_match = re.search(r"SELECT[\s\S]+?;", prompt, re.IGNORECASE)
            extracted_sql = sql_match.group(0) if sql_match else "SELECT 1;"
            
            result = SQLQuerySpec(
                sql_query=extracted_sql,
                query_purpose="Auto-generated SQL query (fallback)",
                expected_columns=[]
            )
        
        # === VALIDATE GENERATED SQL ===
        sql = result.sql_query
        
        # Check for common issues
        issues = []
        if 'INNER JOIN' in sql.upper():
            issues.append("⚠ Using INNER JOIN - consider LEFT JOIN to avoid losing rows")
        if 'GROUP BY' not in sql.upper() and 'AVG(' in sql.upper():
            issues.append("⚠ Using AVG() without GROUP BY - may cause error")
        if 'LIMIT' not in sql.upper():
            issues.append("⚠ No LIMIT clause - may return too many rows")
        if not any(filter_clause in sql.upper() for filter_clause in ['IS_ACTIVE = TRUE', "IS_ACTIVE='TRUE'", "IS_ACTIVE=TRUE"]):
            issues.append("⚠ Missing is_active = true filter")
        
        if issues:
            print(f"\n[SQL Validation] Potential issues found:")
            for issue in issues:
                print(f"  {issue}")
        
        # Log schema retrieval stats
        print(f"\n[SQL Generation] Summary:")
        print(f"  - Schema context size: {len(schema_context)} chars")
        print(f"  - Retry attempt: {retry_count}")
        print(f"  - Generated SQL length: {len(sql)} chars")
        
        return {
            "generated_sql": result.sql_query,
            "sql_purpose": result.query_purpose,
            "expected_columns": result.expected_columns,
            "execution_path": execution_path,
            "messages": [AIMessage(content=f"Generated SQL query (Attempt {retry_count + 1}): {result.query_purpose}")]
        }
    
    except Exception as e:
        print(f"[SQL Generation] Failed: {e}")
        import traceback
        traceback.print_exc()
        
        return {
            "generated_sql": None,
            "sql_purpose": "Failed to generate SQL",
            "expected_columns": [],
            "execution_path": execution_path,
            "messages": [AIMessage(content=f"Failed to generate SQL: {str(e)}")]
        }


def data_retrieval_agent(state: AgentState) -> dict:
    execution_path = state.get('execution_path', [])
    execution_path.append('data_retrieval')
    
    generated_sql = state.get('generated_sql')
    retry_count = state.get('sql_retry_count', 0)
    
    if not generated_sql:
        return {
            "execution_path": execution_path,
            "messages": [AIMessage(content="No SQL query was generated")],
            "needs_sql_retry": False
        }
    
    # Execute the SQL query
    response = execute_sql_query.invoke(generated_sql)
    tool_message = ToolMessage(
        content=json.dumps(response), 
        tool_call_id=f"sql_exec_{id(generated_sql)}"
    )
    
    # Check if query returned 0 rows
    result_data = response.get('data', []) if response.get('success') else []
    row_count = len(result_data) if isinstance(result_data, list) else 0
    
    # Determine if we need to retry
    updated_state = {
        "messages": [tool_message],
        "execution_path": execution_path,
    }
    
    if row_count == 0 and retry_count < 3:
        # Need to retry - send back to SQL generation
        print(f" Query returned 0 rows. Initiating retry {retry_count + 1}/3")
        updated_state.update({
            "needs_sql_retry": True,
            "sql_retry_count": retry_count + 1,
            "sql_error_feedback": (
                f"Query returned 0 rows. The query may be too restrictive. "
                f"Consider: (1) Using broader ILIKE patterns, "
                f"(2) Reducing the number of filters, "
                f"(3) Using LEFT JOINs instead of INNER JOINs, "
                f"(4) Verifying table/column names and values exist in database."
                f"(5) consider changing the entire query if needed"
            )
        })
    
    elif row_count == 0 and retry_count >= 3:
        # Max retries reached
        print(f"❌ Maximum retry attempts (3) reached. No data found after {retry_count} attempts.")
        updated_state.update({
            "needs_sql_retry": False,
            "sql_retry_count": retry_count,
            "sql_error_feedback": (
                f"Maximum retry attempts (3) reached. No data found. "
                f"This may indicate: (1) No matching data exists in database, "
                f"(2) Query requirements are too specific, "
                f"(3) Data quality or schema issues."
            )
        })
    
    else:
        # Success - data retrieved
        extracted_models = extract_model_names_from_text(str(result_data))
        print(f"✅ Successfully retrieved {row_count} rows")
        updated_state.update({
            "needs_sql_retry": False,
            "sql_retry_count": 0,  # Reset counter on success
            "mentioned_models": extracted_models if extracted_models else None
        })
    
    return updated_state


def should_retry_sql(state: AgentState) -> str:
    """
    Routing function to determine if SQL generation should be retried.
    Returns the next node name.
    """
    needs_retry = state.get('needs_sql_retry', False)
    
    if needs_retry:
        print(f"🔄 Routing back to sql_generation for retry...")
        return "sql_generation"  # Route back to SQL generation
    else:
        return "analysis"  # Continue to analysis node


class AnalysisOutput(BaseModel):
    computed_metrics: Dict[str, Any] = Field(description="Key calculated metrics")
    patterns: List[str] = Field(description="Identified patterns and trends")
    anomalies: List[str] = Field(description="Unusual observations or outliers")
    statistical_summary: Dict[str, Any] = Field(description="Statistical summaries")

def analysis_computation_agent(state: AgentState) -> dict:
    """
    FIXED VERSION - Better debugging and data validation
    
    Two-stage analysis with comprehensive DEBUG logging
    """
    execution_path = state.get('execution_path', [])
    execution_path.append('analysis_computation')

    messages = state.get('messages', [])
    tool_results = []

    from langchain_core.messages import ToolMessage
    for msg in messages:
        if isinstance(msg, ToolMessage):
            try:
                result = json.loads(msg.content) if isinstance(msg.content, str) else msg.content
                tool_results.append(result)
            except:
                continue

    # Extract successful data
    successful_data = []
    for result in tool_results:
        if result.get('success') and result.get('data'):
            successful_data.extend(result['data'])

    # ===== CRITICAL DEBUG LOGGING =====
    print(f"\n{'='*80}")
    print(f"[Analysis Debug] Data Inspection")
    print(f"{'='*80}")
    print(f"Total tool results: {len(tool_results)}")
    print(f"Total rows extracted: {len(successful_data)}")
    
    if successful_data:
        print(f"\n[Sample Row]")
        print(json.dumps(successful_data[0], indent=2, default=str))
        print(f"\n[All Columns]: {list(successful_data[0].keys())}")
        print(f"[Column Count]: {len(successful_data[0].keys())}")
        
        # Check for metric columns
        metric_cols = [col for col in successful_data[0].keys() 
                      if any(m in col.lower() for m in ['rmse', 'mae', 'r2', 'auc', 'accuracy', 'precision'])]
        print(f"[Detected Metric Columns]: {metric_cols}")
        
        # Check if data is in long or wide format
        has_metric_name = 'metric_name' in successful_data[0]
        has_metric_value = 'metric_value' in successful_data[0]
        has_model_name = 'model_name' in successful_data[0]
        
        print(f"\n[Format Detection]:")
        print(f"  - Has 'metric_name': {has_metric_name}")
        print(f"  - Has 'metric_value': {has_metric_value}")
        print(f"  - Has 'model_name': {has_model_name}")
        print(f"  - Format: {'LONG' if (has_metric_name and has_metric_value) else 'WIDE'}")
    else:
        print("[ERROR] No successful data found!")
    print(f"{'='*80}\n")
    # ===== END DEBUG =====

    if not successful_data:
        print("[Analysis] No data - returning empty result")
        return {
            "analysis_results": {
                'analysis_type': 'empty',
                'error': 'No data returned from SQL',
                'story': 'NO_DATA: No data available for analysis.',
                'story_elements': {},
                'computed_values': {},
                'raw_metrics': {}
            },
            "execution_path": execution_path
        }

    # Build query context
    query_context = {
        'user_query': state.get('user_query'),
        'comparison_type': state.get('comparison_type'),
        'use_case': state.get('use_case'),
        'models_requested': state.get('models_requested'),
        'metrics_requested': state.get('metrics_requested'),
        'data_row_count': len(successful_data),
        'data_columns': list(successful_data[0].keys()) if successful_data else []
    }
    
    # === ENHANCED LLM DECISION WITH BETTER PROMPTING ===
    analysis_decision_prompt = f"""You are an analysis router. Based on the query context and data, decide which analysis type to run.

USER QUERY: {query_context['user_query']}

DATA STRUCTURE:
- Row Count: {query_context['data_row_count']}
- Columns: {', '.join(query_context['data_columns'][:15])}
- Has model_name: {'model_name' in query_context['data_columns']}
- Has metric columns: {any('rmse' in col.lower() or 'mae' in col.lower() or 'r2' in col.lower() for col in query_context['data_columns'])}

QUERY CONTEXT:
- Use Case: {query_context['use_case']}
- Comparison Type: {query_context['comparison_type']}
- Models Requested: {query_context['models_requested']}

AVAILABLE ANALYSIS TYPES:
- performance: Model performance comparison (use this for NRx forecasting, model comparisons, "tell me about models")
- drift: Drift detection analysis
- ensemble_vs_base: Ensemble vs base model comparison
- feature_importance: Feature importance analysis
- general: General data analysis (fallback)

DECISION RULES:
1. If query contains "models for" or "tell me about models" → performance
2. If data has model_name + metric columns (rmse, r2, mae, etc.) → performance
3. If use_case = "NRx_forecasting" → performance
4. If comparison_type = "performance" → performance
5. If query asks about "drift" → drift
6. If query asks about "features" or "importance" → feature_importance
7. Default to performance for model-related queries

CRITICAL: Return ONLY ONE WORD - the analysis type (e.g., "performance")
"""

    try:
        decision_response = llm.invoke(analysis_decision_prompt)
        analysis_type = decision_response.content.strip().lower()
        
        valid_types = [
            "performance", "drift", "ensemble_vs_base", "feature_importance",
            "uplift", "territory_performance", "market_share", "price_sensitivity",
            "competitor_share", "clustering", "predictions", "versions", "general"
        ]
        
        if analysis_type not in valid_types:
            # Try to find matching type
            for vtype in valid_types:
                if vtype in analysis_type:
                    analysis_type = vtype
                    break
            else:
                # Default to performance for model queries
                if 'model' in query_context['user_query'].lower():
                    analysis_type = "performance"
                else:
                    analysis_type = "general"
        
        print(f"[Analysis] LLM selected type: {analysis_type}")
        query_context['comparison_type'] = analysis_type
        
    except Exception as e:
        print(f"[Analysis] LLM decision failed: {e}, defaulting to performance")
        analysis_type = "performance"
        query_context['comparison_type'] = analysis_type

    # === CALL AGGREGATOR ===
    print(f"[Analysis] Calling aggregator with type: {analysis_type}")
    from core.analysis_aggregators import analyze_data
    analysis_results = analyze_data(successful_data, query_context)

    # === VALIDATE RESULTS ===
    print(f"\n[Analysis Results Validation]")
    print(f"  - Analysis Type: {analysis_results.get('analysis_type')}")
    print(f"  - Story Length: {len(analysis_results.get('story', ''))}")
    print(f"  - Computed Values Keys: {list(analysis_results.get('computed_values', {}).keys())[:10]}")
    print(f"  - Story Elements Keys: {list(analysis_results.get('story_elements', {}).keys())}")
    print(f"  - Raw Metrics Keys: {list(analysis_results.get('raw_metrics', {}).keys())}")
    
    # Check if we got meaningful results
    story = analysis_results.get('story', '')
    if story and story != "NO_DATA: No data available for analysis.":
        print(f"  - Status: ✓ Valid analysis generated")
    else:
        print(f"  - Status: ✗ Empty or invalid analysis!")
        print(f"  - Story preview: {story[:200]}")

    # Attach raw data for reference
    analysis_results['raw_data'] = tool_results
    analysis_results['data_row_count'] = len(successful_data)

    return {
        "analysis_results": analysis_results,
        "execution_path": execution_path
    }

def _compute_basic_metrics(data: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not data:
        return {}
    
    metrics = {}
    
    metric_columns = ['rmse', 'mae', 'r2_score', 'r2', 'auc_roc', 'accuracy', 
                     'precision', 'recall', 'f1_score', 'drift_score',
                     'ensemble_advantage', 'metric_value', 'importance_score']
    
    for col in metric_columns:
        values = [row.get(col) for row in data if row.get(col) is not None]
        if values and all(isinstance(v, (int, float)) for v in values):
            metrics[col] = {
                'mean': sum(values) / len(values),
                'min': min(values),
                'max': max(values),
                'count': len(values)
            }
    
    model_names = list(set(row.get('model_name') for row in data if row.get('model_name')))
    if model_names:
        metrics['models_analyzed'] = model_names
    
    return metrics

class ChartSpec(BaseModel):
    chart_type: str = Field(description="bar, line, scatter, box, histogram")
    title: str
    x_axis: str
    y_axis: str
    color: Optional[str] = None
    facet: Optional[str] = None
    orientation: Optional[str] = Field(default="v", description="v or h")
    barmode: Optional[str] = Field(default=None, description="group, stack, or null")
    sort_by: Optional[str] = None
    sort_ascending: bool = True
    filter: Optional[Dict[str, Any]] = None
    explanation: str


class VizSpecOutput(BaseModel):
    strategy: str = Field(description="single_chart, multiple_charts, or no_visualization")
    reason: str
    charts: List[ChartSpec]
    warnings: List[str] = Field(default_factory=list)


VIZ_RULES = """
CRITICAL VISUALIZATION RULES:

1. NEVER stack metrics with different scales (RMSE, R2, MAE, AUC, etc.)
   - These metrics have different units and interpretations
   - Stacking them creates nonsensical visualizations

2. For model performance comparison with multiple metrics:
   - PREFERRED: Create separate chart for each metric
   - Each chart compares all models on ONE metric
   - Use barmode='group' if combining in one chart (rarely recommended)

3. For single metric comparison:
   - Simple bar chart: models on x-axis, metric on y-axis
   - Sort by metric value for clarity

4. For time series:
   - Use line charts
   - Color by entity (model, HCP, region)

5. For rankings/importance:
   - Horizontal bars sorted by value
   - Most important at top

6. For distributions:
   - Histograms for continuous
   - Box plots for comparison across groups

7. Data size limits:
   - >100 points: Suggest filtering or aggregation
   - High cardinality (>20 unique values): Aggregate or limit

8. Always be explicit:
   - Specify barmode for bar charts
   - Specify orientation for clarity
   - Provide clear titles and explanations
"""


def visualization_specification_agent(state: AgentState) -> dict:
    """
    ENHANCED: Better handling of wide-format SQL results with robust fallback
    
    Improvements:
    - Handles both LONG and WIDE format data
    - Comprehensive error handling with fallback visualizations
    - Better column detection and validation
    - Creates simple bar charts when LLM fails
    """
    execution_path = state.get('execution_path', [])
    execution_path.append('visualization_spec')
    
    analysis_results = state.get('analysis_results', {})
    raw_data = analysis_results.get('raw_data', [])

    personalized_context_section = get_personalized_context_section(state)
    
    # Extract DataFrame from results
    df = None
    for result in raw_data:
        if result.get('success') and result.get('data'):
            df = pd.DataFrame(result['data'])
            break
    
    if df is None or df.empty:
        print("[Viz] No data for visualization")
        return {
            "execution_path": execution_path,
            "visualization_specs": [],
            "rendered_charts": []
        }
    
    print(f"\n[Viz] DataFrame shape: {df.shape}")
    print(f"[Viz] Columns: {list(df.columns)}")
    print(f"[Viz] Sample data:\n{df.head(2)}")
    
    if len(df) > 100:
        print(f"[Viz] Limiting data from {len(df)} to 100 rows")
        df = df.head(100)
    
    user_query = state['user_query']
    
    # ===== ENHANCED: Safe cardinality calculation =====
    cardinality = {}
    for col in df.columns:
        try:
            # Try to compute unique count
            cardinality[col] = int(df[col].nunique())
        except TypeError:
            # Column contains unhashable types (dict, list)
            cardinality[col] = len(df)  # Treat as unique per row
            print(f"[Viz] Column '{col}' contains unhashable types, skipping")
    
    # ===== DETECT DATA FORMAT =====
    is_long_format = 'metric_name' in df.columns and 'metric_value' in df.columns
    is_wide_format = not is_long_format and len([c for c in df.columns if any(m in c.lower() for m in ['rmse', 'mae', 'r2', 'auc'])]) > 0
    
    print(f"[Viz] Format detection: Long={is_long_format}, Wide={is_wide_format}")
    
    data_summary = {
        'shape': {'rows': len(df), 'columns': len(df.columns)},
        'columns': list(df.columns),
        'dtypes': {col: str(dtype) for col, dtype in df.dtypes.items()},
        'cardinality': cardinality,
        'sample_data': df.head(5).to_dict('records'),
        'is_long_format': is_long_format,
        'is_wide_format': is_wide_format
    }
    
    # ===== ENHANCED PROMPT FOR WIDE FORMAT =====
    prompt = f"""You are a visualization expert for pharma analytics.

USER QUERY: {user_query}

DATA STRUCTURE:
{json.dumps(data_summary, indent=2, default=str)}

DATA FORMAT DETECTED: {"LONG format (metric_name + metric_value)" if is_long_format else "WIDE format (separate metric columns)"}

{VIZ_RULES}
{personalized_context_section}

CRITICAL INSTRUCTIONS FOR WIDE FORMAT:

If data is in WIDE format (separate columns for each metric):
1. **Each metric gets its own chart** (avoid stacking different metrics)
2. Use model_name (or similar) on x-axis
3. Use specific metric column on y-axis (e.g., avg_test_rmse)
4. Sort by metric value for clarity
5. Example spec:
   {{
     "chart_type": "bar",
     "title": "RMSE Comparison Across Models",
     "x_axis": "model_name",
     "y_axis": "avg_test_rmse",
     "sort_by": "avg_test_rmse",
     "sort_ascending": true
   }}

SPECIAL CASE: Simple count/execution data
If data only has model_name and a count column (like execution_count):
- Create a simple bar chart
- X-axis: model_name
- Y-axis: count column
- Title: "Model Execution Count" or similar
- Sort by count descending

EXAMPLES:

Example 1 - Simple Count Data:
Input: columns=['model_name', 'execution_count']
       rows=7 (7 models)
       
Output:
{{
  "strategy": "single_chart",
  "reason": "Simple comparison of execution counts across models",
  "charts": [
    {{
      "chart_type": "bar",
      "title": "Model Execution Count",
      "x_axis": "model_name",
      "y_axis": "execution_count",
      "sort_by": "execution_count",
      "sort_ascending": false,
      "barmode": null,
      "explanation": "Bar chart showing number of executions per model"
    }}
  ]
}}

Example 2 - Wide Format (MOST COMMON NOW):
Input: columns=['model_name', 'avg_test_rmse', 'avg_test_r2', 'avg_test_mae']
       rows=5 (5 models)
       
Output:
{{
  "strategy": "multiple_charts",
  "reason": "Wide format with multiple metrics - create separate chart for each metric",
  "charts": [
    {{
      "chart_type": "bar",
      "title": "RMSE Comparison Across Models",
      "x_axis": "model_name",
      "y_axis": "avg_test_rmse",
      "sort_by": "avg_test_rmse",
      "sort_ascending": true,
      "barmode": null,
      "explanation": "Lower RMSE indicates better accuracy. Models sorted from best to worst."
    }},
    {{
      "chart_type": "bar",
      "title": "R² Score Comparison Across Models",
      "x_axis": "model_name",
      "y_axis": "avg_test_r2",
      "sort_by": "avg_test_r2",
      "sort_ascending": false,
      "barmode": null,
      "explanation": "Higher R² indicates better model fit. Models sorted from best to worst."
    }},
    {{
      "chart_type": "bar",
      "title": "MAE Comparison Across Models",
      "x_axis": "model_name",
      "y_axis": "avg_test_mae",
      "sort_by": "avg_test_mae",
      "sort_ascending": true,
      "barmode": null,
      "explanation": "Lower MAE indicates better accuracy. Models sorted from best to worst."
    }}
  ]
}}

Example 3 - Long Format:
Input: columns=['model_name', 'metric_name', 'metric_value']
       metric_name has values: ['rmse', 'mae', 'r2']
       
Output:
{{
  "strategy": "multiple_charts",
  "reason": "Multiple metrics require separate charts for clarity",
  "charts": [
    {{
      "chart_type": "bar",
      "title": "RMSE Comparison",
      "x_axis": "model_name",
      "y_axis": "metric_value",
      "filter": {{"metric_name": "rmse"}},
      "sort_by": "metric_value",
      "sort_ascending": true,
      "explanation": "Lower RMSE is better"
    }}
  ]
}}

Now analyze the actual data and generate visualization specifications.
Return valid JSON only.
"""
    
    # ===== TRY LLM GENERATION WITH FALLBACK =====
    viz_spec = None
    llm_error = None
    
    try:
        structured_llm = llm.with_structured_output(VizSpecOutput, method="function_calling")
        viz_spec = structured_llm.invoke(prompt)
        
        print(f"[Viz] LLM generated {len(viz_spec.charts)} chart specs")
        
        validation_issues = validate_viz_spec(viz_spec, df)
        
        if validation_issues:
            print(f"[Viz] Validation issues: {validation_issues}")
            viz_spec.warnings.extend(validation_issues)
        
    except Exception as e:
        print(f"[Viz] ✗ LLM viz spec generation failed: {e}")
        import traceback
        traceback.print_exc()
        llm_error = str(e)
        
        # ===== FALLBACK: Create simple visualization =====
        print(f"[Viz] Creating fallback visualization...")
        
        # Find suitable columns for simple bar chart
        categorical_cols = [col for col in df.columns 
                           if df[col].dtype == 'object' or df[col].dtype.name == 'category']
        numeric_cols = [col for col in df.columns 
                       if pd.api.types.is_numeric_dtype(df[col])]
        
        print(f"[Viz] Fallback - Found {len(categorical_cols)} categorical and {len(numeric_cols)} numeric columns")
        
        if categorical_cols and numeric_cols:
            # Pick best columns for visualization
            # Prefer model_name, name, or id columns for x-axis
            x_col = None
            for cat_col in categorical_cols:
                if 'name' in cat_col.lower() or 'model' in cat_col.lower():
                    x_col = cat_col
                    break
            if not x_col:
                x_col = categorical_cols[0]
            
            # Pick first numeric column for y-axis
            y_col = numeric_cols[0]
            
            # Determine sort direction based on column name
            sort_ascending = True
            if any(term in y_col.lower() for term in ['error', 'rmse', 'mae', 'mse', 'loss']):
                sort_ascending = True  # Lower is better
            elif any(term in y_col.lower() for term in ['score', 'accuracy', 'r2', 'auc', 'precision']):
                sort_ascending = False  # Higher is better
            else:
                sort_ascending = False  # Default to descending for counts
            
            print(f"[Viz] Fallback chart: {y_col} by {x_col} (sort_ascending={sort_ascending})")
            
            # Create simple chart spec
            viz_spec = VizSpecOutput(
                strategy="single_chart",
                reason=f"Fallback: Simple bar chart showing {y_col} by {x_col}",
                charts=[ChartSpec(
                    chart_type="bar",
                    title=f"{y_col.replace('_', ' ').title()} by {x_col.replace('_', ' ').title()}",
                    x_axis=x_col,
                    y_axis=y_col,
                    sort_by=y_col,
                    sort_ascending=sort_ascending,
                    barmode=None,
                    explanation=f"Bar chart comparing {y_col} across different {x_col} values"
                )],
                warnings=[f"Using fallback visualization due to LLM error: {llm_error}"]
            )
        else:
            # No suitable columns found
            print(f"[Viz] ✗ Cannot create fallback chart: categorical={len(categorical_cols)}, numeric={len(numeric_cols)}")
            return {
                "visualization_specs": [],
                "rendered_charts": [],
                "viz_warnings": [
                    f"Visualization failed: {llm_error}",
                    f"No suitable columns for fallback chart (need categorical + numeric)",
                    f"Available columns: {list(df.columns)}"
                ],
                "execution_path": execution_path
            }
    
    # ===== RENDER CHARTS =====
    rendered_charts = []
    for i, chart_spec in enumerate(viz_spec.charts, 1):
        try:
            print(f"[Viz] Rendering chart {i}: {chart_spec.title}")
            fig = render_chart(df, chart_spec)
            if fig:
                rendered_charts.append({
                    'title': chart_spec.title,
                    'figure': fig,
                    'type': chart_spec.chart_type,
                    'explanation': chart_spec.explanation
                })
                print(f"[Viz]   ✓ Chart {i} rendered successfully")
            else:
                print(f"[Viz]   ✗ Chart {i} rendering returned None")
                viz_spec.warnings.append(f"Chart {i} ({chart_spec.title}) rendering returned None")
        except Exception as e:
            print(f"[Viz]   ✗ Chart {i} rendering failed: {e}")
            import traceback
            traceback.print_exc()
            viz_spec.warnings.append(f"Failed to render {chart_spec.title}: {str(e)}")
    
    print(f"[Viz] Successfully rendered {len(rendered_charts)}/{len(viz_spec.charts)} charts")
    
    # If no charts were rendered but we have specs, add warning
    if len(viz_spec.charts) > 0 and len(rendered_charts) == 0:
        viz_spec.warnings.append("All chart rendering attempts failed. Check data types and column names.")
    
    return {
        "visualization_specs": [spec.dict() for spec in viz_spec.charts],
        "rendered_charts": rendered_charts,
        "viz_strategy": viz_spec.strategy,
        "viz_reasoning": viz_spec.reason,
        "viz_warnings": viz_spec.warnings,
        "execution_path": execution_path,
        "requires_visualization": len(rendered_charts) > 0
    }

def validate_viz_spec(spec: VizSpecOutput, df: pd.DataFrame) -> List[str]:
    issues = []
    
    for chart in spec.charts:
        required_cols = [chart.x_axis, chart.y_axis]
        if chart.color:
            required_cols.append(chart.color)
        if chart.facet:
            required_cols.append(chart.facet)
        
        for col in required_cols:
            if col not in df.columns:
                issues.append(f"Column '{col}' not found in data")
        
        if chart.barmode == 'stack' and is_metrics_comparison(df, chart):
            issues.append(f"BLOCKED: Cannot stack metrics in '{chart.title}' - switching to grouped")
            chart.barmode = 'group'
        
        if chart.filter:
            filter_col = list(chart.filter.keys())[0]
            if filter_col not in df.columns:
                issues.append(f"Filter column '{filter_col}' not found")
    
    return issues


def is_metrics_comparison(df: pd.DataFrame, chart: ChartSpec) -> bool:
    metric_indicators = ['metric', 'rmse', 'mae', 'r2', 'auc', 'accuracy', 
                        'precision', 'recall', 'f1', 'mape', 'mse']
    
    if chart.color:
        color_values = df[chart.color].unique() if chart.color in df.columns else []
        for val in color_values:
            if any(indicator in str(val).lower() for indicator in metric_indicators):
                return True
    
    y_col_name = chart.y_axis.lower()
    if any(indicator in y_col_name for indicator in metric_indicators):
        return True
    
    return False


def render_chart(df: pd.DataFrame, spec: ChartSpec) -> Optional[go.Figure]:
    """
    FIXED: Better data handling and debugging for chart rendering
    """
    try:
        plot_df = df.copy()
        
        # ===== DEBUG: Show what we're working with =====
        print(f"\n[Render] Chart: {spec.title}")
        print(f"[Render] Type: {spec.chart_type}")
        print(f"[Render] X: {spec.x_axis}, Y: {spec.y_axis}")
        print(f"[Render] Input data shape: {plot_df.shape}")
        print(f"[Render] Columns: {list(plot_df.columns)}")
        
        # ===== CRITICAL: Validate columns exist =====
        if spec.x_axis not in plot_df.columns:
            print(f"[Render] ERROR: X-axis column '{spec.x_axis}' not found!")
            print(f"[Render] Available columns: {list(plot_df.columns)}")
            return None
        
        if spec.y_axis not in plot_df.columns:
            print(f"[Render] ERROR: Y-axis column '{spec.y_axis}' not found!")
            print(f"[Render] Available columns: {list(plot_df.columns)}")
            return None
        
        # ===== Apply filter if specified =====
        if spec.filter:
            print(f"[Render] Applying filter: {spec.filter}")
            for col, val in spec.filter.items():
                if col in plot_df.columns:
                    plot_df = plot_df[plot_df[col] == val]
                else:
                    print(f"[Render] WARNING: Filter column '{col}' not found")
        
        print(f"[Render] Data shape after filter: {plot_df.shape}")
        
        if len(plot_df) == 0:
            print(f"[Render] ERROR: No data after filtering!")
            return None
        
        # ===== CRITICAL: Convert Y-axis to numeric =====
        y_col = spec.y_axis
        print(f"[Render] Y-axis column '{y_col}' dtype: {plot_df[y_col].dtype}")
        print(f"[Render] Y-axis sample values: {plot_df[y_col].head().tolist()}")
        
        # Check for NULL values
        null_count = plot_df[y_col].isna().sum()
        if null_count > 0:
            print(f"[Render] WARNING: {null_count} NULL values in Y-axis, dropping...")
            plot_df = plot_df.dropna(subset=[y_col])
        
        if len(plot_df) == 0:
            print(f"[Render] ERROR: No data after removing NULLs!")
            return None
        
        # Force convert to numeric
        try:
            plot_df[y_col] = pd.to_numeric(plot_df[y_col], errors='coerce')
            
            # Check if conversion created NaNs
            nan_count = plot_df[y_col].isna().sum()
            if nan_count > 0:
                print(f"[Render] WARNING: {nan_count} values couldn't convert to numeric")
                plot_df = plot_df.dropna(subset=[y_col])
            
            print(f"[Render] After numeric conversion: {plot_df[y_col].head().tolist()}")
            
        except Exception as e:
            print(f"[Render] ERROR: Failed to convert Y-axis to numeric: {e}")
            return None
        
        if len(plot_df) == 0:
            print(f"[Render] ERROR: No valid numeric data!")
            return None
        
        # ===== Sort if requested =====
        if spec.sort_by and spec.sort_by in plot_df.columns:
            print(f"[Render] Sorting by {spec.sort_by} (ascending={spec.sort_ascending})")
            plot_df = plot_df.sort_values(spec.sort_by, ascending=spec.sort_ascending)
        
        # ===== Build plotly kwargs =====
        kwargs = {
            'data_frame': plot_df,
            'x': spec.x_axis,
            'y': spec.y_axis,
            'title': spec.title
        }
        
        if spec.color and spec.color in plot_df.columns:
            kwargs['color'] = spec.color
        
        if spec.facet and spec.facet in plot_df.columns:
            kwargs['facet_col'] = spec.facet
        
        # ===== Create chart based on type =====
        print(f"[Render] Creating {spec.chart_type} chart...")
        
        if spec.chart_type == 'bar':
            if spec.orientation:
                kwargs['orientation'] = spec.orientation
            fig = px.bar(**kwargs)
            if spec.barmode:
                fig.update_layout(barmode=spec.barmode)
        
        elif spec.chart_type == 'line':
            kwargs['markers'] = True
            fig = px.line(**kwargs)
        
        elif spec.chart_type == 'scatter':
            fig = px.scatter(**kwargs)
        
        elif spec.chart_type == 'box':
            fig = px.box(**kwargs)
        
        elif spec.chart_type == 'histogram':
            fig = px.histogram(**kwargs)
        
        else:
            print(f"[Render] ERROR: Unsupported chart type: {spec.chart_type}")
            return None
        
        # ===== Update layout for better visibility =====
        fig.update_layout(
            template="plotly_white",
            height=500,  # Increased from 400
            margin=dict(l=60, r=40, t=80, b=100),  # More space for labels
            font=dict(size=12),
            xaxis=dict(
                tickangle=-45,  # Angle labels for readability
                title=dict(font=dict(size=14, color='white')),
                tickfont=dict(color='white')
            ),
            yaxis=dict(
                title=dict(font=dict(size=14, color='white')),
                tickfont=dict(color='white'),
                gridcolor='rgba(128, 128, 128, 0.2)'
            ),
            title=dict(
                font=dict(size=16, color='white'),
                x=0.5,
                xanchor='center'
            ),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        
        # ===== DEBUG: Check if data is actually plotted =====
        print(f"[Render] Chart created successfully")
        print(f"[Render] Data points plotted: {len(plot_df)}")
        print(f"[Render] Y-axis range: {plot_df[y_col].min():.4f} to {plot_df[y_col].max():.4f}")
        
        return fig
    
    except Exception as e:
        print(f"[Render] ERROR: Failed to render chart '{spec.title}': {e}")
        import traceback
        traceback.print_exc()
        return None


# ===== ALSO ADD THIS VALIDATION FUNCTION =====

def validate_viz_spec(spec: VizSpecOutput, df: pd.DataFrame) -> List[str]:
    """
    ENHANCED: Better validation with specific error messages
    """
    issues = []
    
    for chart in spec.charts:
        chart_issues = []
        
        # Check all required columns exist
        required_cols = [chart.x_axis, chart.y_axis]
        if chart.color:
            required_cols.append(chart.color)
        if chart.facet:
            required_cols.append(chart.facet)
        
        for col in required_cols:
            if col not in df.columns:
                chart_issues.append(f"Column '{col}' not found in data")
        
        # Check if y-axis column has numeric data
        if chart.y_axis in df.columns:
            y_data = df[chart.y_axis]
            
            # Check for NULLs
            null_pct = (y_data.isna().sum() / len(y_data)) * 100
            if null_pct > 50:
                chart_issues.append(f"Y-axis '{chart.y_axis}' has {null_pct:.1f}% NULL values")
            
            # Check if numeric
            try:
                numeric_data = pd.to_numeric(y_data, errors='coerce')
                non_numeric_pct = (numeric_data.isna().sum() / len(y_data)) * 100
                if non_numeric_pct > 50:
                    chart_issues.append(f"Y-axis '{chart.y_axis}' has {non_numeric_pct:.1f}% non-numeric values")
            except:
                chart_issues.append(f"Y-axis '{chart.y_axis}' cannot be converted to numeric")
        
        # Check for metric stacking issue
        if chart.barmode == 'stack' and is_metrics_comparison(df, chart):
            chart_issues.append(f"BLOCKED: Cannot stack different metrics - switching to grouped")
            chart.barmode = 'group'
        
        # Check filter validity
        if chart.filter:
            filter_col = list(chart.filter.keys())[0]
            if filter_col not in df.columns:
                chart_issues.append(f"Filter column '{filter_col}' not found")
        
        if chart_issues:
            issues.append(f"Chart '{chart.title}': {'; '.join(chart_issues)}")
    
    return issues


def visualization_rendering_agent(state: AgentState) -> dict:
    execution_path = state.get('execution_path', [])
    execution_path.append('visualization_rendering')
    
    rendered_charts = state.get('rendered_charts', [])
    
    return {
        "rendered_charts": rendered_charts,
        "execution_path": execution_path
    }

def insight_generation_agent(state: AgentState) -> dict:
    """
    FIXED VERSION - Better formatting and explicit model listing
    """
    execution_path = state.get('execution_path', [])
    execution_path.append('insight_generation')

    analysis_results = state.get('analysis_results', {}) or {}
    context_docs = state.get('context_documents', []) or []
    rendered_charts = state.get('rendered_charts', []) or []
    conversation_summaries = state.get('conversation_summaries', []) or []
    personalized_context_section = get_personalized_context_section(state)

    # Extract analysis components
    story = analysis_results.get("story", "")
    story_elements = analysis_results.get("story_elements", {}) or {}
    computed_values = analysis_results.get("computed_values", {}) or {}
    raw_metrics = analysis_results.get("raw_metrics", {}) or {}
    analysis_type = analysis_results.get("analysis_type", "unknown")
    models_analyzed = analysis_results.get("models_analyzed", [])
    
    print(f"\n{'='*80}")
    print(f"[Insight Generation] Starting")
    print(f"{'='*80}")
    print(f"Analysis type: {analysis_type}")
    print(f"Story length: {len(story)} chars")
    print(f"Models analyzed: {models_analyzed}")
    print(f"Computed values: {len(computed_values)} keys")
    print(f"Raw metrics: {len(raw_metrics)} keys")
    
    # Validate data
    has_data = bool(computed_values and story and story != "NO_DATA: No data available for analysis.")
    
    if not has_data:
        print("[Insight Generation] ✗ No meaningful analysis data found!")
        return {
            "messages": [AIMessage(content="I couldn't find data to analyze. Please check:\n- Are there models registered in the database?\n- Are the model names spelled correctly?\n- Is the use case specified correctly?")],
            "final_insights": "No data available for analysis.",
            "rendered_charts": rendered_charts,
            "execution_path": execution_path
        }
    
    print("[Insight Generation] ✓ Valid analysis data found")
    
    # Build context sections
    viz_context = ""
    if rendered_charts:
        parts = []
        for i, ch in enumerate(rendered_charts, 1):
            parts.append(
                f"**Chart {i}: {ch.get('title','Chart')}**\n"
                f"Shows: {ch.get('explanation','Metric visualization')}"
            )
        viz_context = "\n".join(parts)

    conv_context = ""
    if conversation_summaries:
        conv_context = "**Previous Discussion:**\n"
        for chunk in conversation_summaries[:2]:
            txt = chunk.get("insight_chunk","")
            preview = txt[:200] + ("..." if len(txt)>200 else "")
            conv_context += f"- Turn {chunk.get('turn')}: {preview}\n"

    domain_context = ""
    if context_docs:
        snip = []
        for doc in context_docs[:2]:
            title = doc.get("title","Context")
            content = doc.get("content","")[:200]
            snip.append(f"**{title}**: {content}...")
        if snip:
            domain_context = "\n".join(snip)

    # Extract model list for explicit mention
    model_list_text = ""
    if models_analyzed:
        model_list_text = f"\n\n**Models Analyzed:** {len(models_analyzed)} models found:\n"
        for i, model in enumerate(models_analyzed, 1):
            model_list_text += f"{i}. {model}\n"
    
    computed_values_preview = {
        k: v for k, v in list(computed_values.items())[:15]
    }
    
    # Enhanced prompt
    prompt = f"""You are a pharma analytics expert producing stakeholder insights.

USER QUERY: {state.get('user_query')}

ANALYSIS TYPE: {analysis_type}

MODELS FOUND: {len(models_analyzed)} models
{model_list_text}

=== ANALYSIS STORY (Main Findings) ===
{story}

=== STORY ELEMENTS (Structured Data) ===
{json.dumps(story_elements, indent=2, default=str)}

=== KEY METRICS AVAILABLE ===
{json.dumps(computed_values_preview, indent=2, default=str)}

{domain_context}

{viz_context}

{conv_context}

{personalized_context_section}

CRITICAL INSTRUCTIONS:

1. **ALWAYS List All Models First**
   - Start your response by explicitly listing all {len(models_analyzed)} models found
   - Use the models_analyzed list: {models_analyzed}
   - Format: "We analyzed X models: Model1, Model2, Model3..."

2. **Structure Your Response:**
   
   ## Models Found
   [List all models explicitly - REQUIRED]
   
   ## Executive Summary
   [2-3 sentences about key finding - which model is best?]
   
   ## Performance Breakdown
   [For each model or top 3-5 models, show their metrics]
   
   ## Key Findings
   [Bullet points from story_elements]
   
   ## Recommendations
   [Actionable next steps]

3. **Be Specific and Use Actual Values**
   - Quote actual model names from the list
   - Reference actual metric values from story_elements
   - Example: "XGBoost achieved RMSE of 32.5, outperforming Random Forest (RMSE: 35.2)"

4. **DO NOT use JSON or code blocks in your response**
   - Write in natural language
   - Use tables or bullet points for clarity
   - NO raw JSON dumps

5. **Reference the story for insights**
   - The story already contains the analysis
   - Extract model names and values from story_elements

EXAMPLE - GOOD STRUCTURE:

## Models Analyzed

We found 7 models for NRx forecasting:
1. NRx_Ensemble_Boosting_v1
2. NRx_XGBoost_v3
3. NRx_Ensemble_Voting_v1
4. NRx_Ensemble_Stacking_v2
5. NRx_LightGBM_v2
6. NRx_RandomForest_v2
7. NRx_LinearRegression_v1

## Executive Summary

Analysis of 7 NRx forecasting models shows that **NRx_Ensemble_Boosting_v1** achieved the best overall performance with RMSE of 39.27 and MAE of 23.60. The **NRx_Ensemble_Voting_v1** model leads in R² (0.81), indicating superior explanatory power.

## Performance Breakdown

**Top Performers:**
- **NRx_Ensemble_Boosting_v1**: RMSE 39.27 | MAE 23.60 | R² 0.76 ⭐ Best RMSE
- **NRx_Ensemble_Voting_v1**: RMSE 42.75 | MAE 26.48 | R² 0.81 ⭐ Best R²
- **NRx_XGBoost_v3**: RMSE 42.18 | MAE 30.57 | R² 0.72

**Other Models:**
- NRx_Ensemble_Stacking_v2: RMSE 43.23
- NRx_LightGBM_v2: RMSE 46.09
- NRx_RandomForest_v2: RMSE 49.60
- NRx_LinearRegression_v1: RMSE 59.25

## Key Findings
[Rest of analysis...]

Now generate insights following this structure for the actual data provided above.
"""

    # Call LLM
    try:
        llm_response = llm.invoke(prompt)
        generated_insights = llm_response.content
        
        print(f"[Insight Generation] ✓ Generated {len(generated_insights)} chars")
        
        # Validate insights quality
        if not generated_insights or len(generated_insights) < 100:
            print(f"[Insight Generation] ✗ WARNING: Generated insights too short!")
        
        # Check if models are mentioned
        models_mentioned = sum(1 for model in models_analyzed if model in generated_insights)
        print(f"[Insight Generation] Models mentioned in output: {models_mentioned}/{len(models_analyzed)}")
        
        if models_mentioned < len(models_analyzed) * 0.5:  # Less than 50% mentioned
            print(f"[Insight Generation] ⚠ WARNING: Many models not mentioned in insights!")
        
    except Exception as e:
        print(f"[Insight Generation] ✗ LLM failed: {e}")
        
        # Fallback
        generated_insights = f"""## Models Analyzed

We found {len(models_analyzed)} models:
{chr(10).join(f'{i}. {model}' for i, model in enumerate(models_analyzed, 1))}

## Analysis Results

{story}

Please review the findings above for detailed performance metrics.
"""

    # ========== FORMAT RAW METRICS AS NICE TABLES (NOT JSON) ==========
    if raw_metrics:
        metrics_display = "\n\n---\n\n## 📊 Detailed Performance Metrics\n\n"
        
        for metric_name, metric_data in raw_metrics.items():
            # Clean up metric name
            display_name = metric_name.replace('_', ' ').title()
            metrics_display += f"### {display_name}\n\n"
            
            if isinstance(metric_data, dict):
                try:
                    # Check if it's a nested dict (model -> metrics)
                    sample_value = list(metric_data.values())[0] if metric_data else None
                    
                    if isinstance(sample_value, dict):
                        # Nested structure: Convert to DataFrame and table
                        df = pd.DataFrame(metric_data).T
                        
                        # Format numeric columns
                        for col in df.columns:
                            if col in ['mean', 'std', 'min', 'max']:
                                df[col] = df[col].astype(float).round(4)
                        
                        # Convert to markdown table
                        metrics_display += df.to_markdown() + "\n\n"
                    else:
                        # Simple key-value dict
                        # Sort by value (ascending for errors, descending for scores)
                        lower_better = any(term in metric_name.lower() for term in ['rmse', 'mae', 'mse', 'error'])
                        sorted_items = sorted(metric_data.items(), 
                                            key=lambda x: float(str(x[1]).replace(',', '')) if isinstance(x[1], (int, float, str)) else 0,
                                            reverse=not lower_better)
                        
                        # Create a nice table
                        table_lines = ["| Rank | Model | Value |", "|------|-------|-------|"]
                        for rank, (model, value) in enumerate(sorted_items, 1):
                            # Format value
                            if isinstance(value, float):
                                value_str = f"{value:.4f}"
                            elif isinstance(value, str) and '%' in value:
                                value_str = value
                            else:
                                value_str = str(value)
                            
                            # Add emoji for top 3
                            emoji = ""
                            if rank == 1:
                                emoji = " 🥇"
                            elif rank == 2:
                                emoji = " 🥈"
                            elif rank == 3:
                                emoji = " 🥉"
                            
                            table_lines.append(f"| {rank}{emoji} | {model} | {value_str} |")
                        
                        metrics_display += "\n".join(table_lines) + "\n\n"
                
                except Exception as e:
                    print(f"[Insight] Error formatting {metric_name}: {e}")
                    # Fallback: Simple list
                    for key, val in list(metric_data.items())[:10]:
                        metrics_display += f"- **{key}**: {val}\n"
                    metrics_display += "\n"
            else:
                metrics_display += f"{metric_data}\n\n"
        
        # Append formatted metrics to insights
        generated_insights += metrics_display

    # Store in memory
    try:
        from core.memory_manager import get_memory_manager
        mem = get_memory_manager()
        mem.store_insight(
            session_id=state.get("session_id","default_session"),
            turn_number=state.get("turn_number",1),
            user_query=state.get("user_query",""),
            insight_text=generated_insights
        )
    except Exception as e:
        print(f"[Insight Generation] Memory store failed: {e}")

    extracted_models = extract_model_names_from_text(generated_insights)

    print(f"[Insight Generation] ✓ Complete")
    print(f"{'='*80}\n")

    return {
        "messages": [AIMessage(content=generated_insights)],
        "final_insights": generated_insights,
        "rendered_charts": rendered_charts,
        "execution_path": execution_path,
        "mentioned_models": extracted_models or None
    }

def orchestrator_agent(state: AgentState) -> dict:
    needs_clarification = state.get('needs_clarification', False)
    loop_count = state.get('loop_count', 0)
    clarification_attempts = state.get('clarification_attempts', 0)
    needs_memory = state.get('needs_memory', False)
    needs_database = state.get('needs_database', True)
    
    execution_path = state.get('execution_path', [])
    execution_path.append('orchestrator')

    if clarification_attempts >= 3:
        print(f"Max clarification attempts reached, proceeding")
        needs_clarification = False
    
    if loop_count > 16:
        return {
            "next_action": "end",
            "execution_path": execution_path,
            "messages": [AIMessage(content="Maximum conversation depth reached.")]
        }
    
    if needs_clarification:
        clarification = state.get('clarification_question', "Could you please provide more details?")
        return {
            "next_action": "ask_clarification",
            "execution_path": execution_path,
            "clarification_attempts": clarification_attempts + 1,
            "messages": [AIMessage(content=clarification)]
        }
    
    # Route based on memory and database needs
    if needs_memory:
        # Need to retrieve conversation history
        return {
            "next_action": "retrieve_memory",
            "execution_path": execution_path,
            "loop_count": loop_count + 1
        }
    elif needs_database:
        # Skip memory, go straight to SQL
        return {
            "next_action": "skip_to_sql",
            "execution_path": execution_path,
            "loop_count": loop_count + 1
        }
    else:
        # Ambiguous query
        return {
            "next_action": "ask_clarification",
            "execution_path": execution_path,
            "needs_clarification": True,
            "clarification_question": "I need more information. What would you like to know?",
            "messages": [AIMessage(content="I need more information. What would you like to know?")]
        }