"""
Enhanced drift query detection and SQL generation helper
Add this to your codebase to fix drift-related queries
"""

def detect_drift_query_intent(user_query: str, parsed_intent: dict) -> dict:
    """
    Enhanced drift query detection
    
    Args:
        user_query: User's query string
        parsed_intent: Already parsed intent from query_understanding_agent
        
    Returns:
        Enhanced intent with drift-specific fields
    """
    query_lower = user_query.lower()
    
    # Drift indicators
    drift_keywords = [
        'drift', 'drifting', 'degradation', 'degraded', 
        'performance change', 'model stability', 'concept drift',
        'data drift', 'distribution shift'
    ]
    
    is_drift_query = any(keyword in query_lower for keyword in drift_keywords)
    
    if not is_drift_query:
        return parsed_intent
    
    # Enhance parsed intent
    enhanced = parsed_intent.copy()
    enhanced['comparison_type'] = 'drift'
    enhanced['requires_drift_analysis'] = True
    
    # Detect specific drift type
    if 'concept drift' in query_lower:
        enhanced['drift_type'] = 'concept_drift'
    elif 'data drift' in query_lower:
        enhanced['drift_type'] = 'data_drift'
    elif 'performance' in query_lower:
        enhanced['drift_type'] = 'performance_drift'
    else:
        enhanced['drift_type'] = 'all'  # Check all drift types
    
    # Detect time range if mentioned
    if 'last' in query_lower:
        if 'month' in query_lower:
            enhanced['time_range'] = {'period': 'last_month'}
        elif 'week' in query_lower:
            enhanced['time_range'] = {'period': 'last_week'}
        elif '6 months' in query_lower or 'six months' in query_lower:
            enhanced['time_range'] = {'period': 'last_6_months'}
    
    return enhanced


def generate_drift_sql(parsed_intent: dict, use_case: str = None) -> str:
    """
    Generate optimized SQL for drift queries
    
    Args:
        parsed_intent: Parsed intent with drift fields
        use_case: Use case filter
        
    Returns:
        SQL query string
    """
    drift_type = parsed_intent.get('drift_type', 'all')
    models_requested = parsed_intent.get('models_requested', [])
    time_range = parsed_intent.get('time_range', {})
    
    # Base query - get drift detection results with model info
    sql_parts = [
        "SELECT",
        "    m.model_name,",
        "    m.model_type,",
        "    m.algorithm,",
        "    m.use_case,",
        "    ddr.drift_type,",
        "    ddr.drift_metric,",
        "    ddr.drift_score,",
        "    ddr.threshold_value,",
        "    ddr.is_significant,",
        "    ddr.drift_explanation,",
        "    ddr.detected_at,",
        "    me.execution_timestamp,",
        "    me.execution_id,",
        "    me.baseline_execution_id",
        "FROM drift_detection_results ddr",
        "JOIN model_executions me ON ddr.execution_id = me.execution_id",
        "JOIN models m ON me.model_id = m.model_id",
        "WHERE m.is_active = true"
    ]
    
    # Add use case filter
    if use_case:
        sql_parts.append(f"  AND m.use_case ILIKE '%{use_case}%'")
    
    # Add drift type filter
    if drift_type != 'all':
        sql_parts.append(f"  AND ddr.drift_type = '{drift_type}'")
    
    # Add model filter if specified
    if models_requested:
        model_conditions = " OR ".join([
            f"m.model_name ILIKE '%{model}%'" for model in models_requested
        ])
        sql_parts.append(f"  AND ({model_conditions})")
    
    # Add time range filter
    if time_range:
        period = time_range.get('period')
        if period == 'last_month':
            sql_parts.append("  AND ddr.detected_at >= NOW() - INTERVAL '1 month'")
        elif period == 'last_week':
            sql_parts.append("  AND ddr.detected_at >= NOW() - INTERVAL '1 week'")
        elif period == 'last_6_months':
            sql_parts.append("  AND ddr.detected_at >= NOW() - INTERVAL '6 months'")
    
    # Order by most recent first
    sql_parts.append("ORDER BY ddr.detected_at DESC")
    sql_parts.append("LIMIT 100;")
    
    return "\n".join(sql_parts)


def generate_drift_trend_sql(parsed_intent: dict, use_case: str = None) -> str:
    """
    Generate SQL for drift trends over time
    
    Args:
        parsed_intent: Parsed intent
        use_case: Use case filter
        
    Returns:
        SQL query for drift trends
    """
    models_requested = parsed_intent.get('models_requested', [])
    
    sql_parts = [
        "SELECT",
        "    m.model_name,",
        "    m.use_case,",
        "    DATE_TRUNC('month', ddr.detected_at) as month,",
        "    COUNT(*) as drift_detections,",
        "    AVG(ddr.drift_score) as avg_drift_score,",
        "    COUNT(*) FILTER (WHERE ddr.is_significant = true) as significant_drifts,",
        "    STRING_AGG(DISTINCT ddr.drift_type, ', ') as drift_types",
        "FROM drift_detection_results ddr",
        "JOIN model_executions me ON ddr.execution_id = me.execution_id",
        "JOIN models m ON me.model_id = m.model_id",
        "WHERE m.is_active = true",
        "  AND ddr.detected_at >= NOW() - INTERVAL '6 months'"
    ]
    
    if use_case:
        sql_parts.append(f"  AND m.use_case ILIKE '%{use_case}%'")
    
    if models_requested:
        model_conditions = " OR ".join([
            f"m.model_name ILIKE '%{model}%'" for model in models_requested
        ])
        sql_parts.append(f"  AND ({model_conditions})")
    
    sql_parts.extend([
        "GROUP BY m.model_name, m.use_case, DATE_TRUNC('month', ddr.detected_at)",
        "ORDER BY month DESC, m.model_name",
        "LIMIT 100;"
    ])
    
    return "\n".join(sql_parts)


def generate_performance_degradation_sql(use_case: str = None) -> str:
    """
    Generate SQL to detect performance degradation over time
    
    Args:
        use_case: Use case filter
        
    Returns:
        SQL query for performance degradation analysis
    """
    sql = """
SELECT 
    m.model_name,
    m.model_type,
    m.algorithm,
    m.use_case,
    pm_current.metric_name,
    pm_current.metric_value as current_value,
    pm_baseline.metric_value as baseline_value,
    (pm_current.metric_value - pm_baseline.metric_value) as absolute_change,
    CASE 
        WHEN pm_baseline.metric_value != 0 THEN
            ((pm_current.metric_value - pm_baseline.metric_value) / ABS(pm_baseline.metric_value) * 100)
        ELSE 0
    END as percent_change,
    me_current.execution_timestamp as current_execution,
    me_baseline.execution_timestamp as baseline_execution,
    CASE 
        WHEN pm_current.metric_name IN ('rmse', 'mae', 'mse', 'mape') THEN
            CASE WHEN pm_current.metric_value > pm_baseline.metric_value THEN 'degraded' ELSE 'improved' END
        WHEN pm_current.metric_name IN ('r2_score', 'auc_roc', 'accuracy', 'precision', 'recall', 'f1_score') THEN
            CASE WHEN pm_current.metric_value < pm_baseline.metric_value THEN 'degraded' ELSE 'improved' END
        ELSE 'unknown'
    END as performance_trend
FROM models m
-- Get latest execution per model
JOIN LATERAL (
    SELECT execution_id, execution_timestamp, model_id
    FROM model_executions
    WHERE model_id = m.model_id
      AND execution_status = 'success'
    ORDER BY execution_timestamp DESC
    LIMIT 1
) me_current ON true
-- Get baseline execution (6 months ago)
JOIN LATERAL (
    SELECT execution_id, execution_timestamp
    FROM model_executions
    WHERE model_id = m.model_id
      AND execution_status = 'success'
      AND execution_timestamp <= NOW() - INTERVAL '6 months'
    ORDER BY execution_timestamp DESC
    LIMIT 1
) me_baseline ON true
-- Get current metrics
JOIN performance_metrics pm_current ON me_current.execution_id = pm_current.execution_id
-- Get baseline metrics
JOIN performance_metrics pm_baseline ON me_baseline.execution_id = pm_baseline.execution_id
    AND pm_baseline.metric_name = pm_current.metric_name
    AND pm_baseline.data_split = pm_current.data_split
WHERE m.is_active = true
  AND pm_current.data_split = 'test'
  AND pm_current.metric_name IN ('rmse', 'mae', 'r2_score', 'auc_roc', 'accuracy')
"""
    
    if use_case:
        sql += f"  AND m.use_case ILIKE '%{use_case}%'\n"
    
    sql += """
ORDER BY 
    CASE 
        WHEN performance_trend = 'degraded' THEN 1
        ELSE 2
    END,
    ABS(percent_change) DESC
LIMIT 100;
"""
    
    return sql


# ============================================================================
# Integration functions for nodes.py
# ============================================================================

def should_use_drift_sql(state) -> bool:
    """
    Determine if query should use drift-specific SQL
    
    Returns:
        True if drift query detected
    """
    comparison_type = state.get('comparison_type', '')
    user_query = state.get('user_query', '').lower()
    
    drift_indicators = ['drift', 'degradation', 'performance change', 'stability']
    
    return (
        comparison_type == 'drift' or
        any(indicator in user_query for indicator in drift_indicators)
    )


def get_drift_sql_query(state) -> str:
    """
    Get appropriate drift SQL based on query type
    
    Returns:
        SQL query string
    """
    user_query = state.get('user_query', '').lower()
    parsed_intent = state.get('parsed_intent', {})
    use_case = state.get('use_case')
    
    # Detect if asking for trends/history
    if any(word in user_query for word in ['trend', 'over time', 'history', 'last 6 months']):
        return generate_drift_trend_sql(parsed_intent, use_case)
    
    # Detect if asking about performance degradation
    if any(word in user_query for word in ['degradation', 'performance change', 'worse', 'declining']):
        return generate_performance_degradation_sql(use_case)
    
    # Default: drift detection results
    return generate_drift_sql(parsed_intent, use_case)