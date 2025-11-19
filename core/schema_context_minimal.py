"""
Minimal core schema that's ALWAYS injected into SQL generation prompts.
Contains essential structure without overwhelming detail.
"""

CORE_SCHEMA_ALWAYS_PRESENT = """
# DATABASE SCHEMA - ESSENTIAL STRUCTURE

## CORE TABLES (Primary entities)

1. **models**: model_id (PK), model_name, model_type, use_case, algorithm, version, is_active
2. **model_executions**: execution_id (PK), model_id (FK), execution_timestamp, execution_status
3. **performance_metrics**: metric_id (PK), execution_id (FK), metric_name, metric_value, data_split
4. **predictions**: prediction_id (PK), execution_id (FK), entity_id, prediction_value, actual_value
5. **feature_importance**: importance_id (PK), execution_id (FK), feature_name, importance_score, rank
6. **drift_detection_results**: drift_id (PK), execution_id (FK), drift_score, is_significant
7. **ensemble_members**: ensemble_id (FK), base_model_id (FK), weight, role

## KEY VIEWS (Use these for convenience)

- **latest_model_executions**: Latest successful execution per model
- **model_performance_summary**: Pre-joined models + latest metrics
- **ensemble_vs_base_performance**: Ensemble vs base comparison

## CRITICAL JOIN PATH

models → latest_model_executions → performance_metrics/predictions/feature_importance

## MANDATORY RULES (ALWAYS enforce)

1. **JOIN Type**: ALWAYS use LEFT JOIN (NEVER INNER JOIN)
2. **Active Filter**: ALWAYS include WHERE m.is_active = true
3. **Metrics Format**: performance_metrics is LONG format - use AVG() with FILTER:
```sql
   AVG(pm.metric_value) FILTER (WHERE pm.metric_name = 'rmse' AND pm.data_split = 'test') as avg_test_rmse
```
4. **Latest Data**: Use latest_model_executions view (not model_executions directly)
5. **NULL Safety**: Use NULLS LAST in ORDER BY, HAVING COUNT(...) > 0 after GROUP BY
6. **Case Insensitive**: Use ILIKE for string matching (e.g., ILIKE '%xgboost%')
7. **Limit Results**: Always add LIMIT 100 (or appropriate number)
8. **Data Split Default**: Use data_split = 'test' unless specified otherwise

## COMMON METRIC NAMES

- Regression: rmse, mae, r2_score, mse, mape
- Classification: auc_roc, accuracy, precision, recall, f1_score

## COMMON USE CASES

NRx_forecasting, HCP_engagement, feature_importance_analysis, model_drift_detection, 
messaging_optimization, territory_performance_forecasting, market_share_prediction

---
IMPORTANT: The above is the CORE structure. Additional detailed schema for specific tables 
will be provided below based on your query requirements.
---
"""


# Dependency map - which tables require which other tables for JOINs
TABLE_DEPENDENCIES = {
    'performance_metrics': ['model_executions', 'models'],
    'predictions': ['model_executions', 'models'],
    'feature_importance': ['model_executions', 'models'],
    'drift_detection_results': ['model_executions', 'models'],
    'ensemble_members': ['models'],  # Needs models table twice (ensemble + base)
}


def get_table_dependencies(table_names):
    """
    Get all dependencies for a list of tables.
    
    Args:
        table_names: List of table names
        
    Returns:
        Set of all required tables (including dependencies)
    """
    all_tables = set(table_names)
    
    for table in table_names:
        if table in TABLE_DEPENDENCIES:
            all_tables.update(TABLE_DEPENDENCIES[table])
    
    return list(all_tables)