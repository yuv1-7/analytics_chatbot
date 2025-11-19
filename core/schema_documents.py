"""
Schema documents for Pinecone embedding.
Each table/view/pattern is a separate document for semantic retrieval.
"""

SCHEMA_DOCUMENTS = [
    # ========================================================================
    # CORE TABLES
    # ========================================================================
    
    {
        "doc_id": "schema_models",
        "category": "schema",
        "table_name": "models",
        "content": """
# models table

Stores all model definitions (base models and ensembles).

## Columns:
- model_id (UUID, PK): Unique identifier
- model_name (VARCHAR): Name of the model (e.g., 'NRx_XGBoost_v2', 'Ensemble_Stacking_v1')
- model_type (VARCHAR): 'base_model' or 'ensemble'
- use_case (VARCHAR): Business use case
  * 'NRx_forecasting', 'HCP_engagement', 'feature_importance_analysis', 
  * 'model_drift_detection', 'messaging_optimization', 'territory_performance_forecasting',
  * 'market_share_prediction', 'hcp_clustering_segmentation', 'price_sensitivity_analysis',
  * 'competitor_share_forecasting'
- version (VARCHAR): Model version (e.g., 'v1.0', 'v2.1')
- algorithm (VARCHAR): Algorithm used
  * Base: 'Random Forest', 'XGBoost', 'LightGBM', 'Logistic Regression', 'SVM', 'Neural Network'
  * Ensemble: 'Stacking', 'Boosting', 'Bagging', 'Voting', 'Blending'
- hyperparameters (JSONB): Model hyperparameters
- is_active (BOOLEAN): Active status - ALWAYS filter by is_active = true
- created_at (TIMESTAMP): Creation timestamp
- parent_model_id (UUID, FK → models): Parent model reference

## Relationships:
- One-to-many with model_executions (model_id)
- Many-to-many with ensemble_members (for ensemble models)

## Common Filters:
- WHERE is_active = true (MANDATORY for production queries)
- WHERE use_case ILIKE '%nrx%' (case-insensitive search)
- WHERE model_type = 'base_model' (exclude ensembles)
- WHERE model_name ILIKE '%xgboost%' (fuzzy name matching)

## Example Usage:
SELECT model_id, model_name, algorithm, use_case
FROM models
WHERE is_active = true
  AND use_case = 'NRx_forecasting';
        """,
        "metadata": {
            "joins_to": ["model_executions", "ensemble_members", "version_comparisons"],
            "common_use_cases": ["NRx_forecasting", "HCP_engagement", "all"],
            "always_filter": ["is_active = true"]
        },
        "keywords": ["models", "model_name", "algorithm", "use_case", "base_model", "ensemble"]
    },
    
    {
        "doc_id": "schema_model_executions",
        "category": "schema",
        "table_name": "model_executions",
        "content": """
# model_executions table

Tracks each execution/run of a model. This is the JOIN hub connecting models to metrics and predictions.

## Columns:
- execution_id (UUID, PK): Unique execution identifier
- model_id (UUID, FK → models): Model that was executed
- execution_timestamp (TIMESTAMP): When executed
- execution_status (VARCHAR): 'success', 'failed', 'running', 'pending'
- training_data_start_date (DATE): Training data start
- training_data_end_date (DATE): Training data end
- prediction_date (DATE): Date predictions were made for
- execution_metadata (JSONB): Additional metadata
- runtime_seconds (INT): Execution duration
- drift_detected (BOOLEAN): Whether drift was detected
- drift_score (FLOAT): Drift score
- baseline_execution_id (UUID, FK → model_executions): Baseline for drift comparison

## Relationships:
- Many-to-one with models (model_id)
- One-to-many with performance_metrics (execution_id)
- One-to-many with predictions (execution_id)
- One-to-many with feature_importance (execution_id)

## JOIN Pattern:
To get model metrics:
models → model_executions → performance_metrics

To get model predictions:
models → model_executions → predictions

## Common Filters:
- WHERE execution_status = 'success' (only successful runs)
- WHERE execution_timestamp >= NOW() - INTERVAL '6 months' (recent executions)

## Example Usage:
SELECT me.execution_id, me.execution_timestamp, me.execution_status
FROM model_executions me
WHERE me.model_id = 'some-uuid'
  AND me.execution_status = 'success'
ORDER BY me.execution_timestamp DESC;
        """,
        "metadata": {
            "joins_to": ["models", "performance_metrics", "predictions", "feature_importance", "drift_detection_results"],
            "join_hub": True,
            "common_filters": ["execution_status = 'success'"]
        },
        "keywords": ["executions", "runs", "execution_timestamp", "drift", "baseline"]
    },
    
    {
        "doc_id": "schema_performance_metrics",
        "category": "schema",
        "table_name": "performance_metrics",
        "content": """
# performance_metrics table

Model performance metrics per execution. LONG FORMAT: one row per metric per execution.

## Columns:
- metric_id (UUID, PK): Unique metric identifier
- execution_id (UUID, FK → model_executions): Associated execution
- metric_name (VARCHAR): Metric name
  * Regression: 'rmse', 'mae', 'r2_score', 'mse', 'mape'
  * Classification: 'auc_roc', 'accuracy', 'precision', 'recall', 'f1_score'
- metric_value (FLOAT): Metric value
- data_split (VARCHAR): 'train', 'validation', 'test', 'holdout'
- calculated_at (TIMESTAMP): Calculation timestamp

## IMPORTANT: LONG FORMAT
This table stores metrics in LONG format (one row per metric).
To get WIDE format (one row per model with multiple metric columns), use aggregation:

SELECT 
    m.model_name,
    AVG(pm.metric_value) FILTER (WHERE pm.metric_name = 'rmse' AND pm.data_split = 'test') as avg_test_rmse,
    AVG(pm.metric_value) FILTER (WHERE pm.metric_name = 'r2_score' AND pm.data_split = 'test') as avg_test_r2
FROM models m
LEFT JOIN model_executions me ON m.model_id = me.model_id
LEFT JOIN performance_metrics pm ON me.execution_id = pm.execution_id
GROUP BY m.model_name;

## Relationships:
- Many-to-one with model_executions (execution_id)

## JOIN Path:
models → model_executions → performance_metrics

## Common Filters:
- WHERE data_split = 'test' (test set metrics)
- WHERE metric_name IN ('rmse', 'r2_score', 'mae') (specific metrics)

## Metric Interpretation:
- RMSE, MAE, MSE, MAPE: Lower is better
- R2, AUC_ROC, Accuracy, Precision, Recall, F1: Higher is better

## Example Usage:
-- Get test RMSE for all models
SELECT 
    m.model_name,
    AVG(pm.metric_value) as avg_rmse
FROM models m
LEFT JOIN model_executions me ON m.model_id = me.model_id
LEFT JOIN performance_metrics pm ON me.execution_id = pm.execution_id
WHERE pm.metric_name = 'rmse'
  AND pm.data_split = 'test'
  AND m.is_active = true
GROUP BY m.model_name;
        """,
        "metadata": {
            "joins_through": ["model_executions"],
            "format": "LONG",
            "aggregation_required": True,
            "common_metrics": ["rmse", "mae", "r2_score", "auc_roc", "accuracy"],
            "data_splits": ["train", "test", "validation"]
        },
        "keywords": ["metrics", "performance", "rmse", "mae", "r2", "auc", "accuracy", "test", "validation"]
    },
    
    {
        "doc_id": "schema_predictions",
        "category": "schema",
        "table_name": "predictions",
        "content": """
# predictions table

Prediction results for entities (HCP, territory, segment, product).

## Columns:
- prediction_id (UUID, PK): Unique identifier
- execution_id (UUID, FK → model_executions): Associated execution
- entity_type (VARCHAR): 'HCP', 'territory', 'segment', 'product'
- entity_id (VARCHAR): Entity identifier (e.g., 'HCP_000001', 'TERR_0001')
- prediction_value (FLOAT): Predicted value
- prediction_probability (FLOAT): Probability (0-1) for classification
- prediction_class (VARCHAR): Predicted class for classification
- actual_value (FLOAT): Actual observed value (if available)
- residual (FLOAT): actual_value - prediction_value
- prediction_date (DATE): Prediction date
- confidence_interval_lower (FLOAT): Lower confidence bound
- confidence_interval_upper (FLOAT): Upper confidence bound

## Relationships:
- Many-to-one with model_executions (execution_id)

## JOIN Path:
models → model_executions → predictions

## Common Filters:
- WHERE entity_type = 'HCP' (filter by entity type)
- WHERE prediction_date >= '2024-01-01' (date range)

## Example Usage:
SELECT 
    p.entity_id,
    p.prediction_value,
    p.actual_value,
    ABS(p.residual) as absolute_error
FROM predictions p
JOIN model_executions me ON p.execution_id = me.execution_id
WHERE me.model_id = 'some-uuid'
  AND p.entity_type = 'HCP'
ORDER BY ABS(p.residual) DESC
LIMIT 20;
        """,
        "metadata": {
            "joins_through": ["model_executions"],
            "entity_types": ["HCP", "territory", "segment", "product"]
        },
        "keywords": ["predictions", "forecast", "entity", "HCP", "territory", "actual", "residual"]
    },
    
    {
        "doc_id": "schema_feature_importance",
        "category": "schema",
        "table_name": "feature_importance",
        "content": """
# feature_importance table

Feature importance scores per execution.

## Columns:
- importance_id (UUID, PK): Unique identifier
- execution_id (UUID, FK → model_executions): Associated execution
- feature_name (VARCHAR): Feature name
- importance_score (FLOAT): Importance value (0-1 range typically)
- importance_type (VARCHAR): 'gain', 'split', 'shap', 'permutation', 'weight'
- rank (INT): Feature rank (1 = most important)
- standard_error (FLOAT): Standard error

## Relationships:
- Many-to-one with model_executions (execution_id)

## JOIN Path:
models → model_executions → feature_importance

## Common Filters:
- WHERE rank <= 10 (top 10 features)
- WHERE importance_type = 'shap' (specific type)

## Example Usage:
SELECT 
    fi.feature_name,
    fi.importance_score,
    fi.rank
FROM feature_importance fi
JOIN model_executions me ON fi.execution_id = me.execution_id
WHERE me.model_id = 'some-uuid'
  AND fi.rank <= 10
ORDER BY fi.rank;
        """,
        "metadata": {
            "joins_through": ["model_executions"],
            "importance_types": ["gain", "split", "shap", "permutation"]
        },
        "keywords": ["features", "importance", "drivers", "shap", "rank", "top features"]
    },
    
    {
        "doc_id": "schema_drift_detection",
        "category": "schema",
        "table_name": "drift_detection_results",
        "content": """
# drift_detection_results table

Model drift detection results.

## Columns:
- drift_id (UUID, PK): Unique identifier
- execution_id (UUID, FK → model_executions): Current execution
- baseline_execution_id (UUID, FK → model_executions): Baseline execution
- drift_type (VARCHAR): 'concept_drift', 'data_drift', 'performance_drift'
- drift_metric (VARCHAR): Metric used (e.g., 'PSI', 'KS_statistic', 'RMSE_delta')
- drift_score (FLOAT): Drift score (higher = more drift)
- threshold_value (FLOAT): Threshold used (typically 0.10)
- is_significant (BOOLEAN): Whether drift is significant
- affected_features (JSONB): Features affected by drift
- drift_explanation (TEXT): Explanation

## Relationships:
- Many-to-one with model_executions (execution_id)

## JOIN Path:
models → model_executions → drift_detection_results

## Common Filters:
- WHERE is_significant = true (significant drift only)
- WHERE drift_score > 0.10 (above threshold)

## Example Usage:
SELECT 
    m.model_name,
    ddr.drift_type,
    ddr.drift_score,
    ddr.is_significant
FROM drift_detection_results ddr
JOIN model_executions me ON ddr.execution_id = me.execution_id
JOIN models m ON me.model_id = m.model_id
WHERE ddr.is_significant = true;
        """,
        "metadata": {
            "joins_through": ["model_executions"],
            "drift_types": ["concept_drift", "data_drift", "performance_drift"]
        },
        "keywords": ["drift", "detection", "performance degradation", "monitoring"]
    },
    
    {
        "doc_id": "schema_ensemble_members",
        "category": "schema",
        "table_name": "ensemble_members",
        "content": """
# ensemble_members table

Junction table linking ensemble models to base models.

## Columns:
- ensemble_id (UUID, PK, FK → models): Ensemble model ID
- base_model_id (UUID, PK, FK → models): Base model ID
- weight (FLOAT): Model weight in ensemble (0-1)
- role (VARCHAR): 'base_learner', 'meta_learner', 'stacking_layer'
- ensemble_type (VARCHAR): 'boosting', 'bagging', 'stacking', 'blending', 'voting'

## Relationships:
- Many-to-one with models (ensemble_id)
- Many-to-one with models (base_model_id)

## JOIN Pattern:
To get ensemble composition:
models (ensemble) → ensemble_members → models (base)

## Example Usage:
SELECT 
    e.model_name as ensemble_name,
    b.model_name as base_model_name,
    em.weight,
    em.role
FROM models e
JOIN ensemble_members em ON e.model_id = em.ensemble_id
JOIN models b ON em.base_model_id = b.model_id
WHERE e.model_type = 'ensemble'
  AND e.is_active = true;
        """,
        "metadata": {
            "joins_to": ["models"],
            "many_to_many": True
        },
        "keywords": ["ensemble", "base models", "composition", "stacking", "boosting", "weights"]
    },
    
    # ========================================================================
    # VIEWS (Pre-joined for convenience)
    # ========================================================================
    
    {
        "doc_id": "schema_view_latest_executions",
        "category": "schema",
        "table_name": "latest_model_executions",
        "view": True,
        "content": """
# latest_model_executions VIEW

Pre-computed view showing the latest successful execution per model.

## Columns:
- model_id (UUID): Model identifier
- execution_id (UUID): Latest execution identifier
- execution_timestamp (TIMESTAMP): When executed
- execution_status (VARCHAR): Status (always 'success' in this view)
- prediction_date (DATE): Prediction date

## Purpose:
Use this view instead of joining models → model_executions when you only need the latest execution.

## Usage:
SELECT m.model_name, lme.execution_timestamp
FROM models m
JOIN latest_model_executions lme ON m.model_id = lme.model_id
WHERE m.is_active = true;

## Replaces:
SELECT m.model_name, me.execution_timestamp
FROM models m
JOIN model_executions me ON m.model_id = me.model_id
WHERE me.execution_status = 'success'
  AND me.execution_timestamp = (
    SELECT MAX(execution_timestamp)
    FROM model_executions
    WHERE model_id = m.model_id
  );
        """,
        "metadata": {
            "replaces_join": ["models", "model_executions"],
            "filters_applied": ["execution_status = 'success'", "latest only"]
        },
        "keywords": ["latest", "most recent", "current", "execution"]
    },
    
    {
        "doc_id": "schema_view_performance_summary",
        "category": "schema",
        "table_name": "model_performance_summary",
        "view": True,
        "content": """
# model_performance_summary VIEW

Pre-joined view combining models with their latest metrics.

## Columns:
- model_id, model_name, model_type, use_case, version
- execution_id, execution_timestamp, drift_detected
- metric_name, metric_value, data_split

## Purpose:
Use this view for quick model + metrics queries without complex JOINs.

## Usage:
SELECT model_name, metric_name, metric_value
FROM model_performance_summary
WHERE use_case = 'NRx_forecasting'
  AND data_split = 'test'
  AND metric_name = 'rmse';

## Replaces:
3-table JOIN: models → latest_model_executions → performance_metrics
        """,
        "metadata": {
            "replaces_join": ["models", "latest_model_executions", "performance_metrics"],
            "quick_access": True
        },
        "keywords": ["summary", "quick", "pre-joined", "performance", "metrics"]
    },
    
    {
        "doc_id": "schema_view_ensemble_vs_base",
        "category": "schema",
        "table_name": "ensemble_vs_base_performance",
        "view": True,
        "content": """
# ensemble_vs_base_performance VIEW

Pre-computed ensemble vs base model performance comparison.

## Columns:
- model_name, use_case
- ensemble_rmse, avg_base_rmse
- ensemble_r2, avg_base_r2
- ensemble_auc, avg_base_auc
- ensemble_advantage (percentage improvement)

## Purpose:
Quick comparison of ensemble performance vs average base model performance.

## Usage:
SELECT model_name, ensemble_rmse, avg_base_rmse, ensemble_advantage
FROM ensemble_vs_base_performance
WHERE use_case = 'NRx_forecasting'
ORDER BY ensemble_advantage DESC;
        """,
        "metadata": {
            "comparison_type": "ensemble_vs_base",
            "aggregated": True
        },
        "keywords": ["ensemble", "base", "comparison", "advantage", "improvement"]
    },
    
    # ========================================================================
    # QUERY PATTERNS (Examples for specific use cases)
    # ========================================================================
    
    {
        "doc_id": "pattern_performance_comparison",
        "category": "query_pattern",
        "pattern_name": "Model Performance Comparison",
        "content": """
# Query Pattern: Compare Model Performance

Use this pattern to compare performance metrics across multiple models.

## CRITICAL: Use AGGREGATION with FILTER for WIDE format

The performance_metrics table is in LONG format (one row per metric).
To compare models side-by-side, you MUST aggregate using FILTER clause.

## Template:
SELECT 
    m.model_name,
    m.model_type,
    m.algorithm,
    COUNT(DISTINCT lme.execution_id) as execution_count,
    -- Regression metrics
    AVG(pm.metric_value) FILTER (WHERE pm.metric_name = 'rmse' AND pm.data_split = 'test') as avg_test_rmse,
    AVG(pm.metric_value) FILTER (WHERE pm.metric_name = 'mae' AND pm.data_split = 'test') as avg_test_mae,
    AVG(pm.metric_value) FILTER (WHERE pm.metric_name = 'r2_score' AND pm.data_split = 'test') as avg_test_r2,
    -- Classification metrics
    AVG(pm.metric_value) FILTER (WHERE pm.metric_name = 'auc_roc' AND pm.data_split = 'test') as avg_test_auc,
    AVG(pm.metric_value) FILTER (WHERE pm.metric_name = 'accuracy' AND pm.data_split = 'test') as avg_test_accuracy
FROM models m
LEFT JOIN latest_model_executions lme ON m.model_id = lme.model_id
LEFT JOIN performance_metrics pm ON lme.execution_id = pm.execution_id
WHERE m.is_active = true
  AND m.use_case ILIKE '%nrx%'
GROUP BY m.model_name, m.model_type, m.algorithm
HAVING COUNT(DISTINCT lme.execution_id) > 0
ORDER BY avg_test_rmse ASC NULLS LAST
LIMIT 100;

## Key Points:
1. Use AVG() with FILTER clause for each metric
2. Always GROUP BY model attributes
3. Use HAVING to ensure models have executions
4. Use NULLS LAST in ORDER BY for robustness
5. Include both regression and classification metrics (query will return NULL for non-applicable)
        """,
        "metadata": {
            "use_cases": ["NRx_forecasting", "HCP_engagement", "performance"],
            "tables_used": ["models", "latest_model_executions", "performance_metrics"],
            "complexity": "medium"
        },
        "keywords": ["compare", "performance", "metrics", "models", "rmse", "auc", "accuracy"]
    },
    
    {
        "doc_id": "pattern_feature_importance",
        "category": "query_pattern",
        "pattern_name": "Feature Importance Analysis",
        "content": """
# Query Pattern: Feature Importance Rankings

Get top features driving model predictions.

## Template:
SELECT 
    m.model_name,
    fi.feature_name,
    fi.importance_score,
    fi.rank,
    fi.importance_type
FROM models m
JOIN latest_model_executions lme ON m.model_id = lme.model_id
JOIN feature_importance fi ON lme.execution_id = fi.execution_id
WHERE m.is_active = true
  AND m.model_name ILIKE '%xgboost%'
  AND fi.rank <= 10
ORDER BY m.model_name, fi.rank;

## Key Points:
1. Use rank <= 10 to get top features
2. Join through latest_model_executions to get most recent importance
3. Can compare importance across models by removing model_name filter
        """,
        "metadata": {
            "use_cases": ["feature_importance_analysis"],
            "tables_used": ["models", "latest_model_executions", "feature_importance"],
            "complexity": "simple"
        },
        "keywords": ["features", "importance", "drivers", "top features", "rank"]
    },
    
    {
        "doc_id": "pattern_drift_detection",
        "category": "query_pattern",
        "pattern_name": "Drift Detection",
        "content": """
# Query Pattern: Identify Models with Drift

Find models experiencing performance or data drift.

## Template:
SELECT 
    m.model_name,
    m.use_case,
    ddr.drift_type,
    ddr.drift_score,
    ddr.threshold_value,
    ddr.is_significant,
    ddr.drift_explanation
FROM models m
JOIN latest_model_executions lme ON m.model_id = lme.model_id
JOIN drift_detection_results ddr ON lme.execution_id = ddr.execution_id
WHERE m.is_active = true
  AND ddr.is_significant = true
ORDER BY ddr.drift_score DESC;

## Key Points:
1. Filter by is_significant = true for actionable drift
2. Higher drift_score = more severe drift
3. drift_type indicates what changed (concept, data, performance)
        """,
        "metadata": {
            "use_cases": ["model_drift_detection"],
            "tables_used": ["models", "latest_model_executions", "drift_detection_results"],
            "complexity": "simple"
        },
        "keywords": ["drift", "detection", "monitoring", "performance degradation"]
    },
    
    {
        "doc_id": "pattern_ensemble_composition",
        "category": "query_pattern",
        "pattern_name": "Ensemble Composition",
        "content": """
# Query Pattern: Analyze Ensemble Composition

Show which base models make up an ensemble.

## Template:
SELECT 
    e.model_name as ensemble_name,
    b.model_name as base_model_name,
    b.algorithm,
    em.weight,
    em.role,
    em.ensemble_type
FROM models e
JOIN ensemble_members em ON e.model_id = em.ensemble_id
JOIN models b ON em.base_model_id = b.model_id
WHERE e.model_type = 'ensemble'
  AND e.is_active = true
  AND e.model_name ILIKE '%stacking%'
ORDER BY em.weight DESC;

## Key Points:
1. Join models table twice (once for ensemble, once for base)
2. weight shows contribution of each base model
3. Order by weight to see most important contributors
        """,
        "metadata": {
            "use_cases": ["ensemble_vs_base"],
            "tables_used": ["models", "ensemble_members"],
            "complexity": "medium"
        },
        "keywords": ["ensemble", "composition", "base models", "weights", "stacking"]
    },
    
    # ========================================================================
    # JOIN RELATIONSHIPS (Critical for complex queries)
    # ========================================================================
    
    {
        "doc_id": "joins_core_relationships",
        "category": "join_guide",
        "content": """
# Core Table Relationships

Understanding how tables connect is critical for correct SQL generation.

## Primary JOIN Paths:

### Path 1: Models → Metrics
models (model_id) 
  → latest_model_executions (model_id, execution_id)
  → performance_metrics (execution_id)

USE CASE: Get model performance metrics
ALWAYS USE: LEFT JOIN to avoid losing models without metrics

### Path 2: Models → Predictions
models (model_id)
  → latest_model_executions (model_id, execution_id)
  → predictions (execution_id)

USE CASE: Get model predictions by entity (HCP, territory)

### Path 3: Models → Features
models (model_id)
  → latest_model_executions (model_id, execution_id)
  → feature_importance (execution_id)

USE CASE: Get feature importance for models

### Path 4: Models → Drift
models (model_id)
  → latest_model_executions (model_id, execution_id)
  → drift_detection_results (execution_id)

USE CASE: Detect model drift

### Path 5: Ensemble → Base Models
models (ensemble, model_id as ensemble_id)
  → ensemble_members (ensemble_id, base_model_id)
  → models (base, model_id as base_model_id)

USE CASE: Analyze ensemble composition

## Critical Rules:
1. ALWAYS use LEFT JOIN (not INNER JOIN) to avoid losing rows
2. ALWAYS join through latest_model_executions (not model_executions directly)
3. ALWAYS filter by is_active = true on models table
4. For metrics: Use FILTER clause to separate by metric_name and data_split
        """,
        "metadata": {
            "category": "join_relationships",
            "critical": True
        },
        "keywords": ["joins", "relationships", "foreign keys", "paths"]
    },
    
    # ========================================================================
    # CRITICAL RULES (Always enforce)
    # ========================================================================
    
    {
        "doc_id": "rules_sql_generation",
        "category": "critical_rules",
        "content": """
# CRITICAL SQL GENERATION RULES

These rules MUST be followed for every query.

## 1. JOIN Type
❌ NEVER: INNER JOIN
✅ ALWAYS: LEFT JOIN

Reason: INNER JOIN eliminates rows when data is missing.
Example: A model with no metrics would disappear with INNER JOIN.

## 2. Active Models Filter
❌ NEVER: Omit is_active filter
✅ ALWAYS: WHERE m.is_active = true

Reason: Deprecated models should not appear in results.

## 3. Metrics Aggregation
❌ NEVER: SELECT avg_test_rmse FROM performance_metrics
✅ ALWAYS: Use AVG() with FILTER clause

Reason: performance_metrics is LONG format (one row per metric).

Correct pattern:
AVG(pm.metric_value) FILTER (WHERE pm.metric_name = 'rmse' AND pm.data_split = 'test') as avg_test_rmse

## 4. Latest Executions
❌ RARELY: Join model_executions directly
✅ USUALLY: Use latest_model_executions view

Reason: Most queries need latest execution, not all executions.

## 5. NULL Handling
✅ ALWAYS: Use NULLS LAST in ORDER BY
✅ ALWAYS: Use HAVING clause after GROUP BY to ensure data exists

Example:
HAVING COUNT(DISTINCT lme.execution_id) > 0

## 6. Case-Insensitive Matching
❌ NEVER: WHERE model_name = 'xgboost'
✅ ALWAYS: WHERE model_name ILIKE '%xgboost%'

Reason: Model names may have different casing or formatting.

## 7. LIMIT Clause
✅ ALWAYS: Add LIMIT 100 (or appropriate number)

Reason: Prevent accidentally returning millions of rows.

## 8. Data Split
✅ DEFAULT: Use data_split = 'test' for metrics
Only use 'train' or 'validation' if explicitly requested.

Reason: Test metrics are the standard for model evaluation.
        """,
        "metadata": {
            "category": "critical_rules",
            "enforce_always": True
        },
        "keywords": ["rules", "best practices", "requirements", "mandatory"]
    }
]


# Helper function to export for Pinecone ingestion
def get_schema_documents():
    """Get all schema documents for Pinecone embedding"""
    return SCHEMA_DOCUMENTS


def get_documents_by_category(category):
    """Get documents for a specific category"""
    return [doc for doc in SCHEMA_DOCUMENTS if doc.get('category') == category]


def get_table_document(table_name):
    """Get schema document for a specific table"""
    return next((doc for doc in SCHEMA_DOCUMENTS 
                 if doc.get('table_name') == table_name), None)