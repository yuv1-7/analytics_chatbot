"""
Schema documentation for LLM context when generating SQL queries.
Provides table structures, relationships, and business context.
Updated with complete schema coverage including all use cases.
"""

SCHEMA_CONTEXT = """
# DATABASE SCHEMA DOCUMENTATION - PHARMA COMMERCIAL ANALYTICS

## Core Tables

### 1. models
Stores all model definitions (base models and ensembles)
Columns:
- model_id (UUID, PK): Unique identifier
- model_name (VARCHAR): Name of the model
- model_type (VARCHAR): 'base_model' or 'ensemble'
- use_case (VARCHAR): Business use case
  * 'NRx_forecasting' - Predicting new prescriptions
  * 'HCP_engagement' - HCP response to marketing
  * 'feature_importance_analysis' - Understanding key drivers
  * 'model_drift_detection' - Detecting model performance changes
  * 'messaging_optimization' - Next-best-action for HCP targeting
  * 'territory_performance_forecasting' - Territory-level sales forecasting
  * 'market_share_prediction' - Market share forecasting
  * 'hcp_clustering_segmentation' - HCP segmentation via clustering
  * 'price_sensitivity_analysis' - Price elasticity analysis
  * 'competitor_share_forecasting' - Competitive market dynamics
- version (VARCHAR): Model version (e.g., 'v1.0', 'v2.1')
- algorithm (VARCHAR): Algorithm used
  * Base models: 'Random Forest', 'XGBoost', 'LightGBM', 'Logistic Regression', 
    'SVM', 'Neural Network', 'Decision Tree', 'LSTM', 'Prophet', 'ARIMA', 
    'GLM', 'VAR', 'K-Means', 'DBSCAN', 'Causal Forest'
  * Ensembles: 'Stacking', 'Boosting', 'Bagging', 'Voting', 'Blending', 
    'Meta-Learning', 'Kernel Ensemble'
- hyperparameters (JSONB): Model hyperparameters
- description (TEXT): Model description
- created_at (TIMESTAMP): Creation timestamp
- created_by (VARCHAR): Creator
- is_active (BOOLEAN): Active status (filter by is_active = true for production models)
- parent_model_id (UUID, FK → models): Parent model reference
- version_notes (TEXT): Version change notes
- deprecated_at (TIMESTAMP): Deprecation timestamp
- performance_trend (VARCHAR): 'improving', 'degrading', 'stable', 'unknown'

### 2. ensemble_members
Junction table linking ensemble models to base models
Columns:
- ensemble_id (UUID, PK, FK → models): Ensemble model ID
- base_model_id (UUID, PK, FK → models): Base model ID
- weight (FLOAT): Model weight in ensemble (0-1)
- role (VARCHAR): 'base_learner', 'meta_learner', 'stacking_layer'
- added_at (TIMESTAMP): When added to ensemble
- configuration (JSONB): Additional configuration
- ensemble_type (VARCHAR): 'boosting', 'bagging', 'stacking', 'blending', 
  'meta_learning', 'voting', 'kernel_ensemble', 'boosting_bagging_meta', 'uplift_ensemble'

### 3. model_executions
Tracks each execution/run of a model
Columns:
- execution_id (UUID, PK): Unique execution identifier
- model_id (UUID, FK → models): Model that was executed
- execution_timestamp (TIMESTAMP): When executed
- execution_status (VARCHAR): 'success', 'failed', 'running', 'pending'
- training_data_start_date (DATE): Training data start
- training_data_end_date (DATE): Training data end
- prediction_date (DATE): Date predictions were made for
- execution_metadata (JSONB): Additional metadata (data_version, features_count, training_samples)
- error_message (TEXT): Error details if failed
- runtime_seconds (INT): Execution duration
- baseline_execution_id (UUID, FK → model_executions): Baseline for drift comparison
- drift_detected (BOOLEAN): Whether drift was detected
- drift_score (FLOAT): Drift score

### 4. performance_metrics
Model performance metrics per execution
Columns:
- metric_id (UUID, PK): Unique metric identifier
- execution_id (UUID, FK → model_executions): Associated execution
- metric_name (VARCHAR): Metric name
  * Regression: 'rmse', 'mae', 'r2_score', 'mse', 'mape'
  * Classification: 'auc_roc', 'accuracy', 'precision', 'recall', 'f1_score'
  * Clustering: 'silhouette_score', 'davies_bouldin_index', 'calinski_harabasz_score'
  * Uplift: 'auuc', 'qini_coefficient'
  * Custom: 'elasticity_accuracy', 'forecast_accuracy'
- metric_value (FLOAT): Metric value
- data_split (VARCHAR): 'train', 'validation', 'test', 'holdout', 'cross_validation'
- metric_metadata (JSONB): Additional metadata
- calculated_at (TIMESTAMP): Calculation timestamp

### 5. feature_importance
Feature importance scores per execution
Columns:
- importance_id (UUID, PK): Unique identifier
- execution_id (UUID, FK → model_executions): Associated execution
- feature_name (VARCHAR): Feature name (see common features below)
- importance_score (FLOAT): Importance value (0-1 range typically)
- importance_type (VARCHAR): 'gain', 'split', 'shap', 'permutation', 'weight', 'coefficient'
- rank (INT): Feature rank (1 = most important)
- standard_error (FLOAT): Standard error
- calculated_at (TIMESTAMP): Calculation timestamp

Common Feature Names:
- 'historical_nrx_lag1', 'historical_nrx_lag3', 'historical_trx_lag1'
- 'patient_volume', 'hcp_specialty', 'years_in_practice'
- 'call_frequency_lag1', 'sample_volume_lag1', 'email_engagement'
- 'competitive_prescribing', 'brand_loyalty', 'local_market_share'
- 'geographic_region', 'practice_setting', 'prescribing_trend'
- 'share_of_voice', 'event_attendance', 'last_touchpoint_recency'

### 6. predictions
Prediction results for entities (HCP, territory, segment, product)
Columns:
- prediction_id (UUID, PK): Unique identifier
- execution_id (UUID, FK → model_executions): Associated execution
- entity_type (VARCHAR): 'HCP', 'territory', 'segment', 'product', 'account'
- entity_id (VARCHAR): Entity identifier (e.g., 'HCP_000001', 'TERR_0001', 'SEG_001')
- prediction_value (FLOAT): Predicted value
- prediction_probability (FLOAT): Probability (0-1) for classification
- prediction_class (VARCHAR): Predicted class for classification
- actual_value (FLOAT): Actual observed value (if available)
- residual (FLOAT): actual_value - prediction_value (auto-calculated)
- prediction_date (DATE): Prediction date
- confidence_interval_lower (FLOAT): Lower confidence bound
- confidence_interval_upper (FLOAT): Upper confidence bound
- created_at (TIMESTAMP): Creation timestamp
- prediction_type (VARCHAR): 'TRx', 'NRx', 'response_probability', 'uplift', 
  'share_of_voice', 'incremental_lift', 'monthly_sales', 'market_share_pct', 
  'price_elasticity'
- trx_value (FLOAT): Total prescriptions (new + refill)
- nrx_value (FLOAT): New prescriptions only
- control_prediction (FLOAT): Control group prediction (uplift models)
- treatment_prediction (FLOAT): Treatment group prediction (uplift models)
- uplift_value (FLOAT): treatment - control

### 7. drift_detection_results
Model drift detection results
Columns:
- drift_id (UUID, PK): Unique identifier
- execution_id (UUID, FK → model_executions): Current execution
- baseline_execution_id (UUID, FK → model_executions): Baseline execution for comparison
- drift_type (VARCHAR): 'concept_drift', 'data_drift', 'performance_drift', 'prediction_drift'
- drift_metric (VARCHAR): Metric used for detection (e.g., 'PSI', 'KS_statistic', 'RMSE_delta')
- drift_score (FLOAT): Drift score (higher = more drift)
- threshold_value (FLOAT): Threshold used (typically 0.10 = 10%)
- is_significant (BOOLEAN): Whether drift is significant (drift_score > threshold)
- affected_features (JSONB): Features affected by drift
- drift_explanation (TEXT): Explanation of drift
- detected_at (TIMESTAMP): Detection timestamp

### 8. features_metadata
Reference table with feature definitions
Columns:
- feature_id (UUID, PK): Unique identifier
- feature_name (VARCHAR): Feature name
- feature_description (TEXT): Description
- feature_type (VARCHAR): 'categorical', 'numerical', 'text', 'datetime', 'binary', 'ordinal'
- business_meaning (TEXT): Business interpretation
- source_system (VARCHAR): Source system (e.g., 'HCP_Master', 'Claims_Data', 'CRM_System')
- calculation_logic (TEXT): How feature is calculated
- data_quality_notes (TEXT): Quality notes
- created_at (TIMESTAMP): Creation timestamp
- updated_at (TIMESTAMP): Last update timestamp

### 9. model_comparisons
Pre-computed model comparison results
Columns:
- comparison_id (UUID, PK): Unique identifier
- model_1_id (UUID, FK → models): First model
- model_2_id (UUID, FK → models): Second model
- comparison_type (VARCHAR): 'performance', 'feature_importance', 'predictions', 'drift'
- comparison_results (JSONB): Comparison results (model_1_rmse, model_2_rmse, winner, etc.)
- execution_1_id (UUID, FK → model_executions): First execution
- execution_2_id (UUID, FK → model_executions): Second execution
- created_at (TIMESTAMP): Creation timestamp

### 10. version_comparisons
Version-to-version performance comparisons
Columns:
- comparison_id (UUID, PK): Unique identifier
- model_id (UUID, FK → models): Model being compared
- old_version (VARCHAR): Old version string (e.g., 'v1.0')
- new_version (VARCHAR): New version string (e.g., 'v2.0')
- old_execution_id (UUID, FK → model_executions): Old version execution
- new_execution_id (UUID, FK → model_executions): New version execution
- metric_changes (JSONB): Metric changes (rmse_change, r2_change, etc.)
- performance_verdict (VARCHAR): 'improvement', 'degradation', 'neutral', 'mixed'
- key_differences (TEXT): Key differences between versions
- rollback_recommended (BOOLEAN): Whether rollback is recommended
- compared_at (TIMESTAMP): Comparison timestamp

### 11. performance_explanations
Causal explanations for model performance
Columns:
- explanation_id (UUID, PK): Unique identifier
- execution_id (UUID, FK → model_executions): Associated execution
- comparison_execution_id (UUID, FK → model_executions): Comparison execution
- explanation_type (VARCHAR): 'ensemble_advantage', 'performance_degradation', 
  'improvement_cause', 'feature_impact', 'data_quality'
- primary_reason (TEXT): Primary explanation
- contributing_factors (JSONB): Contributing factors
- quantitative_evidence (JSONB): Quantitative evidence (ensemble_rmse, best_base_rmse, etc.)
- recommendation (TEXT): Recommendations
- confidence_score (FLOAT): Confidence (0-1)
- generated_by (VARCHAR): Who/what generated it (default: 'agent_system')
- created_at (TIMESTAMP): Creation timestamp

### 12. feature_interactions
Feature interaction effects
Columns:
- interaction_id (UUID, PK): Unique identifier
- execution_id (UUID, FK → model_executions): Associated execution
- feature_1 (VARCHAR): First feature
- feature_2 (VARCHAR): Second feature
- interaction_strength (FLOAT): Interaction strength
- interaction_type (VARCHAR): 'synergistic', 'antagonistic', 'neutral'
- business_interpretation (TEXT): Business interpretation
- created_at (TIMESTAMP): Creation timestamp

## Pharma-Specific Tables

### 13. hcp_segments
HCP segmentation definitions
Columns:
- segment_id (UUID, PK): Unique identifier
- segment_name (VARCHAR): Segment name (e.g., 'High Value Primary Care', 'Early Adopter Specialists')
- segment_description (TEXT): Description
- criteria (JSONB): Segmentation criteria (specialty, patient_volume thresholds, etc.)
- tier (VARCHAR): 'HIGH', 'MEDIUM', 'LOW', 'TIER_1', 'TIER_2', 'TIER_3'
- hcp_count (INT): Number of HCPs in segment
- created_at (TIMESTAMP): Creation timestamp
- updated_at (TIMESTAMP): Last update timestamp

### 14. campaigns
Marketing campaigns
Columns:
- campaign_id (UUID, PK): Unique identifier
- campaign_name (VARCHAR): Campaign name
- campaign_type (VARCHAR): 'email', 'call', 'event', 'digital', 'sample', 'multi_channel'
- message_content (TEXT): Message content
- target_segment_id (UUID, FK → hcp_segments): Target segment
- start_date (DATE): Campaign start
- end_date (DATE): Campaign end
- budget (DECIMAL): Campaign budget
- campaign_metadata (JSONB): Additional metadata
- created_at (TIMESTAMP): Creation timestamp

### 15. hcp_responses
HCP responses to campaigns
Columns:
- response_id (UUID, PK): Unique identifier
- hcp_id (VARCHAR): HCP identifier
- campaign_id (UUID, FK → campaigns): Campaign
- response_date (DATE): Response date
- responded (BOOLEAN): Whether responded
- response_type (VARCHAR): 'engaged', 'prescribe_increased', 'event_attended', 'no_response', 'opt_out'
- prescription_change_trx (FLOAT): TRx change
- prescription_change_nrx (FLOAT): NRx change
- response_metadata (JSONB): Additional metadata
- created_at (TIMESTAMP): Creation timestamp

### 16. uplift_predictions
Uplift modeling predictions
Columns:
- uplift_id (UUID, PK): Unique identifier
- execution_id (UUID, FK → model_executions): Associated execution
- entity_id (VARCHAR): Entity identifier
- campaign_id (UUID, FK → campaigns): Campaign
- control_prediction (FLOAT): Control prediction
- treatment_prediction (FLOAT): Treatment prediction
- uplift_score (FLOAT): Uplift score (treatment - control)
- confidence_interval_lower (FLOAT): Lower bound
- confidence_interval_upper (FLOAT): Upper bound
- recommended_action (VARCHAR): Recommended action
- action_priority (INT): Priority (1-5, 1=highest)
- predicted_roi (FLOAT): Predicted ROI
- prediction_date (DATE): Prediction date
- created_at (TIMESTAMP): Creation timestamp

### 17. time_series_forecasts
Time series forecasts
Columns:
- forecast_id (UUID, PK): Unique identifier
- execution_id (UUID, FK → model_executions): Associated execution
- entity_id (VARCHAR): Entity identifier
- forecast_date (DATE): Forecast date
- forecast_horizon_days (INT): Forecast horizon (1-365)
- point_forecast (FLOAT): Point forecast
- lower_bound (FLOAT): Lower bound
- upper_bound (FLOAT): Upper bound
- forecast_type (VARCHAR): 'share_of_voice', 'demand', 'sales', 'prescriptions'
- seasonality_component (FLOAT): Seasonality component
- trend_component (FLOAT): Trend component
- residual_component (FLOAT): Residual component
- created_at (TIMESTAMP): Creation timestamp

## Useful Views

### ensemble_composition
Shows ensemble models with their base models
Columns: ensemble_id, ensemble_name, use_case, base_model_id, base_model_name, base_algorithm, weight, role

Query:
```sql
SELECT * FROM ensemble_composition 
WHERE ensemble_name = 'NRx_Ensemble_Stacking_v2';
```

### ensemble_vs_base_performance
Compares ensemble performance to base models (computed averages)
Columns: model_name, use_case, ensemble_rmse, avg_base_rmse, ensemble_r2, avg_base_r2, 
         ensemble_auc, avg_base_auc, ensemble_advantage

Query:
```sql
SELECT * FROM ensemble_vs_base_performance 
WHERE use_case = 'NRx_forecasting';
```

### latest_model_executions
Latest execution per model
Columns: model_id, execution_id, execution_timestamp, execution_status, prediction_date

Query:
```sql
SELECT m.model_name, lme.execution_timestamp, lme.execution_status
FROM latest_model_executions lme
JOIN models m ON lme.model_id = m.model_id
WHERE m.is_active = true;
```

### latest_executions_with_drift
Latest execution per model with drift info
Columns: model_id, execution_id, execution_timestamp, execution_status, prediction_date, 
         drift_detected, drift_score, baseline_execution_id

Query:
```sql
SELECT m.model_name, led.drift_detected, led.drift_score
FROM latest_executions_with_drift led
JOIN models m ON led.model_id = m.model_id
WHERE led.drift_detected = true;
```

### model_performance_summary
Model performance summary with metrics
Columns: model_id, model_name, model_type, use_case, version, performance_trend, 
         execution_id, execution_timestamp, drift_detected, metric_name, metric_value, data_split

Query:
```sql
SELECT model_name, metric_name, metric_value
FROM model_performance_summary
WHERE use_case = 'NRx_forecasting' 
  AND data_split = 'test'
  AND metric_name IN ('rmse', 'r2_score');
```

### top_features
Top 10 features per execution
Columns: execution_id, feature_name, importance_score, importance_type, rank

Query:
```sql
SELECT tf.feature_name, tf.importance_score, tf.rank
FROM top_features tf
WHERE tf.execution_id = 'some-execution-id'
ORDER BY tf.rank;
```

## Common Query Patterns

### 1. Get latest metrics for a model:
```sql
SELECT pm.metric_name, pm.metric_value 
FROM models m
JOIN latest_model_executions lme ON m.model_id = lme.model_id
JOIN performance_metrics pm ON lme.execution_id = pm.execution_id
WHERE m.model_name ILIKE '%XGBoost%' 
  AND m.use_case = 'NRx_forecasting'
  AND pm.data_split = 'test';
```

### 2. Compare multiple models:
```sql
SELECT 
    m.model_name,
    m.model_type,
    m.algorithm,
    pm.metric_name,
    pm.metric_value
FROM models m
JOIN latest_model_executions lme ON m.model_id = lme.model_id
JOIN performance_metrics pm ON lme.execution_id = pm.execution_id
WHERE m.use_case = 'NRx_forecasting'
  AND m.is_active = true
  AND pm.data_split = 'test'
  AND pm.metric_name IN ('rmse', 'r2_score')
ORDER BY pm.metric_name, pm.metric_value;
```

### 3. Get feature importance for a model:
```sql
SELECT fi.feature_name, fi.importance_score, fi.rank
FROM models m
JOIN latest_model_executions lme ON m.model_id = lme.model_id
JOIN feature_importance fi ON lme.execution_id = fi.execution_id
WHERE m.model_name ILIKE '%Random Forest%'
  AND m.use_case = 'NRx_forecasting'
ORDER BY fi.rank
LIMIT 10;
```

### 4. Check for drift:
```sql
SELECT 
    m.model_name,
    ddr.drift_type,
    ddr.drift_score,
    ddr.is_significant,
    ddr.drift_explanation
FROM models m
JOIN latest_executions_with_drift led ON m.model_id = led.model_id
JOIN drift_detection_results ddr ON led.execution_id = ddr.execution_id
WHERE m.use_case = 'NRx_forecasting'
  AND ddr.is_significant = true;
```

### 5. Compare model versions:
```sql
SELECT 
    m.model_name,
    vc.old_version,
    vc.new_version,
    vc.performance_verdict,
    vc.metric_changes->>'rmse_change' AS rmse_change,
    vc.metric_changes->>'r2_change' AS r2_change,
    vc.rollback_recommended
FROM version_comparisons vc
JOIN models m ON vc.model_id = m.model_id
WHERE m.model_name ILIKE '%XGBoost%'
ORDER BY vc.compared_at DESC;
```

### 6. Get ensemble composition:
```sql
SELECT 
    e.model_name AS ensemble_name,
    b.model_name AS base_model_name,
    b.algorithm,
    em.weight,
    em.role
FROM models e
JOIN ensemble_members em ON e.model_id = em.ensemble_id
JOIN models b ON em.base_model_id = b.model_id
WHERE e.model_name ILIKE '%Ensemble%'
  AND e.is_active = true
ORDER BY em.weight DESC;
```

### 7. Analyze predictions by entity:
```sql
SELECT 
    p.entity_id,
    COUNT(*) AS prediction_count,
    AVG(p.prediction_value) AS avg_predicted,
    AVG(p.actual_value) AS avg_actual,
    AVG(ABS(p.residual)) AS avg_absolute_error
FROM predictions p
JOIN model_executions me ON p.execution_id = me.execution_id
JOIN models m ON me.model_id = m.model_id
WHERE m.use_case = 'NRx_forecasting'
  AND p.entity_type = 'HCP'
  AND p.prediction_date >= '2024-01-01'
GROUP BY p.entity_id
ORDER BY avg_absolute_error DESC
LIMIT 20;
```

### 8. Performance trend over time:
```sql
SELECT 
    DATE_TRUNC('month', me.execution_timestamp) AS month,
    m.model_name,
    AVG(pm.metric_value) AS avg_rmse
FROM models m
JOIN model_executions me ON m.model_id = me.model_id
JOIN performance_metrics pm ON me.execution_id = pm.execution_id
WHERE m.use_case = 'NRx_forecasting'
  AND pm.metric_name = 'rmse'
  AND pm.data_split = 'test'
  AND me.execution_timestamp >= NOW() - INTERVAL '6 months'
GROUP BY DATE_TRUNC('month', me.execution_timestamp), m.model_name
ORDER BY month DESC, avg_rmse;
```

### 9. Territory performance analysis:
```sql
SELECT 
    p.entity_id AS territory,
    COUNT(*) AS num_predictions,
    AVG(p.prediction_value) AS avg_predicted_sales,
    AVG(p.actual_value) AS avg_actual_sales,
    AVG(p.residual) AS avg_forecast_error
FROM predictions p
JOIN model_executions me ON p.execution_id = me.execution_id
JOIN models m ON me.model_id = m.model_id
WHERE m.use_case = 'territory_performance_forecasting'
  AND p.entity_type = 'territory'
  AND p.prediction_date >= '2024-01-01'
GROUP BY p.entity_id
ORDER BY avg_actual_sales DESC
LIMIT 10;
```

### 10. Uplift model analysis:
```sql
SELECT 
    up.entity_id,
    up.control_prediction,
    up.treatment_prediction,
    up.uplift_score,
    up.recommended_action,
    up.predicted_roi
FROM uplift_predictions up
JOIN model_executions me ON up.execution_id = me.execution_id
JOIN models m ON me.model_id = m.model_id
WHERE m.use_case = 'messaging_optimization'
  AND up.uplift_score > 0
ORDER BY up.uplift_score DESC
LIMIT 20;
```

## Important Notes

1. **Always use test data split** for performance metrics unless specifically asked for train/validation
2. **Use latest_model_executions view** to get most recent execution
3. **Join through model_executions** to connect models to metrics/predictions
4. **Check is_active = true** when querying models (unless looking for deprecated models)
5. **Use ILIKE** for case-insensitive string matching (e.g., `model_name ILIKE '%xgboost%'`)
6. **Aggregate base model metrics** when comparing to ensembles
7. **Filter by use_case** when appropriate to narrow results
8. **Use views** when available for common patterns
9. **Handle NULL values** appropriately (use NULLIF, COALESCE)
10. **Use JSONB operators** for extracting data from JSON columns (->>, ->, #>)
11. **Date filtering**: Use proper date comparisons and DATE_TRUNC for grouping
12. **Avoid self-joins** on drift tables (execution_id != baseline_execution_id)

## Metric Interpretation

### Regression Metrics (lower is better):
- rmse: Root Mean Squared Error
- mae: Mean Absolute Error  
- mse: Mean Squared Error
- mape: Mean Absolute Percentage Error

### Regression Metrics (higher is better):
- r2_score: R-squared (0-1, higher = better fit)

### Classification Metrics (higher is better):
- accuracy: Overall accuracy (0-1)
- precision: Precision score (0-1)
- recall: Recall score (0-1)
- f1_score: F1 score (0-1)
- auc_roc: Area Under ROC Curve (0-1)

### Clustering Metrics:
- silhouette_score: Higher is better (-1 to 1)
- davies_bouldin_index: Lower is better (0 to infinity)
- calinski_harabasz_score: Higher is better

### Uplift Metrics:
- auuc: Area Under Uplift Curve (higher is better)
- qini_coefficient: Higher is better

### When comparing ensemble vs base:
- For RMSE/MAE/MSE: ensemble < base = ensemble is better
- For R2/AUC/Accuracy: ensemble > base = ensemble is better

## Use Case Mapping

- **NRx_forecasting**: Predicting new prescription volumes
- **HCP_engagement**: Predicting HCP response to campaigns
- **territory_performance_forecasting**: Territory sales forecasting
- **market_share_prediction**: Market share forecasting
- **hcp_clustering_segmentation**: HCP segmentation
- **price_sensitivity_analysis**: Price elasticity analysis
- **competitor_share_forecasting**: Competitive dynamics
- **feature_importance_analysis**: Understanding drivers
- **model_drift_detection**: Monitoring model performance
- **messaging_optimization**: Next-best-action/uplift modeling
"""

# Metric interpretation guide
METRIC_GUIDE = """
## Metric Interpretation

### Regression Metrics (lower is better):
- rmse: Root Mean Squared Error
- mae: Mean Absolute Error  
- mse: Mean Squared Error
- mape: Mean Absolute Percentage Error

### Regression Metrics (higher is better):
- r2_score: R-squared (0-1, higher = better fit)

### Classification Metrics (higher is better):
- accuracy: Overall accuracy (0-1)
- precision: Precision score (0-1)
- recall: Recall score (0-1)
- f1_score: F1 score (0-1)
- auc_roc: Area Under ROC Curve (0-1)

### When comparing ensemble vs base:
- For RMSE/MAE/MSE: ensemble < base = ensemble is better
- For R2/AUC/Accuracy: ensemble > base = ensemble is better
"""

