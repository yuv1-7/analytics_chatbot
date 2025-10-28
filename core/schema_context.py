"""
Schema documentation for LLM context when generating SQL queries.
Provides table structures, relationships, and business context.
"""

SCHEMA_CONTEXT = """
# DATABASE SCHEMA DOCUMENTATION

## Core Tables

### 1. models
Stores all model definitions (base models and ensembles)
Columns:
- model_id (UUID, PK): Unique identifier
- model_name (VARCHAR): Name of the model
- model_type (VARCHAR): 'base_model' or 'ensemble'
- use_case (VARCHAR): Business use case (e.g., 'NRx_forecasting', 'HCP_engagement', 'feature_importance_analysis', 'model_drift_detection', 'messaging_optimization')
- version (VARCHAR): Model version
- algorithm (VARCHAR): Algorithm used (e.g., 'Random Forest', 'XGBoost', 'LightGBM', 'Logistic Regression', 'SVM', 'Neural Network')
- hyperparameters (JSONB): Model hyperparameters
- description (TEXT): Model description
- created_at (TIMESTAMP): Creation timestamp
- created_by (VARCHAR): Creator
- is_active (BOOLEAN): Active status
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
- ensemble_type (VARCHAR): 'boosting', 'bagging', 'stacking', 'blending', 'meta_learning', 'voting', 'kernel_ensemble', 'boosting_bagging_meta', 'uplift_ensemble'

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
- execution_metadata (JSONB): Additional metadata
- error_message (TEXT): Error details if failed
- runtime_seconds (INT): Execution duration
- baseline_execution_id (UUID, FK → model_executions): Baseline for comparison
- drift_detected (BOOLEAN): Whether drift was detected
- drift_score (FLOAT): Drift score

### 4. performance_metrics
Model performance metrics per execution
Columns:
- metric_id (UUID, PK): Unique metric identifier
- execution_id (UUID, FK → model_executions): Associated execution
- metric_name (VARCHAR): Metric name (e.g., 'rmse', 'mae', 'r2_score', 'auc_roc', 'accuracy', 'precision', 'recall', 'f1_score')
- metric_value (FLOAT): Metric value
- data_split (VARCHAR): 'train', 'validation', 'test', 'holdout', 'cross_validation'
- metric_metadata (JSONB): Additional metadata
- calculated_at (TIMESTAMP): Calculation timestamp

### 5. feature_importance
Feature importance scores per execution
Columns:
- importance_id (UUID, PK): Unique identifier
- execution_id (UUID, FK → model_executions): Associated execution
- feature_name (VARCHAR): Feature name
- importance_score (FLOAT): Importance value
- importance_type (VARCHAR): 'gain', 'split', 'shap', 'permutation', 'weight', 'coefficient'
- rank (INT): Feature rank (1 = most important)
- standard_error (FLOAT): Standard error
- calculated_at (TIMESTAMP): Calculation timestamp

### 6. predictions
Prediction results for entities (HCP, territory, etc.)
Columns:
- prediction_id (UUID, PK): Unique identifier
- execution_id (UUID, FK → model_executions): Associated execution
- entity_type (VARCHAR): 'HCP', 'territory', 'segment', 'product', 'account'
- entity_id (VARCHAR): Entity identifier
- prediction_value (FLOAT): Predicted value
- prediction_probability (FLOAT): Probability (0-1) for classification
- prediction_class (VARCHAR): Predicted class for classification
- actual_value (FLOAT): Actual observed value (if available)
- residual (FLOAT): actual_value - prediction_value (auto-calculated)
- prediction_date (DATE): Prediction date
- confidence_interval_lower (FLOAT): Lower confidence bound
- confidence_interval_upper (FLOAT): Upper confidence bound
- created_at (TIMESTAMP): Creation timestamp
- prediction_type (VARCHAR): 'TRx', 'NRx', 'response_probability', 'uplift', 'share_of_voice', 'incremental_lift'
- trx_value (FLOAT): Total prescriptions
- nrx_value (FLOAT): New prescriptions
- control_prediction (FLOAT): Control group prediction (uplift models)
- treatment_prediction (FLOAT): Treatment group prediction (uplift models)
- uplift_value (FLOAT): treatment - control

### 7. drift_detection_results
Model drift detection results
Columns:
- drift_id (UUID, PK): Unique identifier
- execution_id (UUID, FK → model_executions): Current execution
- baseline_execution_id (UUID, FK → model_executions): Baseline execution
- drift_type (VARCHAR): 'concept_drift', 'data_drift', 'performance_drift', 'prediction_drift'
- drift_metric (VARCHAR): Metric used for detection
- drift_score (FLOAT): Drift score
- threshold_value (FLOAT): Threshold used
- is_significant (BOOLEAN): Whether drift is significant
- affected_features (JSONB): Features affected by drift
- drift_explanation (TEXT): Explanation
- detected_at (TIMESTAMP): Detection timestamp

### 8. features_metadata
Reference table with feature definitions
Columns:
- feature_id (UUID, PK): Unique identifier
- feature_name (VARCHAR): Feature name
- feature_description (TEXT): Description
- feature_type (VARCHAR): 'categorical', 'numerical', 'text', 'datetime', 'binary', 'ordinal'
- business_meaning (TEXT): Business interpretation
- source_system (VARCHAR): Source system
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
- comparison_results (JSONB): Comparison results
- execution_1_id (UUID, FK → model_executions): First execution
- execution_2_id (UUID, FK → model_executions): Second execution
- created_at (TIMESTAMP): Creation timestamp

### 10. version_comparisons
Version-to-version performance comparisons
Columns:
- comparison_id (UUID, PK): Unique identifier
- model_id (UUID, FK → models): Model being compared
- old_version (VARCHAR): Old version string
- new_version (VARCHAR): New version string
- old_execution_id (UUID, FK → model_executions): Old version execution
- new_execution_id (UUID, FK → model_executions): New version execution
- metric_changes (JSONB): Metric changes
- performance_verdict (VARCHAR): 'improvement', 'degradation', 'neutral', 'mixed'
- key_differences (TEXT): Key differences
- rollback_recommended (BOOLEAN): Whether rollback is recommended
- compared_at (TIMESTAMP): Comparison timestamp

### 11. performance_explanations
Causal explanations for model performance
Columns:
- explanation_id (UUID, PK): Unique identifier
- execution_id (UUID, FK → model_executions): Associated execution
- comparison_execution_id (UUID, FK → model_executions): Comparison execution
- explanation_type (VARCHAR): 'ensemble_advantage', 'performance_degradation', 'improvement_cause', 'feature_impact', 'data_quality'
- primary_reason (TEXT): Primary explanation
- contributing_factors (JSONB): Contributing factors
- quantitative_evidence (JSONB): Quantitative evidence
- recommendation (TEXT): Recommendations
- confidence_score (FLOAT): Confidence (0-1)
- generated_by (VARCHAR): Who/what generated it
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
- segment_name (VARCHAR): Segment name
- segment_description (TEXT): Description
- criteria (JSONB): Segmentation criteria
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
- uplift_score (FLOAT): Uplift score
- confidence_interval_lower (FLOAT): Lower bound
- confidence_interval_upper (FLOAT): Upper bound
- recommended_action (VARCHAR): Recommended action
- action_priority (INT): Priority (1-5)
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

### ensemble_vs_base_performance
Compares ensemble performance to base models
Columns: model_name, use_case, ensemble_rmse, avg_base_rmse, ensemble_r2, avg_base_r2, ensemble_auc, avg_base_auc, ensemble_advantage

### latest_model_executions
Latest execution per model
Columns: model_id, execution_id, execution_timestamp, execution_status, prediction_date

### latest_executions_with_drift
Latest execution per model with drift info
Columns: model_id, execution_id, execution_timestamp, execution_status, prediction_date, drift_detected, drift_score, baseline_execution_id

### model_performance_summary
Model performance summary with metrics
Columns: model_id, model_name, model_type, use_case, version, performance_trend, execution_id, execution_timestamp, drift_detected, metric_name, metric_value, data_split

### top_features
Top 10 features per execution
Columns: execution_id, feature_name, importance_score, importance_type, rank

## Common Query Patterns

### Get ensemble vs base model performance:
```sql
SELECT * FROM ensemble_vs_base_performance 
WHERE model_name = 'ensemble_name';
```

### Get latest metrics for a model:
```sql
SELECT pm.metric_name, pm.metric_value 
FROM models m
JOIN latest_model_executions lme ON m.model_id = lme.model_id
JOIN performance_metrics pm ON lme.execution_id = pm.execution_id
WHERE m.model_name = 'model_name' AND pm.data_split = 'test';
```

### Get feature importance:
```sql
SELECT feature_name, importance_score, rank
FROM feature_importance fi
JOIN latest_model_executions lme ON fi.execution_id = lme.execution_id
WHERE lme.model_id = (SELECT model_id FROM models WHERE model_name = 'model_name')
ORDER BY rank LIMIT 20;
```

### Check for drift:
```sql
SELECT drift_detected, drift_score 
FROM latest_executions_with_drift
WHERE model_id = (SELECT model_id FROM models WHERE model_name = 'model_name');
```

### Compare model versions:
```sql
SELECT vc.* FROM version_comparisons vc
JOIN models m ON vc.model_id = m.model_id
WHERE m.model_name = 'model_name' 
  AND vc.old_version = 'v1.0' 
  AND vc.new_version = 'v2.0';
```

## Important Notes

1. **Always use test data split** for performance metrics unless specifically asked for train/validation
2. **Use latest_model_executions view** to get most recent execution
3. **Join through model_executions** to connect models to metrics/predictions
4. **Check is_active = true** when querying models
5. **Use ILIKE** for case-insensitive string matching
6. **Aggregate base model metrics** when comparing to ensembles
7. **Filter by use_case** when appropriate
8. **Use views** when available for common patterns
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