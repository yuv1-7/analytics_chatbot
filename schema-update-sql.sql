-- ============================================================================
-- PHARMA COMMERCIAL ANALYTICS - SCHEMA UPDATES (MIGRATION)
-- Updates existing schema to support all use case requirements
-- ============================================================================

BEGIN;

-- ============================================================================
-- 1. ADD NEW COLUMNS TO EXISTING TABLES
-- ============================================================================

-- Update models table to track version history better
ALTER TABLE models 
ADD COLUMN IF NOT EXISTS parent_model_id UUID REFERENCES models(model_id),
ADD COLUMN IF NOT EXISTS version_notes TEXT,
ADD COLUMN IF NOT EXISTS deprecated_at TIMESTAMP,
ADD COLUMN IF NOT EXISTS performance_trend VARCHAR(20) CHECK (performance_trend IN ('improving', 'degrading', 'stable', 'unknown'));

-- Update predictions table for TRx/NRx specificity
ALTER TABLE predictions
ADD COLUMN IF NOT EXISTS prediction_type VARCHAR(50) CHECK (prediction_type IN ('TRx', 'NRx', 'response_probability', 'uplift', 'share_of_voice', 'incremental_lift')),
ADD COLUMN IF NOT EXISTS trx_value FLOAT,
ADD COLUMN IF NOT EXISTS nrx_value FLOAT,
ADD COLUMN IF NOT EXISTS control_prediction FLOAT,
ADD COLUMN IF NOT EXISTS treatment_prediction FLOAT,
ADD COLUMN IF NOT EXISTS uplift_value FLOAT;

COMMENT ON COLUMN predictions.trx_value IS 'Total prescriptions (new + refill)';
COMMENT ON COLUMN predictions.nrx_value IS 'New prescriptions only';
COMMENT ON COLUMN predictions.control_prediction IS 'Predicted value without treatment (for uplift models)';
COMMENT ON COLUMN predictions.treatment_prediction IS 'Predicted value with treatment (for uplift models)';
COMMENT ON COLUMN predictions.uplift_value IS 'Incremental lift = treatment - control';

-- Update ensemble_members to distinguish ensemble types
ALTER TABLE ensemble_members
ADD COLUMN IF NOT EXISTS ensemble_type VARCHAR(50) CHECK (ensemble_type IN ('boosting', 'bagging', 'stacking', 'blending', 'meta_learning', 'voting', 'kernel_ensemble', 'boosting_bagging_meta', 'uplift_ensemble'));

-- Update model_executions for drift detection
ALTER TABLE model_executions
ADD COLUMN IF NOT EXISTS baseline_execution_id UUID REFERENCES model_executions(execution_id),
ADD COLUMN IF NOT EXISTS drift_detected BOOLEAN DEFAULT FALSE,
ADD COLUMN IF NOT EXISTS drift_score FLOAT;

-- ============================================================================
-- 2. CREATE NEW TABLES FOR MISSING USE CASES
-- ============================================================================

-- HCP Segments table
CREATE TABLE IF NOT EXISTS hcp_segments (
    segment_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    segment_name VARCHAR(100) NOT NULL UNIQUE,
    segment_description TEXT,
    criteria JSONB,
    tier VARCHAR(20) CHECK (tier IN ('HIGH', 'MEDIUM', 'LOW', 'TIER_1', 'TIER_2', 'TIER_3')),
    hcp_count INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

COMMENT ON TABLE hcp_segments IS 'HCP segmentation definitions for targeting';

CREATE INDEX idx_hcp_segments_tier ON hcp_segments(tier);

-- ============================================================================
-- Marketing Campaigns table (for messaging optimization use case)
CREATE TABLE IF NOT EXISTS campaigns (
    campaign_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    campaign_name VARCHAR(255) NOT NULL,
    campaign_type VARCHAR(100) CHECK (campaign_type IN ('email', 'call', 'event', 'digital', 'sample', 'multi_channel')),
    message_content TEXT,
    target_segment_id UUID REFERENCES hcp_segments(segment_id),
    start_date DATE NOT NULL,
    end_date DATE,
    budget DECIMAL(12, 2),
    campaign_metadata JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

COMMENT ON TABLE campaigns IS 'Marketing campaigns and messaging initiatives';

CREATE INDEX idx_campaigns_dates ON campaigns(start_date, end_date);
CREATE INDEX idx_campaigns_segment ON campaigns(target_segment_id);

-- ============================================================================
-- HCP Response Data table (actual responses to campaigns)
CREATE TABLE IF NOT EXISTS hcp_responses (
    response_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    hcp_id VARCHAR(255) NOT NULL,
    campaign_id UUID REFERENCES campaigns(campaign_id) ON DELETE CASCADE,
    response_date DATE NOT NULL,
    responded BOOLEAN NOT NULL,
    response_type VARCHAR(50) CHECK (response_type IN ('engaged', 'prescribe_increased', 'event_attended', 'no_response', 'opt_out')),
    prescription_change_trx FLOAT,
    prescription_change_nrx FLOAT,
    response_metadata JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

COMMENT ON TABLE hcp_responses IS 'Actual HCP responses to marketing campaigns';

CREATE INDEX idx_hcp_responses_hcp ON hcp_responses(hcp_id);
CREATE INDEX idx_hcp_responses_campaign ON hcp_responses(campaign_id);
CREATE INDEX idx_hcp_responses_date ON hcp_responses(response_date);

-- ============================================================================
-- Model Drift Detection Results
CREATE TABLE IF NOT EXISTS drift_detection_results (
    drift_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    execution_id UUID NOT NULL REFERENCES model_executions(execution_id) ON DELETE CASCADE,
    baseline_execution_id UUID NOT NULL REFERENCES model_executions(execution_id) ON DELETE CASCADE,
    drift_type VARCHAR(50) CHECK (drift_type IN ('concept_drift', 'data_drift', 'performance_drift', 'prediction_drift')),
    drift_metric VARCHAR(100) NOT NULL,
    drift_score FLOAT NOT NULL,
    threshold_value FLOAT,
    is_significant BOOLEAN DEFAULT FALSE,
    affected_features JSONB,
    drift_explanation TEXT,
    detected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT no_self_baseline CHECK (execution_id != baseline_execution_id)
);

COMMENT ON TABLE drift_detection_results IS 'Model drift detection results comparing executions over time';

CREATE INDEX idx_drift_execution ON drift_detection_results(execution_id);
CREATE INDEX idx_drift_baseline ON drift_detection_results(baseline_execution_id);
CREATE INDEX idx_drift_significant ON drift_detection_results(is_significant);

-- ============================================================================
-- Model Performance Explanations (Why ensemble performs better/worse)
CREATE TABLE IF NOT EXISTS performance_explanations (
    explanation_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    execution_id UUID NOT NULL REFERENCES model_executions(execution_id) ON DELETE CASCADE,
    comparison_execution_id UUID REFERENCES model_executions(execution_id) ON DELETE CASCADE,
    explanation_type VARCHAR(50) CHECK (explanation_type IN ('ensemble_advantage', 'performance_degradation', 'improvement_cause', 'feature_impact', 'data_quality')),
    primary_reason TEXT NOT NULL,
    contributing_factors JSONB,
    quantitative_evidence JSONB,
    recommendation TEXT,
    confidence_score FLOAT CHECK (confidence_score >= 0 AND confidence_score <= 1),
    generated_by VARCHAR(100) DEFAULT 'agent_system',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

COMMENT ON TABLE performance_explanations IS 'Stores causal explanations for model performance differences';

CREATE INDEX idx_explanations_execution ON performance_explanations(execution_id);
CREATE INDEX idx_explanations_type ON performance_explanations(explanation_type);

-- ============================================================================
-- Uplift Model Results (for Next-Best-Action use case)
CREATE TABLE IF NOT EXISTS uplift_predictions (
    uplift_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    execution_id UUID NOT NULL REFERENCES model_executions(execution_id) ON DELETE CASCADE,
    entity_id VARCHAR(255) NOT NULL,
    campaign_id UUID REFERENCES campaigns(campaign_id),
    control_prediction FLOAT NOT NULL,
    treatment_prediction FLOAT NOT NULL,
    uplift_score FLOAT NOT NULL,
    confidence_interval_lower FLOAT,
    confidence_interval_upper FLOAT,
    recommended_action VARCHAR(255),
    action_priority INTEGER CHECK (action_priority BETWEEN 1 AND 5),
    predicted_roi FLOAT,
    prediction_date DATE NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

COMMENT ON TABLE uplift_predictions IS 'Uplift modeling predictions for incremental lift estimation';

CREATE INDEX idx_uplift_execution ON uplift_predictions(execution_id);
CREATE INDEX idx_uplift_entity ON uplift_predictions(entity_id);
CREATE INDEX idx_uplift_campaign ON uplift_predictions(campaign_id);

-- ============================================================================
-- Time Series Forecasts (for drift detection use case)
CREATE TABLE IF NOT EXISTS time_series_forecasts (
    forecast_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    execution_id UUID NOT NULL REFERENCES model_executions(execution_id) ON DELETE CASCADE,
    entity_id VARCHAR(255) NOT NULL,
    forecast_date DATE NOT NULL,
    forecast_horizon_days INTEGER NOT NULL,
    point_forecast FLOAT NOT NULL,
    lower_bound FLOAT,
    upper_bound FLOAT,
    forecast_type VARCHAR(50) CHECK (forecast_type IN ('share_of_voice', 'demand', 'sales', 'prescriptions')),
    seasonality_component FLOAT,
    trend_component FLOAT,
    residual_component FLOAT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

COMMENT ON TABLE time_series_forecasts IS 'Time series forecasts for demand and share of voice predictions';

CREATE INDEX idx_ts_forecast_execution ON time_series_forecasts(execution_id);
CREATE INDEX idx_ts_forecast_entity ON time_series_forecasts(entity_id);
CREATE INDEX idx_ts_forecast_date ON time_series_forecasts(forecast_date);

-- ============================================================================
-- Feature Interactions (for understanding ensemble advantages)
CREATE TABLE IF NOT EXISTS feature_interactions (
    interaction_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    execution_id UUID NOT NULL REFERENCES model_executions(execution_id) ON DELETE CASCADE,
    feature_1 VARCHAR(255) NOT NULL,
    feature_2 VARCHAR(255) NOT NULL,
    interaction_strength FLOAT NOT NULL,
    interaction_type VARCHAR(50) CHECK (interaction_type IN ('synergistic', 'antagonistic', 'neutral')),
    business_interpretation TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT no_self_interaction CHECK (feature_1 != feature_2)
);

COMMENT ON TABLE feature_interactions IS 'Feature interaction effects discovered by ensemble models';

CREATE INDEX idx_interactions_execution ON feature_interactions(execution_id);
CREATE INDEX idx_interactions_features ON feature_interactions(feature_1, feature_2);

-- ============================================================================
-- Model Version Comparisons (automated tracking)
CREATE TABLE IF NOT EXISTS version_comparisons (
    comparison_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    model_id UUID NOT NULL REFERENCES models(model_id) ON DELETE CASCADE,
    old_version VARCHAR(50) NOT NULL,
    new_version VARCHAR(50) NOT NULL,
    old_execution_id UUID NOT NULL REFERENCES model_executions(execution_id),
    new_execution_id UUID NOT NULL REFERENCES model_executions(execution_id),
    metric_changes JSONB NOT NULL,
    performance_verdict VARCHAR(50) CHECK (performance_verdict IN ('improvement', 'degradation', 'neutral', 'mixed')),
    key_differences TEXT,
    rollback_recommended BOOLEAN DEFAULT FALSE,
    compared_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

COMMENT ON TABLE version_comparisons IS 'Automated version-to-version performance comparisons';

CREATE INDEX idx_version_comp_model ON version_comparisons(model_id);
CREATE INDEX idx_version_comp_verdict ON version_comparisons(performance_verdict);

-- ============================================================================
-- 3. UPDATE EXISTING INDEXES
-- ============================================================================

-- Add composite indexes for common queries
CREATE INDEX IF NOT EXISTS idx_predictions_type_date ON predictions(prediction_type, prediction_date);
CREATE INDEX IF NOT EXISTS idx_predictions_entity_type_date ON predictions(entity_type, entity_id, prediction_date);
CREATE INDEX IF NOT EXISTS idx_executions_model_status ON model_executions(model_id, execution_status, execution_timestamp);

-- ============================================================================
-- 4. UPDATE VIEWS
-- ============================================================================

-- Drop and recreate views with new fields
DROP VIEW IF EXISTS model_performance_summary CASCADE;

CREATE OR REPLACE VIEW model_performance_summary AS
SELECT 
    m.model_id,
    m.model_name,
    m.model_type,
    m.use_case,
    m.version,
    m.performance_trend,
    me.execution_id,
    me.execution_timestamp,
    me.drift_detected,
    pm.metric_name,
    pm.metric_value,
    pm.data_split
FROM models m
JOIN model_executions me ON m.model_id = me.model_id
JOIN performance_metrics pm ON me.execution_id = pm.execution_id
WHERE me.execution_status = 'success'
  AND m.is_active = TRUE;

-- New view: Latest execution with drift info
CREATE OR REPLACE VIEW latest_executions_with_drift AS
SELECT DISTINCT ON (model_id)
    model_id,
    execution_id,
    execution_timestamp,
    execution_status,
    prediction_date,
    drift_detected,
    drift_score,
    baseline_execution_id
FROM model_executions
ORDER BY model_id, execution_timestamp DESC;

-- New view: Ensemble performance comparison
CREATE OR REPLACE VIEW ensemble_vs_base_performance AS
WITH ensemble_metrics AS (
    SELECT 
        me.model_id,
        me.execution_id,
        AVG(CASE WHEN pm.metric_name = 'rmse' THEN pm.metric_value END) as ensemble_rmse,
        AVG(CASE WHEN pm.metric_name = 'r2_score' THEN pm.metric_value END) as ensemble_r2,
        AVG(CASE WHEN pm.metric_name = 'auc_roc' THEN pm.metric_value END) as ensemble_auc
    FROM model_executions me
    JOIN performance_metrics pm ON me.execution_id = pm.execution_id
    JOIN models m ON me.model_id = m.model_id
    WHERE m.model_type = 'ensemble' AND pm.data_split = 'test'
    GROUP BY me.model_id, me.execution_id
),
base_metrics AS (
    SELECT 
        em.ensemble_id,
        AVG(CASE WHEN pm.metric_name = 'rmse' THEN pm.metric_value END) as avg_base_rmse,
        AVG(CASE WHEN pm.metric_name = 'r2_score' THEN pm.metric_value END) as avg_base_r2,
        AVG(CASE WHEN pm.metric_name = 'auc_roc' THEN pm.metric_value END) as avg_base_auc
    FROM ensemble_members em
    JOIN model_executions me ON em.base_model_id = me.model_id
    JOIN performance_metrics pm ON me.execution_id = pm.execution_id
    WHERE pm.data_split = 'test'
    GROUP BY em.ensemble_id
)
SELECT 
    m.model_name,
    m.use_case,
    em.ensemble_rmse,
    bm.avg_base_rmse,
    em.ensemble_r2,
    bm.avg_base_r2,
    em.ensemble_auc,
    bm.avg_base_auc,
    CASE 
        WHEN em.ensemble_rmse < bm.avg_base_rmse THEN 'Better'
        WHEN em.ensemble_rmse > bm.avg_base_rmse THEN 'Worse'
        ELSE 'Similar'
    END as ensemble_advantage
FROM ensemble_metrics em
JOIN models m ON em.model_id = m.model_id
LEFT JOIN base_metrics bm ON m.model_id = bm.ensemble_id;

-- ============================================================================
-- 5. ADD TRIGGERS FOR AUTOMATION
-- ============================================================================

-- Auto-detect drift when new execution completes
CREATE OR REPLACE FUNCTION check_drift_on_execution()
RETURNS TRIGGER AS $$
DECLARE
    prev_execution_id UUID;
    prev_metrics JSONB;
    curr_metrics JSONB;
    drift_threshold FLOAT := 0.10; -- 10% performance change threshold
BEGIN
    IF NEW.execution_status = 'success' THEN
        -- Find previous successful execution
        SELECT execution_id INTO prev_execution_id
        FROM model_executions
        WHERE model_id = NEW.model_id
          AND execution_status = 'success'
          AND execution_timestamp < NEW.execution_timestamp
        ORDER BY execution_timestamp DESC
        LIMIT 1;
        
        IF prev_execution_id IS NOT NULL THEN
            -- Simple drift detection: compare key metrics
            -- (In production, this would be more sophisticated)
            UPDATE model_executions
            SET baseline_execution_id = prev_execution_id,
                drift_detected = TRUE
            WHERE execution_id = NEW.execution_id
              AND EXISTS (
                  SELECT 1 FROM performance_metrics pm1
                  JOIN performance_metrics pm2 ON pm1.metric_name = pm2.metric_name
                  WHERE pm1.execution_id = NEW.execution_id
                    AND pm2.execution_id = prev_execution_id
                    AND ABS(pm1.metric_value - pm2.metric_value) / NULLIF(pm2.metric_value, 0) > drift_threshold
              );
        END IF;
    END IF;
    
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trg_check_drift
AFTER INSERT OR UPDATE ON model_executions
FOR EACH ROW
EXECUTE FUNCTION check_drift_on_execution();

-- ============================================================================
-- 6. ADD CONSTRAINTS
-- ============================================================================

-- Ensure uplift predictions have valid control/treatment values
ALTER TABLE uplift_predictions
ADD CONSTRAINT valid_uplift CHECK (treatment_prediction >= control_prediction OR treatment_prediction IS NOT NULL);

-- Ensure time series forecasts have valid horizon
ALTER TABLE time_series_forecasts
ADD CONSTRAINT valid_forecast_horizon CHECK (forecast_horizon_days > 0 AND forecast_horizon_days <= 365);

COMMIT;

-- ============================================================================
-- VERIFICATION QUERIES
-- ============================================================================

-- Check new tables
SELECT table_name 
FROM information_schema.tables 
WHERE table_schema = 'public' 
  AND table_name IN ('hcp_segments', 'campaigns', 'hcp_responses', 'drift_detection_results', 
                     'performance_explanations', 'uplift_predictions', 'time_series_forecasts',
                     'feature_interactions', 'version_comparisons')
ORDER BY table_name;

-- Check new columns
SELECT column_name, data_type 
FROM information_schema.columns 
WHERE table_name = 'predictions' 
  AND column_name IN ('prediction_type', 'trx_value', 'nrx_value', 'uplift_value')
ORDER BY column_name;

-- ============================================================================
-- END OF MIGRATION
-- ============================================================================