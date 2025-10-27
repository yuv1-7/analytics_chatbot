
from sqlalchemy import Column, String, Text, Float, Integer, DateTime, Boolean, Date, ForeignKey, JSON, DECIMAL, Enum, CheckConstraint, UniqueConstraint
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from core.database import Base
import uuid
import enum

# Enums
class ModelTypeEnum(enum.Enum):
    base_model = "base_model"
    ensemble = "ensemble"

class ExecutionStatusEnum(enum.Enum):
    success = "success"
    failed = "failed"
    running = "running"
    pending = "pending"

class DataSplitEnum(enum.Enum):
    train = "train"
    validation = "validation"
    test = "test"
    holdout = "holdout"
    cross_validation = "cross_validation"

class EntityTypeEnum(enum.Enum):
    HCP = "HCP"
    territory = "territory"
    segment = "segment"
    product = "product"
    account = "account"

class DriftTypeEnum(enum.Enum):
    concept_drift = "concept_drift"
    data_drift = "data_drift"
    performance_drift = "performance_drift"
    prediction_drift = "prediction_drift"

# Models
class Model(Base):
    __tablename__ = "models"
    
    model_id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    model_name = Column(String(255), nullable=False)
    model_type = Column(Enum(ModelTypeEnum), nullable=False)
    use_case = Column(String(100), nullable=False)
    version = Column(String(50), nullable=False)
    algorithm = Column(String(100), nullable=False)
    hyperparameters = Column(JSON)
    description = Column(Text)
    created_at = Column(DateTime, server_default=func.current_timestamp())
    created_by = Column(String(100))
    is_active = Column(Boolean, default=True)
    parent_model_id = Column(String, ForeignKey("models.model_id"))
    version_notes = Column(Text)
    deprecated_at = Column(DateTime)
    performance_trend = Column(String(20))
    
    # Relationships
    executions = relationship("ModelExecution", back_populates="model")
    ensemble_memberships = relationship("EnsembleMember", foreign_keys="EnsembleMember.ensemble_id", back_populates="ensemble")
    base_model_memberships = relationship("EnsembleMember", foreign_keys="EnsembleMember.base_model_id", back_populates="base_model")
    
    __table_args__ = (
        UniqueConstraint('model_name', 'version', name='unique_model_version'),
    )

class ModelExecution(Base):
    __tablename__ = "model_executions"
    
    execution_id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    model_id = Column(String, ForeignKey("models.model_id", ondelete="CASCADE"), nullable=False)
    execution_timestamp = Column(DateTime, server_default=func.current_timestamp())
    execution_status = Column(Enum(ExecutionStatusEnum), default=ExecutionStatusEnum.success)
    training_data_start_date = Column(Date)
    training_data_end_date = Column(Date)
    prediction_date = Column(Date, nullable=False)
    execution_metadata = Column(JSON)
    error_message = Column(Text)
    runtime_seconds = Column(Integer)
    baseline_execution_id = Column(String, ForeignKey("model_executions.execution_id"))
    drift_detected = Column(Boolean, default=False)
    drift_score = Column(Float)
    
    # Relationships
    model = relationship("Model", back_populates="executions")
    performance_metrics = relationship("PerformanceMetric", back_populates="execution")
    predictions = relationship("Prediction", back_populates="execution")
    feature_importance = relationship("FeatureImportance", back_populates="execution")

class EnsembleMember(Base):
    __tablename__ = "ensemble_members"
    
    ensemble_id = Column(String, ForeignKey("models.model_id", ondelete="CASCADE"), primary_key=True)
    base_model_id = Column(String, ForeignKey("models.model_id", ondelete="CASCADE"), primary_key=True)
    weight = Column(Float)
    role = Column(String(50))
    added_at = Column(DateTime, server_default=func.current_timestamp())
    configuration = Column(JSON)
    ensemble_type = Column(String(50))
    
    # Relationships
    ensemble = relationship("Model", foreign_keys=[ensemble_id], back_populates="ensemble_memberships")
    base_model = relationship("Model", foreign_keys=[base_model_id], back_populates="base_model_memberships")

class PerformanceMetric(Base):
    __tablename__ = "performance_metrics"
    
    metric_id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    execution_id = Column(String, ForeignKey("model_executions.execution_id", ondelete="CASCADE"), nullable=False)
    metric_name = Column(String(100), nullable=False)
    metric_value = Column(Float, nullable=False)
    data_split = Column(Enum(DataSplitEnum), nullable=False)
    metric_metadata = Column(JSON)
    calculated_at = Column(DateTime, server_default=func.current_timestamp())
    
    # Relationships
    execution = relationship("ModelExecution", back_populates="performance_metrics")
    
    __table_args__ = (
        UniqueConstraint('execution_id', 'metric_name', 'data_split', name='unique_metric_per_execution'),
    )

class FeatureImportance(Base):
    __tablename__ = "feature_importance"
    
    importance_id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    execution_id = Column(String, ForeignKey("model_executions.execution_id", ondelete="CASCADE"), nullable=False)
    feature_name = Column(String(255), nullable=False)
    importance_score = Column(Float, nullable=False)
    importance_type = Column(String(50), nullable=False)
    rank = Column(Integer, nullable=False)
    standard_error = Column(Float)
    calculated_at = Column(DateTime, server_default=func.current_timestamp())
    
    # Relationships
    execution = relationship("ModelExecution", back_populates="feature_importance")
    
    __table_args__ = (
        UniqueConstraint('execution_id', 'feature_name', 'importance_type', name='unique_feature_per_execution'),
        CheckConstraint('rank > 0', name='feature_importance_rank_check')
    )

class Prediction(Base):
    __tablename__ = "predictions"
    
    prediction_id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    execution_id = Column(String, ForeignKey("model_executions.execution_id", ondelete="CASCADE"), nullable=False)
    entity_type = Column(Enum(EntityTypeEnum), nullable=False)
    entity_id = Column(String(255), nullable=False)
    prediction_value = Column(Float, nullable=False)
    prediction_probability = Column(Float)
    prediction_class = Column(String(100))
    actual_value = Column(Float)
    residual = Column(Float)
    prediction_date = Column(Date, nullable=False)
    confidence_interval_lower = Column(Float)
    confidence_interval_upper = Column(Float)
    created_at = Column(DateTime, server_default=func.current_timestamp())
    prediction_type = Column(String(50))
    trx_value = Column(Float)
    nrx_value = Column(Float)
    control_prediction = Column(Float)
    treatment_prediction = Column(Float)
    uplift_value = Column(Float)
    
    # Relationships
    execution = relationship("ModelExecution", back_populates="predictions")

class FeaturesMetadata(Base):
    __tablename__ = "features_metadata"
    
    feature_id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    feature_name = Column(String(255), nullable=False, unique=True)
    feature_description = Column(Text)
    feature_type = Column(String(50))
    business_meaning = Column(Text)
    source_system = Column(String(100))
    calculation_logic = Column(Text)
    data_quality_notes = Column(Text)
    created_at = Column(DateTime, server_default=func.current_timestamp())
    updated_at = Column(DateTime, server_default=func.current_timestamp(), onupdate=func.current_timestamp())

class DriftDetectionResult(Base):
    __tablename__ = "drift_detection_results"
    
    drift_id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    execution_id = Column(String, ForeignKey("model_executions.execution_id", ondelete="CASCADE"), nullable=False)
    baseline_execution_id = Column(String, ForeignKey("model_executions.execution_id", ondelete="CASCADE"), nullable=False)
    drift_type = Column(Enum(DriftTypeEnum))
    drift_metric = Column(String(100), nullable=False)
    drift_score = Column(Float, nullable=False)
    threshold_value = Column(Float)
    is_significant = Column(Boolean, default=False)
    affected_features = Column(JSON)
    drift_explanation = Column(Text)
    detected_at = Column(DateTime, server_default=func.current_timestamp())
    
    __table_args__ = (
        CheckConstraint('execution_id != baseline_execution_id', name='no_self_baseline'),
    )

class ModelComparison(Base):
    __tablename__ = "model_comparisons"
    
    comparison_id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    model_1_id = Column(String, ForeignKey("models.model_id", ondelete="CASCADE"), nullable=False)
    model_2_id = Column(String, ForeignKey("models.model_id", ondelete="CASCADE"), nullable=False)
    comparison_type = Column(String(50), nullable=False)
    comparison_results = Column(JSON, nullable=False)
    execution_1_id = Column(String, ForeignKey("model_executions.execution_id", ondelete="CASCADE"))
    execution_2_id = Column(String, ForeignKey("model_executions.execution_id", ondelete="CASCADE"))
    created_at = Column(DateTime, server_default=func.current_timestamp())
    
    __table_args__ = (
        CheckConstraint('model_1_id != model_2_id', name='no_self_comparison'),
    )

class PerformanceExplanation(Base):
    __tablename__ = "performance_explanations"
    
    explanation_id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    execution_id = Column(String, ForeignKey("model_executions.execution_id", ondelete="CASCADE"), nullable=False)
    comparison_execution_id = Column(String, ForeignKey("model_executions.execution_id", ondelete="CASCADE"))
    explanation_type = Column(String(50))
    primary_reason = Column(Text, nullable=False)
    contributing_factors = Column(JSON)
    quantitative_evidence = Column(JSON)
    recommendation = Column(Text)
    confidence_score = Column(Float)
    generated_by = Column(String(100), default='agent_system')
    created_at = Column(DateTime, server_default=func.current_timestamp())

class HCPSegment(Base):
    __tablename__ = "hcp_segments"
    
    segment_id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    segment_name = Column(String(100), nullable=False, unique=True)
    segment_description = Column(Text)
    criteria = Column(JSON)
    tier = Column(String(20))
    hcp_count = Column(Integer)
    created_at = Column(DateTime, server_default=func.current_timestamp())
    updated_at = Column(DateTime, server_default=func.current_timestamp(), onupdate=func.current_timestamp())

class Campaign(Base):
    __tablename__ = "campaigns"
    
    campaign_id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    campaign_name = Column(String(255), nullable=False)
    campaign_type = Column(String(100))
    message_content = Column(Text)
    target_segment_id = Column(String, ForeignKey("hcp_segments.segment_id"))
    start_date = Column(Date, nullable=False)
    end_date = Column(Date)
    budget = Column(DECIMAL(12, 2))
    campaign_metadata = Column(JSON)
    created_at = Column(DateTime, server_default=func.current_timestamp())

class HCPResponse(Base):
    __tablename__ = "hcp_responses"
    
    response_id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    hcp_id = Column(String(255), nullable=False)
    campaign_id = Column(String, ForeignKey("campaigns.campaign_id", ondelete="CASCADE"))
    response_date = Column(Date, nullable=False)
    responded = Column(Boolean, nullable=False)
    response_type = Column(String(50))
    prescription_change_trx = Column(Float)
    prescription_change_nrx = Column(Float)
    response_metadata = Column(JSON)
    created_at = Column(DateTime, server_default=func.current_timestamp())

class UpliftPrediction(Base):
    __tablename__ = "uplift_predictions"
    
    uplift_id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    execution_id = Column(String, ForeignKey("model_executions.execution_id", ondelete="CASCADE"), nullable=False)
    entity_id = Column(String(255), nullable=False)
    campaign_id = Column(String, ForeignKey("campaigns.campaign_id"))
    control_prediction = Column(Float, nullable=False)
    treatment_prediction = Column(Float, nullable=False)
    uplift_score = Column(Float, nullable=False)
    confidence_interval_lower = Column(Float)
    confidence_interval_upper = Column(Float)
    recommended_action = Column(String(255))
    action_priority = Column(Integer)
    predicted_roi = Column(Float)
    prediction_date = Column(Date, nullable=False)
    created_at = Column(DateTime, server_default=func.current_timestamp())

class TimeSeriesForecast(Base):
    __tablename__ = "time_series_forecasts"
    
    forecast_id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    execution_id = Column(String, ForeignKey("model_executions.execution_id", ondelete="CASCADE"), nullable=False)
    entity_id = Column(String(255), nullable=False)
    forecast_date = Column(Date, nullable=False)
    forecast_horizon_days = Column(Integer, nullable=False)
    point_forecast = Column(Float, nullable=False)
    lower_bound = Column(Float)
    upper_bound = Column(Float)
    forecast_type = Column(String(50))
    seasonality_component = Column(Float)
    trend_component = Column(Float)
    residual_component = Column(Float)
    created_at = Column(DateTime, server_default=func.current_timestamp())

class FeatureInteraction(Base):
    __tablename__ = "feature_interactions"
    
    interaction_id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    execution_id = Column(String, ForeignKey("model_executions.execution_id", ondelete="CASCADE"), nullable=False)
    feature_1 = Column(String(255), nullable=False)
    feature_2 = Column(String(255), nullable=False)
    interaction_strength = Column(Float, nullable=False)
    interaction_type = Column(String(50))
    business_interpretation = Column(Text)
    created_at = Column(DateTime, server_default=func.current_timestamp())
    
    __table_args__ = (
        CheckConstraint('feature_1 != feature_2', name='no_self_interaction'),
    )

class VersionComparison(Base):
    __tablename__ = "version_comparisons"
    
    comparison_id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    model_id = Column(String, ForeignKey("models.model_id", ondelete="CASCADE"), nullable=False)
    old_version = Column(String(50), nullable=False)
    new_version = Column(String(50), nullable=False)
    old_execution_id = Column(String, ForeignKey("model_executions.execution_id"), nullable=False)
    new_execution_id = Column(String, ForeignKey("model_executions.execution_id"), nullable=False)
    metric_changes = Column(JSON, nullable=False)
    performance_verdict = Column(String(50))
    key_differences = Column(Text)
    rollback_recommended = Column(Boolean, default=False)
    compared_at = Column(DateTime, server_default=func.current_timestamp())