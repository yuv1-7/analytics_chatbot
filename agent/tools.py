from langchain_core.tools import tool
from typing import Optional, List
from core.services.query_service import QueryService

@tool
def get_ensemble_vs_base_performance(
    ensemble_name: str,
    use_case: Optional[str] = None,
    metrics: Optional[List[str]] = None
) -> dict:
    """
    Compare ensemble model performance against its base models.
    
    Use this when user asks:
    - "How does the ensemble perform vs base models?"
    - "Why is the ensemble better/worse?"
    - "Compare ensemble to individual models"
    
    Args:
        ensemble_name: Name of the ensemble model
        use_case: Optional use case filter (NRx_forecasting, HCP_engagement, etc.)
        metrics: Optional list of metrics (rmse, auc_roc, accuracy, r2_score, etc.)
    
    Returns:
        Dictionary with ensemble vs base comparison including metrics and composition
    """
    return QueryService.get_ensemble_vs_base_performance(ensemble_name, use_case, metrics)

@tool
def get_model_performance_summary(
    model_name: Optional[str] = None,
    use_case: Optional[str] = None,
    model_type: Optional[str] = None
) -> dict:
    """
    Get performance summary for models.
    
    Use this when user asks:
    - "How is the model performing?"
    - "Show me model metrics"
    - "What are the latest results?"
    
    Args:
        model_name: Specific model name (e.g., "NRx_Forecaster_v2")
        use_case: Filter by use case (NRx_forecasting, HCP_engagement, etc.)
        model_type: Filter by type (base_model or ensemble)
    
    Returns:
        Dictionary with model performance data and statistics
    """
    return QueryService.get_model_performance_summary(model_name, use_case, model_type)

@tool
def get_drift_detection_summary(
    model_name: Optional[str] = None,
    use_case: Optional[str] = None,
    days_back: int = 30
) -> dict:
    """
    Get drift detection summary for models.
    
    Use this when user asks:
    - "Has the model drifted?"
    - "Show me drift detection results"
    - "Are there any data quality issues?"
    
    Args:
        model_name: Specific model name
        use_case: Filter by use case
        days_back: Number of days to look back (default 30)
    
    Returns:
        Dictionary with drift detection results and history
    """
    return QueryService.get_drift_detection_summary(model_name, use_case, days_back)

@tool
def compare_model_versions(
    model_name: str,
    old_version: str,
    new_version: str,
    metrics: Optional[List[str]] = None
) -> dict:
    """
    Compare two versions of the same model.
    
    Use this when user asks:
    - "Compare version 1 to version 2"
    - "What changed between versions?"
    - "Is the new version better?"
    
    Args:
        model_name: Name of the model
        old_version: Version string for old version (e.g., "v1.0")
        new_version: Version string for new version (e.g., "v2.0")
        metrics: Optional list of specific metrics to compare
    
    Returns:
        Dictionary with version comparison including metrics and feature importance
    """
    return QueryService.compare_model_versions(model_name, old_version, new_version, metrics)

@tool
def get_feature_importance_analysis(
    model_name: str,
    top_n: int = 20
) -> dict:
    """
    Get feature importance analysis for a model.
    
    Use this when user asks:
    - "What are the most important features?"
    - "Which variables drive the predictions?"
    - "Show me feature importance"
    
    Args:
        model_name: Name of the model
        top_n: Number of top features to return (default 20)
    
    Returns:
        Dictionary with ranked feature importance scores
    """
    return QueryService.get_feature_importance_analysis(model_name, top_n)

@tool
def get_prediction_analysis(
    model_name: str,
    entity_type: Optional[str] = None,
    top_n: int = 20
) -> dict:
    """
    Get prediction analysis for a model.
    
    Use this when user asks:
    - "Show me the top predictions"
    - "What are the model's forecasts?"
    - "How accurate are the predictions?"
    
    Args:
        model_name: Name of the model
        entity_type: Optional entity type (HCP, territory, segment, product, account)
        top_n: Number of top predictions to return (default 20)
    
    Returns:
        Dictionary with prediction summary, top predictions, and accuracy metrics
    """
    return QueryService.get_prediction_analysis(model_name, entity_type, top_n)

@tool
def search_models(search_term: str) -> dict:
    """
    Search for models by name, description, or algorithm.
    
    Use this when user asks:
    - "Find models with XGBoost"
    - "Search for forecasting models"
    - "What models do we have?"
    
    Args:
        search_term: Search term to find models
    
    Returns:
        Dictionary with matching models
    """
    return QueryService.search_models(search_term)

# List of all available tools
ALL_TOOLS = [
    get_ensemble_vs_base_performance,
    get_model_performance_summary,
    get_drift_detection_summary,
    compare_model_versions,
    get_feature_importance_analysis,
    get_prediction_analysis,
    search_models
]