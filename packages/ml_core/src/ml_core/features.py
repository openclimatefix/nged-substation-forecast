import polars as pl

# Static features registry
# These are basic features that don't require parameterization.
STATIC_FEATURE_REGISTRY: dict[str, pl.Expr] = {
    # Example static features (these would typically be more complex in reality)
    # For now, we'll just define some placeholders or simple expressions
    # Assuming 'valid_time' is available and is a datetime
    "local_time_of_day_sin": (pl.col("valid_time").dt.hour().cast(pl.Float32) / 24.0 * 2 * 3.14159).alias("local_time_of_day_sin"), # Simplified
    "local_day_of_week": pl.col("valid_time").dt.weekday().cast(pl.String).alias("local_day_of_week"), # Simplified
}

def build_lag_expr(base_col: str, lag: int) -> pl.Expr:
    """
    Factory function to build a lag expression.
    
    Why a factory? This allows dynamic Hydra configuration where users can request
    any lag value (e.g., 'power_lag_24', 'power_lag_48') without us needing to
    hardcode every possible lag in the STATIC_FEATURE_REGISTRY.
    
    This connects to the dynamic features documented in the `AllFeatures` contract.
    
    Args:
        base_col: The column to lag (e.g., 'power').
        lag: The number of periods to lag.
        
    Returns:
        A Polars expression for the lagged column.
    """
    return pl.col(base_col).shift(lag).alias(f"{base_col}_lag_{lag}h")

def build_rolling_mean_expr(base_col: str, window: int) -> pl.Expr:
    """
    Factory function to build a rolling mean expression.
    
    Why a factory? Similar to lags, this allows infinite parameterization of
    rolling windows (e.g., 'temperature_rolling_mean_6', 'temperature_rolling_mean_24')
    via configuration, avoiding a bloated static registry.
    
    This connects to the dynamic features documented in the `AllFeatures` contract.
    
    Args:
        base_col: The column to calculate the rolling mean for.
        window: The window size.
        
    Returns:
        A Polars expression for the rolling mean.
    """
    return pl.col(base_col).rolling_mean(window_size=window).alias(f"{base_col}_rolling_mean_{window}h")
