
import numpy as np
from sklearn.preprocessing import StandardScaler

def scale_features(df, feature_cols, fit_scaler=True):
    """Apply log transformation + StandardScaler to features"""
    if fit_scaler:
        scaler = StandardScaler()
        # Log transform (adding small value to avoid log(0))
        df_log = np.log(df[feature_cols] + 1e-8)
        scaler.fit(df_log)
        df[feature_cols] = scaler.transform(df_log)
        return df, scaler
    else:
        # Only transform with existing scaler
        return df, None

def inverse_scale(scaler, data):
    """Inverse transform: StandardScaler -> exp -> original values"""
    # First inverse StandardScaler
    data_std_inv = scaler.inverse_transform(data)
    # Then inverse log transformation
    return np.exp(data_std_inv) - 1e-8

def create_labels(df):
    df['label'] = (df['close'].shift(-1) > df['close']).astype(int)
    return df
