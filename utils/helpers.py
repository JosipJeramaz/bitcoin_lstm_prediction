
from sklearn.preprocessing import MinMaxScaler

def scale_features(df, feature_cols):
    scaler = MinMaxScaler()
    df[feature_cols] = scaler.fit_transform(df[feature_cols])
    return df, scaler

def inverse_scale(scaler, data):
    return scaler.inverse_transform(data)

def create_labels(df):
    df['label'] = (df['close'].shift(-1) > df['close']).astype(int)
    return df
