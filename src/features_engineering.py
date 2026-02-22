def features_engineering(df, window_1=5, window_2=20):
    df = df.copy()

    # Lags pollution (1 à 5 jours)
    for lag in range(1, 6):
        df[f'pm25_lag{lag}'] = df['pm25'].shift(lag)

    # Moving averages pollution
    df['pm25_ma5']  = df['pm25'].rolling(window_1).mean()
    df['pm25_ma20'] = df['pm25'].rolling(window_2).mean()

    # Volatilité de la pollution (rolling std)
    df['pm25_vol5']  = df['pm25'].rolling(window_1).std()
    df['pm25_vol20'] = df['pm25'].rolling(window_2).std()

    # Pollution standardisée (pour le SDE plus tard)
    df['pm25_std'] = (df['pm25'] - df['pm25'].mean()) / df['pm25'].std()

    # Lags realized vol (utile pour GARCH-X et le SDE)
    df['rvol_lag1'] = df['realized_vol'].shift(1)
    df['rvol_lag2'] = df['realized_vol'].shift(2)

    df = df.dropna()

    return df