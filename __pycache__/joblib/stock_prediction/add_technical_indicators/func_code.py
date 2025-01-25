# first line: 33
@memory.cache
def add_technical_indicators(df):
    try:
        # Calculate technical indicators
        # Moving averages
        df['MA5'] = df['close'].rolling(window=5).mean()
        df['MA20'] = df['close'].rolling(window=20).mean()
        df['MA50'] = df['close'].rolling(window=50).mean()
        df['MA200'] = df['close'].rolling(window=200).mean()
        
        # Momentum indicators
        df['ROC'] = df['close'].pct_change(periods=12) * 100  # Rate of Change
        df['MOM'] = df['close'].diff(12)  # Momentum
        
        # RSI with different periods
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))  # Using RSI_14 as default RSI
        
        # MACD with signal line
        exp1 = df['close'].ewm(span=12, adjust=False).mean()
        exp2 = df['close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = exp1 - exp2
        df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['MACD_hist'] = df['MACD'] - df['MACD_signal']
        
        # Bollinger Bands (using 20-period as default)
        df['BB_middle'] = df['close'].rolling(window=20).mean()
        df['BB_upper'] = df['BB_middle'] + 2 * df['close'].rolling(window=20).std()
        df['BB_lower'] = df['BB_middle'] - 2 * df['close'].rolling(window=20).std()
        
        # Volume indicators
        df['OBV'] = (df['close'].diff().apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0)) * df['volume']).cumsum()
        df['VWAP'] = (df['volume'] * (df['high'] + df['low'] + df['close']) / 3).cumsum() / df['volume'].cumsum()
        
        # Volatility indicators
        df['ATR'] = df['high'].rolling(window=14).max() - df['low'].rolling(window=14).min()
        df['Volatility'] = df['close'].rolling(window=20).std()
        
        return df
    except Exception as e:
        print(f'Error in technical indicators calculation: {str(e)}')
        raise
