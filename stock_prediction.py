import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, LSTM, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from joblib import Memory, Parallel, delayed
import os

# Set up caching
cachedir = './__pycache__'
memory = Memory(cachedir, verbose=0)

# Load and preprocess the data
def load_data(file_path):
    df = pd.read_csv(file_path)
    # Convert date to datetime
    df['date'] = pd.to_datetime(df['date'])
    return df

# Create sequences for LSTM
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:(i + seq_length)])
        y.append(data[i + seq_length])
    return np.array(X), np.array(y)

# Build LSTM model
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

def build_model(seq_length, n_features):
    model = Sequential([
        # First LSTM layer with increased units and regularization
        LSTM(256, return_sequences=True, input_shape=(seq_length, n_features),
             kernel_regularizer='l2', recurrent_regularizer='l2'),
        BatchNormalization(),
        Dropout(0.3),
        
        # Second LSTM layer with skip connection
        LSTM(128, return_sequences=True),
        BatchNormalization(),
        Dropout(0.3),
        
        # Third LSTM layer
        LSTM(64, return_sequences=True),
        BatchNormalization(),
        Dropout(0.3),
        
        # Fourth LSTM layer
        LSTM(32, return_sequences=False),
        BatchNormalization(),
        Dropout(0.3),
        
        # Dense layers with increased capacity
        Dense(64, activation='relu', kernel_regularizer='l2'),
        BatchNormalization(),
        Dropout(0.2),
        
        Dense(32, activation='relu', kernel_regularizer='l2'),
        BatchNormalization(),
        Dropout(0.2),
        
        Dense(16, activation='relu'),
        Dense(1)
    ])
    
    # Use a custom learning rate schedule and gradient clipping
    optimizer = Adam(learning_rate=0.001, clipnorm=1.0)
    model.compile(optimizer=optimizer, 
                 loss='huber',
                 metrics=['mae', 'mse', 'mape'])
    return model

def main():
    # Load the data
    df = load_data('all_stocks_5yr.csv')
    
    # Select a single stock (e.g., 'AAPL')
    stock_data = df[df['Name'] == 'AAPL'].copy()
    stock_data = stock_data.sort_values('date')
    
    # Add technical indicators
    stock_data = add_technical_indicators(stock_data)
    
    # Prepare features
    feature_columns = ['close', 'MA5', 'MA20', 'RSI', 'MACD', 'BB_middle', 'BB_upper', 'BB_lower', 'ROC', 'MOM']
    features = stock_data[feature_columns].values
    
    # Handle NaN values
    features = features[~np.isnan(features).any(axis=1)]
    
    # Scale the data
    scaler = MinMaxScaler()
    scaled_features = scaler.fit_transform(features)
    
    # Create sequences
    seq_length = 60  # Use 60 days of historical data to predict the next day
    X, y = [], []
    for i in range(len(scaled_features) - seq_length):
        X.append(scaled_features[i:(i + seq_length)])
        y.append(scaled_features[i + seq_length, 0])  # Predict only the closing price
    X, y = np.array(X), np.array(y)
    
    # Split into train, validation and test sets
    train_size = int(len(X) * 0.7)
    val_size = int(len(X) * 0.15)
    X_train = X[:train_size]
    y_train = y[:train_size]
    X_val = X[train_size:train_size+val_size]
    y_val = y[train_size:train_size+val_size]
    X_test = X[train_size+val_size:]
    y_test = y[train_size+val_size:]
    
    # Build and train the model
    model = build_model(seq_length, len(feature_columns))
    
    # Add callbacks for better training
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True,
        mode='min'
    )
    
    from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
    
    checkpoint = ModelCheckpoint(
        f'model_checkpoint_{stock_data["Name"].iloc[0]}.h5',
        monitor='val_loss',
        save_best_only=True,
        mode='min',
        verbose=0
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=3,
        min_lr=0.0001,
        mode='min',
        verbose=0
    )
    
    # Check if cached model exists
    model_path = f'cached_model_{stock_data["Name"].iloc[0]}.h5'
    if os.path.exists(model_path):
        model = load_model(model_path)
    else:
        history = model.fit(
            X_train, y_train,
            epochs=50,
            batch_size=64,
            validation_split=0.1,
            callbacks=[early_stopping],
            verbose=1
        )
        # Cache the model
        model.save(model_path)
    
    # Make predictions
    train_predict = model.predict(X_train)
    test_predict = model.predict(X_test)
    
    # Inverse transform predictions
    train_predict = scaler.inverse_transform(train_predict)
    y_train_inv = scaler.inverse_transform(y_train)
    test_predict = scaler.inverse_transform(test_predict)
    y_test_inv = scaler.inverse_transform(y_test)
    
    # Plot results
    plt.figure(figsize=(15, 6))
    plt.plot(stock_data['date'].values[seq_length:train_size+seq_length], y_train_inv, label='Actual Train')
    plt.plot(stock_data['date'].values[seq_length:train_size+seq_length], train_predict, label='Predicted Train')
    plt.plot(stock_data['date'].values[train_size+seq_length:], y_test_inv, label='Actual Test')
    plt.plot(stock_data['date'].values[train_size+seq_length:], test_predict, label='Predicted Test')
    plt.title('AAPL Stock Price Prediction')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.savefig('stock_prediction_results.png')
    plt.close()
    
    # Print model performance
    train_rmse = np.sqrt(np.mean((train_predict - y_train_inv) ** 2))
    test_rmse = np.sqrt(np.mean((test_predict - y_test_inv) ** 2))
    print(f'Train RMSE: {train_rmse:.2f}')
    print(f'Test RMSE: {test_rmse:.2f}')

if __name__ == '__main__':
    main()