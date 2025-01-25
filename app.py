from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, LSTM, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from stock_prediction import create_sequences, build_model, add_technical_indicators
from joblib import Memory
import os
from news_api import NewsAPI

# Initialize Flask app
app = Flask(__name__)

# Initialize memory cache
memory = Memory(location='__pycache__/joblib', verbose=0)

# Initialize NewsAPI
news_api = NewsAPI()

@app.route('/')
def index():
    # Read the unique stock symbols from the CSV file
    df = pd.read_csv('all_stocks_5yr.csv')
    stocks = sorted(df['Name'].unique())
    return render_template('index.html', stocks=stocks)

@app.route('/news/<symbol>')
def get_news(symbol):
    try:
        news_items = news_api.get_news(symbol)
        return jsonify(news_items)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Cache data preprocessing
@memory.cache
def preprocess_data(stock_symbol, df):
    try:
        # Filter and sort data
        stock_data = df[df['Name'] == stock_symbol].copy()
        if stock_data.empty:
            return pd.DataFrame()
            
        stock_data = stock_data.sort_values('date')
        
        # Add technical indicators with error handling
        try:
            stock_data = add_technical_indicators(stock_data)
            
            # Verify required columns exist
            required_columns = ['close', 'MA5', 'MA20', 'RSI', 'MACD', 
                              'BB_middle', 'BB_upper', 'BB_lower', 
                              'ROC', 'MOM', 'volume', 'OBV', 'VWAP', 'ATR', 'Volatility']
                              
            # No need for column mapping as we use consistent names
            column_mapping = {}
            
            # Apply column mapping
            for old_col, new_col in column_mapping.items():
                if old_col in stock_data.columns and new_col not in stock_data.columns:
                    stock_data[new_col] = stock_data[old_col]
            
            # Verify all required columns exist
            missing_columns = [col for col in required_columns if col not in stock_data.columns]
            if missing_columns:
                print(f'Missing columns after preprocessing: {missing_columns}')
                return pd.DataFrame()
                
            return stock_data
            
        except Exception as e:
            print(f'Error in technical indicators calculation: {str(e)}')
            return pd.DataFrame()
            
    except Exception as e:
        print(f'Error in data preprocessing: {str(e)}')
        return pd.DataFrame()

@app.route('/predict', methods=['POST'])
def predict():
    try:
        print('Starting prediction process...')
        stock_symbol = request.form['stock']
        
        # Load and preprocess the data with caching and parallel processing
        print('Loading data from CSV...')
        df = pd.read_csv('all_stocks_5yr.csv', nrows=None)
        df['date'] = pd.to_datetime(df['date'], cache=True)
        print('Data loaded successfully.')
        
        # Preprocess the data
        stock_data = preprocess_data(stock_symbol, df)
        if stock_data.empty:
            return jsonify({'error': 'No data found for selected stock'}), 404
        
        # Prepare features (matching with stock_prediction.py)
        feature_columns = ['close', 'MA5', 'MA20', 'RSI', 'MACD', 'BB_middle', 'BB_upper', 'BB_lower', 'ROC', 'MOM']
        
        # Debug: Print available columns
        print('Available columns:', stock_data.columns.tolist())
        
        # Ensure all required columns exist
        missing_columns = [col for col in feature_columns if col not in stock_data.columns]
        if missing_columns:
            print('Missing columns:', missing_columns)
            print('Data shape:', stock_data.shape)
            return jsonify({'error': f'Missing required technical indicators: {missing_columns}'}), 500
            
        features = stock_data[feature_columns].values
        
        # Handle NaN values
        valid_indices = ~np.isnan(features).any(axis=1)
        if not valid_indices.any():
            return jsonify({'error': 'Insufficient data after processing'}), 400
            
        features = features[valid_indices]
        
        # Scale the data with improved error handling
        scaler = MinMaxScaler()
        try:
            if features.size == 0:
                return jsonify({'error': 'No valid features available for scaling'}), 400
            
            # Check for infinite values
            if np.any(np.isinf(features)):
                features = np.nan_to_num(features, nan=0, posinf=1e10, neginf=-1e10)
            
            scaled_features = scaler.fit_transform(features)
            
            # Validate scaled data
            if np.any(np.isnan(scaled_features)):
                return jsonify({'error': 'Invalid values after scaling'}), 500
                
            # Store scaler for later use
            feature_scaler = scaler
        except Exception as e:
            print(f'Error in data scaling: {str(e)}')
            return jsonify({'error': 'Error during data scaling'}), 500
        
        # Create sequences
        seq_length = 60
        if len(scaled_features) <= seq_length:
            return jsonify({'error': 'Insufficient data for prediction'}), 400
            
        X, y = [], []
        for i in range(len(scaled_features) - seq_length):
            X.append(scaled_features[i:(i + seq_length)])
            y.append(scaled_features[i + seq_length, 0])  # Predict only the closing price
        X, y = np.array(X), np.array(y)
        
        # Split into train and test sets
        train_size = int(len(X) * 0.7)
        X_train = X[:train_size]
        y_train = y[:train_size]
        X_test = X[train_size:]
        y_test = y[train_size:]
        
        # Build and train model
        model = build_model(seq_length, len(feature_columns))
        
        # Add callbacks for better training
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
            mode='min'
        )
        
        # Check if cached model exists and load it with proper error handling
        model_path = f'cached_model_{stock_symbol}.h5'
        try:
            print(f'Compiling model for {stock_symbol}...')
            model.compile(optimizer=Adam(learning_rate=0.001), loss='huber', metrics=['mae', 'mse'])
            print(f'Model compilation successful for {stock_symbol}')
            
            if os.path.exists(model_path):
                try:
                    print(f'Loading cached model for {stock_symbol}...')
                    model = load_model(model_path)
                    print(f'Successfully loaded cached model for {stock_symbol}')
                except Exception as e:
                    print(f'Error loading cached model: {str(e)}')
                    print('Training new model...')
                    history = model.fit(
                        X_train, y_train,
                        epochs=50,
                        batch_size=64,
                        validation_split=0.1,
                        callbacks=[early_stopping],
                        verbose=1
                    )
                    model.save(model_path)
            else:
                print('Training new model...')
                history = model.fit(
                    X_train, y_train,
                    epochs=50,
                    batch_size=64,
                    validation_split=0.1,
                    callbacks=[early_stopping],
                    verbose=1
                )
                model.save(model_path)
        except Exception as e:
            print(f'Error in model preparation: {str(e)}')
            return jsonify({'error': 'Error in model preparation'}), 500

        try:
            # Make predictions with improved error handling
            print('Making predictions...')
            try:
                train_predict = model.predict(X_train, verbose=0)
                test_predict = model.predict(X_test, verbose=0)
            except Exception as e:
                print(f'Error during model prediction: {str(e)}')
                return jsonify({'error': 'Failed to make predictions'}), 500
            
            # Validate predictions
            if np.any(np.isnan(train_predict)) or np.any(np.isnan(test_predict)):
                return jsonify({'error': 'Invalid prediction values detected'}), 500
            
            # Ensure proper shapes for scaling
            train_predict = np.array(train_predict).reshape(-1, 1)
            test_predict = np.array(test_predict).reshape(-1, 1)
            y_train = np.array(y_train).reshape(-1, 1)
            y_test = np.array(y_test).reshape(-1, 1)
            
            # Create properly shaped arrays for inverse transform
            train_predict_full = np.zeros((train_predict.shape[0], len(feature_columns)))
            test_predict_full = np.zeros((test_predict.shape[0], len(feature_columns)))
            y_train_full = np.zeros((y_train.shape[0], len(feature_columns)))
            y_test_full = np.zeros((y_test.shape[0], len(feature_columns)))
            
            # Fill in the closing price values
            train_predict_full[:, 0] = train_predict.ravel()
            test_predict_full[:, 0] = test_predict.ravel()
            y_train_full[:, 0] = y_train.ravel()
            y_test_full[:, 0] = y_test.ravel()
            
            # Perform inverse transform with error handling
            try:
                train_predict_inv = feature_scaler.inverse_transform(train_predict_full)[:, 0]
                test_predict_inv = feature_scaler.inverse_transform(test_predict_full)[:, 0]
                y_train_inv = feature_scaler.inverse_transform(y_train_full)[:, 0]
                y_test_inv = feature_scaler.inverse_transform(y_test_full)[:, 0]
            except Exception as e:
                print(f'Error during inverse transformation: {str(e)}')
                return jsonify({'error': 'Failed to inverse transform predictions'}), 500
            
            # Prepare dates
            dates = stock_data['date'].values[valid_indices][seq_length:]
            train_dates = dates[:train_size]
            test_dates = dates[train_size:]
            
            # Calculate RMSE
            train_rmse = np.sqrt(np.mean((train_predict_inv - y_train_inv) ** 2))
            test_rmse = np.sqrt(np.mean((test_predict_inv - y_test_inv) ** 2))
            
            response_data = {
                'dates': {
                    'train': train_dates.astype(str).tolist(),
                    'test': test_dates.astype(str).tolist()
                },
                'actual': {
                    'train': y_train_inv.tolist(),
                    'test': y_test_inv.tolist()
                },
                'predicted': {
                    'train': train_predict_inv.tolist(),
                    'test': test_predict_inv.tolist()
                },
                'rmse': {
                    'train': float(train_rmse),
                    'test': float(test_rmse)
                }
            }
            
            return jsonify(response_data)
            
        except Exception as e:
            print(f'Error in prediction process: {str(e)}')
            return jsonify({'error': 'Error during prediction process'}), 500
            
    except Exception as e:
        print(f'Error in prediction: {str(e)}')
        print(f'Error type: {type(e).__name__}')
        print(f'Error details: {e.args}')
        return jsonify({'error': f'An error occurred during prediction: {str(e)}'}), 500

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=8000, help='Port to run the server on')
    args = parser.parse_args()
    app.run(port=args.port, debug=True)