# first line: 39
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
