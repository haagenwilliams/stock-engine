# first line: 27
@memory.cache
def preprocess_data(stock_symbol, df):
    stock_data = df[df['Name'] == stock_symbol].copy()
    stock_data = stock_data.sort_values('date')
    stock_data = add_technical_indicators(stock_data)
    return stock_data
