# Stock Price Prediction with LSTM and Technical Analysis

A sophisticated stock price prediction system that combines deep learning (LSTM) with technical analysis indicators to forecast stock price movements. The system features real-time predictions, interactive visualizations, and comprehensive technical analysis.

## Features

- Deep Learning-based stock price prediction using LSTM neural networks
- Comprehensive technical analysis with multiple indicators:
  - Moving Averages (MA5, MA20, MA50, MA200)
  - Relative Strength Index (RSI)
  - Moving Average Convergence Divergence (MACD)
  - Bollinger Bands
  - Rate of Change (ROC)
  - Momentum Indicators
  - Volume-based indicators (OBV, VWAP)
  - Volatility measures (ATR)
- Interactive web interface for real-time predictions
- Model caching for improved performance
- Robust error handling and data validation

## Technical Architecture

### LSTM Model Structure
- Multi-layer LSTM architecture with batch normalization and dropout
- Skip connections for better gradient flow
- Regularization techniques to prevent overfitting
- Adaptive learning rate with gradient clipping

### Technical Indicators
- Multiple timeframe analysis
- Advanced momentum and trend indicators
- Volume-price relationship analysis
- Volatility measurement

## Installation

1. Clone the repository
2. Install the required dependencies:
```bash
pip install pandas numpy tensorflow scikit-learn joblib flask seaborn matplotlib
```

## Usage

1. Start the web server:
```bash
python app.py
```

2. Access the web interface at `http://localhost:5000`
3. Select a stock symbol and view predictions

## Project Structure

- `app.py`: Web application and API endpoints
- `stock_prediction.py`: Core prediction logic and model definition
- `news_api.py`: News integration module
- `templates/`: Web interface templates

## Model Performance

The system uses various metrics to evaluate prediction accuracy:
- Root Mean Square Error (RMSE)
- Mean Absolute Error (MAE)
- Mean Absolute Percentage Error (MAPE)

## Data Processing

- Real-time technical indicator calculation
- Automated data cleaning and normalization
- Sequence creation for time series analysis
- Efficient data caching for improved performance

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.