# Upstox-Integrated Trade-Position Monitor + IV–RV Context Dashboard

A comprehensive Jupyter notebook-based trading dashboard that provides real-time implied volatility analysis and portfolio monitoring using Upstox v3 API.

## Features

### IV-RV Context Dashboard
- **Real-time Volatility Analytics**: IVR/IVP, realized volatility, VRP analysis
- **Regime Classification**: GREEN (long premium), SKY-BLUE (neutral), RED (short premium) bias
- **Advanced Metrics**: Term structure slope, skew analysis, ATR vs Expected Move
- **India VIX Overlay**: Market regime context

### Trade-Position Monitor
- **Live P&L Tracking**: Real-time portfolio P&L from Upstox positions
- **Greeks Analysis**: Net portfolio Greeks (Δ/Γ/Θ/ν) monitoring
- **Stress Testing**: ±1% spot, ±EM/2, ±5 IV pts, +1 day scenarios
- **Action Prompts**: Delta hedge sizing, roll suggestions, harvest plans

## Quick Start

### 1. Prerequisites
- Python 3.11+
- Upstox API credentials (v3)
- Jupyter notebook environment

### 2. Installation
```bash
# Clone or download the notebook
# Install dependencies
pip install -r requirements.txt

# Create .env file with your Upstox credentials
cp .env.example .env
# Edit .env with your actual credentials
```

### 3. Configuration
Create a `.env` file in the same directory:
```env
UPSTOX_API_KEY=your_api_key_here
UPSTOX_API_SECRET=your_api_secret_here
UPSTOX_REDIRECT_URI=your_redirect_uri_here
UPSTOX_ACCESS_TOKEN=your_access_token_here
```

### 4. Usage
```bash
# Start Jupyter notebook
jupyter notebook upstox_trade_monitor.ipynb

# Or use JupyterLab
jupyter lab upstox_trade_monitor.ipynb
```

1. Run all cells in sequence (Cell → Run All)
2. The dashboard will initialize and run self-tests
3. Use the left pane for IV-RV context analysis
4. Use the right pane for position monitoring

## Architecture

### Data Flow
1. **Universe Builder**: Loads NSE F&O instruments, filters for liquid single-stock options
2. **API Client**: Handles Upstox v3 API calls with retry logic and rate limiting
3. **WebSocket Manager**: Manages live market data feeds (placeholder implementation)
4. **Greeks Manager**: Batches and caches option Greeks data (≤50 keys/request)
5. **Positions Manager**: Tracks live portfolio P&L and positions
6. **Analytics Engines**: Calculate volatility metrics and portfolio analytics

### Key Components

#### Volatility Analytics
- **IV Rank/Percentile**: 52-week historical analysis
- **Realized Volatility**: EWMA-based calculation (λ=0.94)
- **VRP**: Volatility Risk Premium (IV² - RV²)
- **Term Structure**: Short vs long-dated IV comparison
- **Skew**: 25-delta risk reversal analysis

#### Portfolio Analytics
- **Net Greeks**: Portfolio-level risk metrics
- **Stress Testing**: Multi-scenario P&L analysis
- **Hedge Sizing**: Delta-neutral position calculations
- **Action Prompts**: Automated trading suggestions

## API Limits & Compliance

- **Greeks API**: Maximum 50 instrument keys per request
- **Rate Limiting**: Exponential backoff on 429 responses
- **WebSocket**: Combined category limits apply
- **Caching**: 6-hour instrument cache, 15s Greeks refresh

## Trading Hours

- **NSE Trading**: 9:15 AM - 3:30 PM IST (Monday-Friday)
- **Auto-pause**: Heavy polling stops after market close
- **Manual Refresh**: Available during off-hours

## Risk Management

### Delta Control
- Target: |Δ_portfolio| ≤ 0.07 of stock-equivalent notional
- Hedge frequency based on Gamma and Expected Move

### Vega Management
- High IVR (≥60) + VRP > 0: Cap net vega
- Low IVR (≤25) + VRP ≤ 0: Allow long vega up to 1.0%

### Profit Protection
- Tier-1: 0.75×(EM/2) → harvest 25-33%
- Tier-2: 1.25×(EM/2) → harvest 50-70%
- Giveback cap: 25-30% of peak P&L

## Troubleshooting

### Common Issues
1. **API Authentication**: Verify credentials in `.env` file
2. **Rate Limits**: Check logs for 429 errors, increase retry delays
3. **Empty Universe**: Ensure NSE_FO exchange is accessible
4. **Missing Greeks**: Check instrument key validity and API limits

### Logs
- Console: INFO level for user feedback
- File: `upstox_monitor.log` with DEBUG level
- Rotation: Daily log file rotation

## Development

### Adding New Features
1. Extend data classes in section 3
2. Add calculation methods in section 5
3. Update UI components in sections 6-7
4. Add tests in section 9

### Testing
Run the self-test suite in section 9 to validate:
- Universe loading (60-90 instruments)
- API connectivity
- Calculation accuracy
- UI component initialization

## Disclaimer

- **Read-Only**: No order placement functionality
- **Educational**: For decision support only
- **Risk**: Options trading involves significant risk
- **Compliance**: Follow local regulations and broker terms

## Support

For issues or questions:
1. Check the troubleshooting section
2. Review API documentation: [Upstox Developer Portal](https://upstox.com/developer/)
3. Verify configuration and credentials

## License

This project is for educational and personal use only. Please ensure compliance with Upstox terms of service and local regulations.