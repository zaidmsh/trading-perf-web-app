# Trading Performance Analyzer

A comprehensive web application for analyzing trading performance from Interactive Brokers (IBKR) trade data. Built with FastAPI and designed to run locally using Docker.

## Features

### 📊 Enhanced Performance Metrics
- **Batting Average**: Win rate percentage of your trades
- **Average Gain**: Average return on winning trades only
- **Average Loss**: Average loss on losing trades only (absolute value)
- **Win/Loss Ratio**: Simple ratio of average gain to average loss
- **Adjusted Win/Loss Ratio**: Ratio adjusted for batting average frequency

### 📈 Comprehensive Analysis
- Monthly, quarterly, and yearly performance breakdowns
- Interactive charts powered by Chart.js
- Color-coded performance indicators
- Detailed trade statistics and metrics

### 💾 Export Options
- Download processed roundtrips as CSV
- Export performance data as Excel workbooks
- Individual period exports (monthly, quarterly, yearly)
- Complete performance reports

### 🚀 Modern Web Interface
- Drag-and-drop file upload
- Responsive Bootstrap design
- Real-time processing feedback
- Professional data visualization

## Quick Start with Docker

### Prerequisites
- Docker and Docker Compose installed on your system
- Interactive Brokers CSV trade data

### 1. Start the Application
```bash
# Navigate to the project directory
cd trading-web-app

# Build and start the application
docker-compose up --build

# Or run in detached mode
docker-compose up -d --build
```

### 2. Access the Application
Open your browser and navigate to:
```
http://localhost:8000
```

### 3. Upload Your IBKR Data
1. Click the upload area or drag and drop your CSV file
2. Wait for processing to complete
3. View your comprehensive performance analysis

### 4. Stop the Application
```bash
# Stop the containers
docker-compose down

# Stop and remove volumes (removes uploaded files)
docker-compose down -v
```

## IBKR CSV Requirements

Your CSV file should contain the following columns (flexible naming supported):

### Required Columns
- **Symbol** (or "Underlying Symbol", "Contract Description")
- **Quantity** (or "Qty")
- **Price** (or "Trade Price", "T. Price")
- **Date** (or "Trade Date", "TradeDate", "Trade Date/Time")
- **Side** (or "Buy/Sell")

### Optional Columns
- **Commission** (or "Comm/Fee", "IB Commission")
- **Account** (or "Account Alias", "Account Name")
- **Order Reference** (or "Order Ref", "Client Order Id")

### Supported Date Formats
- `YYYYMMDD` (e.g., 20250921)
- `YYYY-MM-DD` (e.g., 2025-09-21)
- Standard date formats parsed by dateutil

## Development

### Project Structure
```
trading-web-app/
├── app.py                 # FastAPI application
├── core/
│   ├── processor.py       # IBKR data processing
│   └── calculator.py      # Performance calculations
├── templates/
│   ├── base.html         # Base template
│   ├── index.html        # Upload page
│   └── results.html      # Results display
├── static/
│   ├── css/style.css     # Custom styling
│   └── js/app.js         # Frontend JavaScript
├── Dockerfile            # Container configuration
├── docker-compose.yml    # Local development setup
└── requirements.txt      # Python dependencies
```

### Running Without Docker
If you prefer to run without Docker:

```bash
# Install Python dependencies
pip install -r requirements.txt

# Start the development server
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

### Customization
- **File Size Limit**: Modify `MAX_FILE_SIZE` in docker-compose.yml
- **Session Timeout**: Adjust `SESSION_TIMEOUT` environment variable
- **Styling**: Edit `static/css/style.css` for custom themes
- **Charts**: Modify chart configurations in `static/js/app.js`

## Performance Metrics Explained

### Batting Average
The percentage of trades that resulted in profits. A batting average above 50% means more than half of your trades were profitable.

### Average Gain/Loss
- **Average Gain**: Mean return percentage of only your winning trades
- **Average Loss**: Mean loss percentage of only your losing trades (shown as positive)

### Win/Loss Ratios
- **Standard Ratio**: Average Gain ÷ Average Loss
- **Adjusted Ratio**: Standard ratio × (Batting Average ÷ 100)

The adjusted ratio accounts for how often you win versus lose, providing a more complete picture of your trading performance.

## API Endpoints

The application provides several API endpoints:

- `GET /` - Main upload interface
- `POST /upload` - File upload and processing
- `GET /results/{session_id}` - View results
- `GET /api/results/{session_id}` - JSON API for results
- `GET /export/{session_id}` - Export data in various formats

## Security Features

- Non-root container user
- File size and type validation
- Session-based data isolation
- Automatic cleanup of old data
- Health check monitoring

## Troubleshooting

### Common Issues

**Container won't start:**
```bash
# Check Docker logs
docker-compose logs trading-analyzer

# Rebuild containers
docker-compose down
docker-compose up --build
```

**File upload fails:**
- Ensure file is CSV format
- Check file size (max 10MB)
- Verify required columns are present

**Charts not displaying:**
- Check browser console for JavaScript errors
- Ensure Chart.js is loading properly
- Verify data format in network tab

### Health Check
The application includes health checks accessible at:
```
http://localhost:8000/
```

## Performance Considerations

- **Memory Usage**: ~100MB for typical datasets
- **Processing Time**: ~1-5 seconds for 1000 trades
- **File Limits**: 10MB maximum file size
- **Session Storage**: In-memory (lost on restart)

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test with Docker
5. Submit a pull request

## License

This project is provided as-is for educational and personal use.

---

**Happy Trading Analysis! 📈**