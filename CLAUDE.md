# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Trading Performance Analyzer is a FastAPI web application for analyzing Interactive Brokers (IBKR) trading performance. It processes CSV trade data, calculates roundtrips, and generates comprehensive performance metrics with data visualization.

## Architecture

### Core Components

- **FastAPI Application** (`app.py`): Main web server handling upload, processing, and API endpoints
- **Data Processing Layer** (`core/`):
  - `processor.py`: IBKR CSV parsing and roundtrip calculation
  - `calculator.py`: Performance metrics computation (batting average, win/loss ratios, period analysis)
- **Frontend**: Jinja2 templates with Chart.js visualization and Bootstrap UI
- **Storage**: In-memory results store with session-based data isolation

### Data Flow

1. CSV upload via drag-drop interface
2. IBKR trade parsing with flexible column mapping
3. Roundtrip aggregation (matching buy/sell pairs)
4. Performance calculation across multiple time periods
5. Interactive visualization with export capabilities

### Key Features

- **Flexible CSV Parsing**: Handles various IBKR column naming conventions
- **Advanced Metrics**: Batting average, adjusted win/loss ratios, period-based analysis
- **Export Options**: CSV, Excel workbooks with multiple sheets
- **Session Management**: UUID-based data isolation with automatic cleanup

## Development Commands

### Docker Development (Recommended)
```bash
# Start application with hot reload
docker-compose up --build

# Start in background
docker-compose up -d --build

# View logs
docker-compose logs trading-analyzer

# Stop containers
docker-compose down

# Clean up including volumes
docker-compose down -v
```

### Local Development
```bash
# Install dependencies
pip install -r requirements.txt

# Run with hot reload
uvicorn app:app --reload --host 0.0.0.0 --port 8000

# Access at http://localhost:8000
```

### Testing Data Flow
Use sample CSV files from `sample-data/` directory to test the upload and processing pipeline.

## Key Implementation Details

### IBKR Column Mapping
The `COLUMN_MAP` in `processor.py` handles flexible CSV parsing. When adding support for new IBKR formats, update this mapping rather than hardcoding column names.

### Performance Calculation
- **Batting Average**: Percentage of profitable trades
- **Win/Loss Ratios**: Both standard and batting-average-adjusted ratios
- **Period Analysis**: Monthly, quarterly, yearly, and since-inception metrics

### Session Storage
Results are stored in memory (`results_store` dict) with UUID session keys. For production, consider Redis or database storage.

### File Upload Constraints
- 10MB file size limit (configurable via `MAX_FILE_SIZE` environment variable)
- CSV format validation
- Session timeout: 24 hours (configurable via `SESSION_TIMEOUT`)

## Configuration

Key environment variables in `docker-compose.yml`:
- `MAX_FILE_SIZE`: Upload limit in bytes
- `SESSION_TIMEOUT`: Session expiration in seconds
- `ENV`: development/production mode

## API Endpoints

- `POST /upload`: Process IBKR CSV files
- `GET /results/{session_id}`: View performance results
- `GET /api/results/{session_id}`: JSON API with period filtering
- `GET /export/{session_id}`: Export data (CSV, Excel, roundtrips)
- `DELETE /results/{session_id}`: Clean up session data

## Frontend Integration

Templates use Jinja2 with Chart.js for visualization. Key JavaScript functionality in `static/js/app.js` handles:
- File upload progress
- Chart rendering and configuration
- Period-based data filtering
- Export functionality

## Security Considerations

- Non-root container execution
- File type and size validation
- Session-based data isolation
- Automatic cleanup of expired data
- Health check monitoring