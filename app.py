"""
Trading Performance Analyzer - Web Application
FastAPI backend with file upload and results display
Updated: 2025-01-10 - Fixed JSON serialization and position loading
"""

import os
import uuid
import json
import logging
from datetime import datetime
from typing import Dict, Any, Optional, List
import pandas as pd
from io import BytesIO

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

from fastapi import FastAPI, UploadFile, File, Request, HTTPException, BackgroundTasks, Form
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import aiofiles

from core.processor import process_ibkr_csv
from core.calculator import calculate_performance
from core.data_manager import DataManager


app = FastAPI(title="Trading Performance Analyzer", version="1.0.0")

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Templates
templates = Jinja2Templates(directory="templates")

# In-memory storage for results (in production, use Redis or database)
results_store: Dict[str, Dict[str, Any]] = {}

# Initialize data manager
data_manager = DataManager()


def get_average_gain_for_period(performance_data: Dict[str, Any], time_period: str) -> float:
    """
    Extract average gain from specified time period
    
    Args:
        performance_data: Performance data dictionary with monthly, quarterly, yearly, since_inception
        time_period: Period identifier (e.g., "since_inception", "2025", "2025-Q1", "2025-01")
        
    Returns:
        Average gain percentage for the specified period, or 0 if not found
    """
    if not performance_data or not time_period:
        return 0.0
    
    time_period = time_period.strip()
    
    # Since inception (default)
    if time_period.lower() == "since_inception":
        since_inception = performance_data.get("since_inception", {})
        return since_inception.get("Avg Gain", 0.0)
    
    # Yearly data (e.g., "2025")
    if time_period.isdigit() and len(time_period) == 4:
        yearly_data = performance_data.get("yearly", [])
        for year_record in yearly_data:
            if str(year_record.get("Date", "")).startswith(time_period):
                return year_record.get("Avg Gain", 0.0)
        return 0.0
    
    # Quarterly data (e.g., "2025-Q1")
    if "-Q" in time_period.upper():
        quarterly_data = performance_data.get("quarterly", [])
        for quarter_record in quarterly_data:
            quarter_date = str(quarter_record.get("Date", ""))
            # Match format like "2025-Q1" or "Q1 2025"
            if time_period.upper() in quarter_date.upper() or quarter_date.upper().startswith(time_period.upper()):
                return quarter_record.get("Avg Gain", 0.0)
        return 0.0
    
    # Monthly data (e.g., "2025-01" or "2025-01-01")
    if "-" in time_period and len(time_period) >= 7:
        monthly_data = performance_data.get("monthly", [])
        target_year_month = time_period[:7]  # Extract "2025-01" part
        
        for month_record in monthly_data:
            month_date = str(month_record.get("Date", ""))
            if month_date.startswith(target_year_month):
                return month_record.get("Avg Gain", 0.0)
        return 0.0
    
    # If no match found, return 0
    return 0.0


def get_available_time_periods(performance_data: Dict[str, Any]) -> List[Dict[str, str]]:
    """
    Get list of available time periods from performance data
    
    Returns:
        List of dictionaries with 'value' and 'label' keys for dropdown options
    """
    periods = []
    
    # Always include since inception
    since_inception = performance_data.get("since_inception", {})
    if since_inception:
        avg_gain = since_inception.get("Avg Gain", 0.0)
        periods.append({
            "value": "since_inception",
            "label": f"Since Inception (Avg: {avg_gain:.2f}%)"
        })
    
    # Add yearly periods
    yearly_data = performance_data.get("yearly", [])
    for year_record in yearly_data:
        date_str = str(year_record.get("Date", ""))
        avg_gain = year_record.get("Avg Gain", 0.0)
        if date_str and avg_gain != 0:
            year = date_str[:4]
            periods.append({
                "value": year,
                "label": f"{year} (Avg: {avg_gain:.2f}%)"
            })
    
    # Add quarterly periods
    quarterly_data = performance_data.get("quarterly", [])
    for quarter_record in quarterly_data:
        date_str = str(quarter_record.get("Date", ""))
        avg_gain = quarter_record.get("Avg Gain", 0.0)
        if date_str and avg_gain != 0:
            # Extract quarter info (format might be "2025-Q1" or "Q1 2025")
            if "Q" in date_str.upper():
                periods.append({
                    "value": date_str,
                    "label": f"{date_str} (Avg: {avg_gain:.2f}%)"
                })
    
    # Add monthly periods (last 12 months to avoid clutter)
    monthly_data = performance_data.get("monthly", [])
    # Sort by date and take last 12 months
    sorted_monthly = sorted(monthly_data, key=lambda x: str(x.get("Date", "")), reverse=True)[:12]
    
    for month_record in sorted_monthly:
        date_str = str(month_record.get("Date", ""))
        avg_gain = month_record.get("Avg Gain", 0.0)
        if date_str and avg_gain != 0:
            # Extract year-month (e.g., "2025-01")
            if len(date_str) >= 7:
                year_month = date_str[:7]
                # Format as readable month name
                try:
                    from datetime import datetime
                    dt = datetime.strptime(year_month + "-01", "%Y-%m-%d")
                    month_name = dt.strftime("%B %Y")
                    periods.append({
                        "value": year_month,
                        "label": f"{month_name} (Avg: {avg_gain:.2f}%)"
                    })
                except:
                    # Fallback to raw format
                    periods.append({
                        "value": year_month,
                        "label": f"{year_month} (Avg: {avg_gain:.2f}%)"
                    })
    
    return periods


def calculate_individual_stop_loss_percentage(position: Dict[str, Any], risk_ratio: str, average_gain: float) -> float:
    """
    Calculate individual stop loss percentage for a position accounting for partial sales
    
    Args:
        position: Position dictionary with entry price, quantity, realized P&L, etc.
        risk_ratio: Risk ratio string (e.g., "2:1", "3:1")
        average_gain: Average gain percentage for the period
        
    Returns:
        Stop loss percentage for this specific position
    """
    if average_gain <= 0:
        raise ValueError("Average gain must be positive")
    
    # Parse risk ratio
    try:
        risk_part, reward_part = risk_ratio.split(":")
        risk_multiple = float(risk_part)
        # reward_multiple = float(reward_part)  # Not used in current calculation
    except (ValueError, AttributeError):
        raise ValueError(f"Invalid risk ratio format: {risk_ratio}")
    
    if risk_multiple <= 0:
        raise ValueError("Risk multiple must be positive")
    
    # Base stop loss calculation: For a 3:1 ratio, risk 1 unit to gain 3 units
    # So stop loss = average_gain / risk_ratio
    base_stop_loss_percentage = average_gain / risk_multiple
    
    # Get position details
    total_cost_basis = position.get("total_cost_basis", position.get("cost_basis", 0))
    current_cost_basis = position.get("cost_basis", 0)
    realized_pnl = position.get("realized_pnl", 0.0)
    
    # If no partial sales or profits realized, use base calculation
    if realized_pnl <= 0 or total_cost_basis <= 0 or current_cost_basis <= 0:
        return base_stop_loss_percentage
    
    # Calculate risk adjustment factor based on realized profits
    # If we've made profits from partial sales, we can afford more risk on remaining shares
    realized_profit_ratio = max(0, realized_pnl) / total_cost_basis
    
    # Adjust stop loss: reduce risk by the proportion of profits already realized
    # This allows for a more aggressive stop loss since we've already secured some gains
    risk_reduction_factor = min(realized_profit_ratio, 0.5)  # Cap at 50% reduction
    adjusted_stop_loss_percentage = base_stop_loss_percentage * (1 + risk_reduction_factor)
    
    # Ensure stop loss doesn't exceed reasonable limits (max 15%)
    adjusted_stop_loss_percentage = min(adjusted_stop_loss_percentage, 15.0)
    
    return round(adjusted_stop_loss_percentage, 2)


def ensure_json_serializable(data):
    """
    Recursively convert data to ensure JSON serialization compatibility
    
    Handles:
    - numpy bool_ to Python bool
    - numpy numeric types to Python numeric types
    - nested dictionaries and lists
    """
    try:
        import numpy as np
        has_numpy = True
    except ImportError:
        has_numpy = False
    
    if isinstance(data, dict):
        return {key: ensure_json_serializable(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [ensure_json_serializable(item) for item in data]
    elif data is None:
        return None
    elif isinstance(data, (int, float, str, bool)):
        return data
    elif has_numpy:
        if isinstance(data, np.bool_):
            return bool(data)
        elif isinstance(data, (np.integer, np.floating)):
            return float(data) if isinstance(data, np.floating) else int(data)
        elif hasattr(data, 'item'):  # For numpy scalar types
            return data.item()
    # Handle pandas types without importing pandas
    elif type(data).__module__ == 'numpy' or 'numpy' in str(type(data)):
        # Convert any numpy-like type to Python native
        if hasattr(data, 'item'):
            return data.item()
        else:
            return float(data) if 'float' in str(type(data)) else int(data)
    # Handle Decimal types
    elif hasattr(data, '__float__'):
        return float(data)
    elif hasattr(data, '__int__'):
        return int(data)
    else:
        # Return as-is for unhandled types
        return data


# Pydantic models for request bodies
class IBKRRequest(BaseModel):
    token: str
    query_id: str
    start_year: Optional[int] = None




class ProcessRequest(BaseModel):
    source: str  # 'csv', 'ibkr', or 'hybrid'
    token: str = None
    query_id: str = None


class HybridRequest(BaseModel):
    token: str
    query_id: str


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Home page with file upload"""
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/upload")
async def upload_file(files: list[UploadFile] = File(...)):
    """Upload and process multiple IBKR CSV files"""
    # Validate files
    if not files:
        raise HTTPException(status_code=400, detail="Please upload at least one CSV file")

    for file in files:
        if not file.filename.endswith('.csv'):
            raise HTTPException(status_code=400, detail=f"File {file.filename} is not a CSV file")

        if file.size > 10 * 1024 * 1024:  # 10MB limit per file
            raise HTTPException(status_code=400, detail=f"File {file.filename} is too large (max 10MB)")

    try:
        all_roundtrips = []
        filenames = []

        # Process each CSV file
        for file in files:
            # Read file content
            content = await file.read()
            csv_content = content.decode('utf-8')

            # Process the CSV
            roundtrips_df, _, _ = process_ibkr_csv(csv_content)

            if not roundtrips_df.empty:
                all_roundtrips.append(roundtrips_df)
                filenames.append(file.filename)

        if not all_roundtrips:
            raise HTTPException(status_code=400, detail="No valid trades found in any CSV files")

        # Merge all roundtrips from multiple files
        merged_roundtrips_df = pd.concat(all_roundtrips, ignore_index=True)

        # Sort by entry date to ensure chronological order
        if 'entry_date' in merged_roundtrips_df.columns:
            merged_roundtrips_df = merged_roundtrips_df.sort_values('entry_date')
            merged_roundtrips_df = merged_roundtrips_df.reset_index(drop=True)

        # Calculate performance metrics on merged data
        performance_data = calculate_performance(merged_roundtrips_df)

        # Generate unique session ID
        session_id = str(uuid.uuid4())

        # Convert timestamps to strings for JSON serialization
        roundtrips_serializable = merged_roundtrips_df.copy()
        for col in roundtrips_serializable.columns:
            if roundtrips_serializable[col].dtype.name.startswith('datetime'):
                roundtrips_serializable[col] = roundtrips_serializable[col].dt.strftime('%Y-%m-%d')

        # Store results
        results_store[session_id] = {
            "filename": ", ".join(filenames),
            "files_count": len(filenames),
            "upload_time": datetime.now().isoformat(),
            "roundtrips": roundtrips_serializable.to_dict('records'),
            "performance": performance_data
        }

        return JSONResponse({
            "success": True,
            "session_id": session_id,
            "message": f"Processed {len(filenames)} file(s) with {len(merged_roundtrips_df)} total roundtrips"
        })

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing files: {str(e)}")


@app.get("/results/{session_id}", response_class=HTMLResponse)
async def view_results(request: Request, session_id: str):
    """Display results page"""
    if session_id not in results_store:
        raise HTTPException(status_code=404, detail="Results not found")

    data = results_store[session_id]
    return templates.TemplateResponse("results.html", {
        "request": request,
        "session_id": session_id,
        "data": data
    })


@app.get("/api/results/{session_id}")
async def get_results_api(session_id: str, period: str = "all"):
    """Get results data as JSON"""
    if session_id not in results_store:
        raise HTTPException(status_code=404, detail="Results not found")

    data = results_store[session_id]
    performance = data["performance"]

    if period == "monthly":
        return {"data": performance["monthly"]}
    elif period == "quarterly":
        return {"data": performance["quarterly"]}
    elif period == "yearly":
        return {"data": performance["yearly"]}
    elif period == "since_inception":
        return {"data": [performance["since_inception"]] if performance["since_inception"] else []}
    else:  # all
        return {
            "summary": performance["summary"],
            "since_inception": performance["since_inception"],
            "yearly": performance["yearly"],
            "quarterly": performance["quarterly"],
            "monthly": performance["monthly"]
        }


@app.get("/export/{session_id}")
async def export_results(session_id: str, format: str = "csv", period: str = "all"):
    """Export results in various formats"""
    if session_id not in results_store:
        raise HTTPException(status_code=404, detail="Results not found")

    data = results_store[session_id]

    if format == "roundtrips":
        # Export roundtrips CSV
        df = pd.DataFrame(data["roundtrips"])
        output = BytesIO()
        df.to_csv(output, index=False)
        output.seek(0)

        return StreamingResponse(
            BytesIO(output.read()),
            media_type="text/csv",
            headers={"Content-Disposition": f"attachment; filename=roundtrips_{session_id[:8]}.csv"}
        )

    elif format == "excel":
        # Export performance to Excel
        performance = data["performance"]
        output = BytesIO()

        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            if performance["since_inception"]:
                pd.DataFrame([performance["since_inception"]]).to_excel(writer, sheet_name='Since Inception', index=False)
            if performance["yearly"]:
                pd.DataFrame(performance["yearly"]).to_excel(writer, sheet_name='Yearly', index=False)
            if performance["quarterly"]:
                pd.DataFrame(performance["quarterly"]).to_excel(writer, sheet_name='Quarterly', index=False)
            if performance["monthly"]:
                pd.DataFrame(performance["monthly"]).to_excel(writer, sheet_name='Monthly', index=False)

        output.seek(0)

        return StreamingResponse(
            BytesIO(output.read()),
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers={"Content-Disposition": f"attachment; filename=performance_{session_id[:8]}.xlsx"}
        )

    else:  # CSV format
        performance = data["performance"]

        if period == "monthly" and performance["monthly"]:
            df = pd.DataFrame(performance["monthly"])
        elif period == "quarterly" and performance["quarterly"]:
            df = pd.DataFrame(performance["quarterly"])
        elif period == "yearly" and performance["yearly"]:
            df = pd.DataFrame(performance["yearly"])
        elif period == "since_inception" and performance["since_inception"]:
            df = pd.DataFrame([performance["since_inception"]])
        else:
            # All periods combined
            all_data = []
            if performance["since_inception"]:
                all_data.append({**performance["since_inception"], "Period": "Since Inception"})
            for item in performance["yearly"]:
                all_data.append({**item, "Period": "Yearly"})
            for item in performance["quarterly"]:
                all_data.append({**item, "Period": "Quarterly"})
            for item in performance["monthly"]:
                all_data.append({**item, "Period": "Monthly"})
            df = pd.DataFrame(all_data)

        if df.empty:
            raise HTTPException(status_code=404, detail="No data found for the specified period")

        # Remove internal sorting columns
        df = df.drop(columns=[col for col in df.columns if col.startswith('_')], errors='ignore')

        output = BytesIO()
        df.to_csv(output, index=False)
        output.seek(0)

        return StreamingResponse(
            BytesIO(output.read()),
            media_type="text/csv",
            headers={"Content-Disposition": f"attachment; filename=performance_{period}_{session_id[:8]}.csv"}
        )


@app.delete("/results/{session_id}")
async def delete_results(session_id: str):
    """Delete stored results"""
    if session_id in results_store:
        del results_store[session_id]
        return {"message": "Results deleted successfully"}
    else:
        raise HTTPException(status_code=404, detail="Results not found")


@app.post("/api/fetch-ibkr")
async def fetch_ibkr_data(request: IBKRRequest):
    """Fetch trades from IBKR FlexQuery API"""
    try:
        # Use data manager to fetch and process trades
        # If start_year is provided, fetch historical data
        if request.start_year:
            result = await data_manager.get_trades(
                source="ibkr",
                token=request.token,
                query_id=request.query_id,
                historical=True,
                start_year=request.start_year
            )
            data_source = f"IBKR Historical ({request.start_year}-present)"
        else:
            result = await data_manager.get_trades(
                source="ibkr",
                token=request.token,
                query_id=request.query_id
            )
            data_source = f"IBKR FlexQuery {request.query_id}"

        # Generate unique session ID
        session_id = str(uuid.uuid4())

        # Convert timestamps to strings for JSON serialization
        roundtrips_serializable = result["roundtrips"].copy()
        for col in roundtrips_serializable.columns:
            if roundtrips_serializable[col].dtype.name.startswith('datetime'):
                roundtrips_serializable[col] = roundtrips_serializable[col].dt.strftime('%Y-%m-%d')

        # Store results in the same format as CSV upload
        results_store[session_id] = {
            "filename": data_source,
            "files_count": 1,
            "upload_time": datetime.now().isoformat(),
            "roundtrips": roundtrips_serializable.to_dict('records'),
            "performance": result["performance"]
        }

        return JSONResponse({
            "success": True,
            "session_id": session_id,
            "message": f"Fetched {len(result['roundtrips'])} roundtrips from IBKR"
        })

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/test-ibkr")
async def test_ibkr_connection(request: IBKRRequest):
    """Test IBKR FlexQuery connection"""
    try:
        result = await data_manager.test_ibkr_connection(request.token, request.query_id)
        return JSONResponse(result)
    except Exception as e:
        return JSONResponse({
            "success": False,
            "message": f"Connection test failed: {str(e)}"
        })


@app.post("/api/hybrid-upload")
async def hybrid_upload(token: str, query_id: str, files: list[UploadFile] = File(...)):
    """Combine CSV files (historical) with IBKR current year data"""
    try:
        # Validate inputs
        if not files:
            raise HTTPException(status_code=400, detail="Please upload at least one CSV file for historical data")

        if not token or not query_id:
            raise HTTPException(status_code=400, detail="IBKR token and Query ID are required")

        # Use data manager to process hybrid data
        result = await data_manager.get_trades(
            source="hybrid",
            files=files,
            token=token,
            query_id=query_id
        )

        # Generate unique session ID
        session_id = str(uuid.uuid4())

        # Convert timestamps to strings for JSON serialization
        roundtrips_serializable = result["roundtrips"].copy()
        for col in roundtrips_serializable.columns:
            if roundtrips_serializable[col].dtype.name.startswith('datetime'):
                roundtrips_serializable[col] = roundtrips_serializable[col].dt.strftime('%Y-%m-%d')

        # Store results
        results_store[session_id] = {
            "filename": f"Hybrid: {len(files)} CSV file(s) + IBKR current year",
            "files_count": len(files) + 1,  # CSV files + IBKR
            "upload_time": datetime.now().isoformat(),
            "roundtrips": roundtrips_serializable.to_dict('records'),
            "performance": result["performance"]
        }

        return JSONResponse({
            "success": True,
            "session_id": session_id,
            "message": f"Combined {len(files)} CSV file(s) with IBKR data: {len(result['roundtrips'])} total roundtrips"
        })

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))



@app.post("/api/hybrid-analyze")
async def hybrid_analyze(
    files: list[UploadFile] = File(None),
    token: str = Form(None),
    query_id: str = Form(None),
    start_year: int = Form(None)
):
    """Unified endpoint that automatically detects and processes available inputs"""
    try:
        # Check what inputs are provided
        has_files = files and len(files) > 0 and files[0].filename
        has_ibkr_credentials = token and token.strip() and query_id and query_id.strip()

        # Validation: At least one input method required
        if not has_files and not has_ibkr_credentials:
            raise HTTPException(
                status_code=400, 
                detail="Please provide either CSV files or IBKR credentials (or both)."
            )

        # Determine processing mode and execute
        if has_files and has_ibkr_credentials:
            # Hybrid mode: Combine CSV files with IBKR data
            result = await data_manager.get_trades(
                source="hybrid",
                files=files,
                token=token,
                query_id=query_id
            )
            source_info = f"Hybrid: {len(files)} CSV file(s) + IBKR current year"
            files_count = len(files) + 1

        elif has_files:
            # CSV only mode
            result = await data_manager.get_trades(source="csv", files=files)
            source_info = f"{len(files)} CSV file(s)"
            files_count = len(files)

        else:
            # IBKR only mode
            if start_year:
                result = await data_manager.get_trades(
                    source="ibkr",
                    token=token,
                    query_id=query_id,
                    historical=True,
                    start_year=start_year
                )
                source_info = f"IBKR Historical ({start_year}-present)"
            else:
                result = await data_manager.get_trades(
                    source="ibkr",
                    token=token,
                    query_id=query_id
                )
                source_info = f"IBKR FlexQuery {query_id}"
            files_count = 1

        # Generate unique session ID
        session_id = str(uuid.uuid4())

        # Convert timestamps to strings for JSON serialization
        roundtrips_serializable = result["roundtrips"].copy()
        for col in roundtrips_serializable.columns:
            if roundtrips_serializable[col].dtype.name.startswith('datetime'):
                roundtrips_serializable[col] = roundtrips_serializable[col].dt.strftime('%Y-%m-%d')

        # Store results
        results_store[session_id] = {
            "filename": source_info,
            "files_count": files_count,
            "upload_time": datetime.now().isoformat(),
            "roundtrips": roundtrips_serializable.to_dict('records'),
            "performance": result["performance"],
            "open_positions": result.get("open_positions", {"positions": [], "summary": {}})
        }

        return JSONResponse({
            "success": True,
            "session_id": session_id,
            "message": f"Processed {len(result['roundtrips'])} roundtrips from {source_info.lower()}"
        })

    except HTTPException as he:
        # Re-raise HTTPExceptions as-is (they already have proper status codes and detail)
        logger.error(f"Hybrid analyze HTTPException: {he.detail} (status: {he.status_code})")
        raise he
    except Exception as e:
        error_message = str(e)
        logger.error(f"Hybrid analyze error: '{error_message}' (type: {type(e).__name__})")
        
        # Handle empty error messages
        if not error_message or error_message.strip() == "":
            error_message = f"Unknown error occurred during analysis ({type(e).__name__})"
        
        # Provide more specific error messages based on error content
        if "Authentication failed" in error_message or "Invalid request or unable to validate request" in error_message:
            error_message = "IBKR authentication failed. Please check your FlexQuery token and Query ID."
        elif "No trades found" in error_message:
            error_message = "No trades found in IBKR FlexQuery response. Please check your query configuration."
        elif "Network error" in error_message:
            error_message = "Network error connecting to IBKR. Please try again later."
        elif "Failed to fetch IBKR data" in error_message:
            # Extract the underlying error message
            if ":" in error_message:
                underlying_error = error_message.split(":", 1)[1].strip()
                error_message = f"IBKR fetch failed: {underlying_error}"
        
        raise HTTPException(status_code=400, detail=error_message)


@app.get("/api/open-positions/{session_id}")
async def get_open_positions(session_id: str):
    """Get open positions with current prices"""
    if session_id not in results_store:
        raise HTTPException(status_code=404, detail="Session data not found. Please re-upload your data - the session may have expired or the server was restarted.")

    try:
        # Get stored open positions data
        stored_data = results_store[session_id]
        open_positions_data = stored_data.get("open_positions", {"positions": [], "summary": {}})
        
        logger.info(f"Getting open positions for session {session_id}: {len(open_positions_data.get('positions', []))} positions found")
        
        if not open_positions_data["positions"]:
            return JSONResponse({
                "positions": [],
                "summary": {
                    "total_positions": 0,
                    "total_cost_basis": 0.0,
                    "total_market_value": 0.0,
                    "total_unrealized_pnl": 0.0,
                    "overall_pnl_pct": 0.0,
                    "positions_with_prices": 0,
                    "long_positions": 0,
                    "short_positions": 0
                }
            })

        # Refresh prices for current positions
        from core.open_positions import calculate_position_pnl, calculate_portfolio_summary
        from core.stop_loss_manager import stop_loss_manager
        positions = open_positions_data["positions"].copy()
        
        # Calculate current P&L with fresh prices
        updated_positions = await calculate_position_pnl(positions)
        
        # Add stop loss data and R-Multiple calculations
        stop_losses = stop_loss_manager.get_all_stop_losses(session_id)
        
        for position in updated_positions:
            symbol = position["symbol"]
            stop_loss = stop_losses.get(symbol)
            
            if stop_loss:
                # Add stop loss information
                position["stop_loss"] = {
                    "price": stop_loss.stop_loss_price,
                    "type": stop_loss.stop_loss_type,
                    "value": stop_loss.stop_loss_value,
                    "risk_amount": stop_loss.risk_amount,
                    "risk_percentage": stop_loss.risk_percentage,
                    "is_triggered": stop_loss.is_triggered
                }
                
                # Calculate R-Multiple
                r_multiple = stop_loss_manager.calculate_r_multiple(position, stop_loss)
                position["r_multiple"] = r_multiple
                
                # Calculate free ride recommendations with precise stop loss data
                free_ride_recommendations = stop_loss_manager.calculate_free_ride_with_stop_loss(position, stop_loss)
                position["free_ride_recommendations"] = free_ride_recommendations
                
                # Find the current actionable free ride recommendation
                current_recommendation = None
                for rec in free_ride_recommendations:
                    if rec["target_reached"]:
                        current_recommendation = rec
                        break
                
                # If no target reached, show the 1R target
                if not current_recommendation and free_ride_recommendations:
                    current_recommendation = free_ride_recommendations[0]  # 1R target
                
                position["current_free_ride"] = current_recommendation
                
                # Check for new triggers
                triggered = stop_loss_manager.check_stop_triggers(session_id, [position])
                if triggered:
                    position["stop_loss"]["is_triggered"] = True
            else:
                position["stop_loss"] = None
                position["r_multiple"] = None
                
                # Calculate free ride recommendations without stop loss (using estimated risk)
                free_ride_recommendations = stop_loss_manager.calculate_free_ride_shares(position)
                position["free_ride_recommendations"] = free_ride_recommendations
                
                # Find current recommendation based on estimated targets
                current_recommendation = None
                for rec in free_ride_recommendations:
                    if rec["target_reached"]:
                        current_recommendation = rec
                        break
                
                if not current_recommendation and free_ride_recommendations:
                    current_recommendation = free_ride_recommendations[0]  # 1R target
                
                position["current_free_ride"] = current_recommendation
        
        # Calculate enhanced portfolio summary with risk metrics
        updated_summary = calculate_portfolio_summary(updated_positions)
        
        # Add risk management metrics to summary
        total_risk = sum(pos.get("stop_loss", {}).get("risk_amount", 0) for pos in updated_positions if pos.get("stop_loss"))
        active_stops = sum(1 for pos in updated_positions if pos.get("stop_loss") and not pos["stop_loss"].get("is_triggered", False))
        triggered_stops = sum(1 for pos in updated_positions if pos.get("stop_loss") and pos["stop_loss"].get("is_triggered", False))
        
        # Calculate average stop loss percentage
        stop_loss_percentages = [pos.get("stop_loss", {}).get("risk_percentage", 0) for pos in updated_positions if pos.get("stop_loss")]
        avg_stop_loss_percentage = sum(stop_loss_percentages) / len(stop_loss_percentages) if stop_loss_percentages else 0
        
        updated_summary.update({
            "total_risk_exposure": total_risk,
            "active_stops_count": active_stops,
            "triggered_stops_count": triggered_stops,
            "average_stop_loss_percentage": avg_stop_loss_percentage
        })
        
        # Ensure all data is JSON serializable
        try:
            response_data = ensure_json_serializable({
                "positions": updated_positions,
                "summary": updated_summary
            })
            # Successfully serialized response data
            return JSONResponse(response_data)
        except Exception as serialize_error:
            logger.error(f"JSON serialization error: {serialize_error}")
            logger.error(f"Problem with position data: {type(serialize_error)}")
            raise HTTPException(status_code=500, detail=f"Data serialization error: {str(serialize_error)}")

    except Exception as e:
        logger.error(f"Error getting open positions for {session_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Error fetching open positions: {str(e)}")


@app.post("/api/refresh-prices/{session_id}")
async def refresh_prices(session_id: str):
    """Refresh prices for open positions"""
    if session_id not in results_store:
        raise HTTPException(status_code=404, detail="Results not found")

    try:
        # Clear price cache to force fresh data
        from core.price_service import price_service
        price_service.clear_cache()
        
        # Get fresh positions data
        return await get_open_positions(session_id)

    except Exception as e:
        logger.error(f"Error refreshing prices for {session_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Error refreshing prices: {str(e)}")


@app.post("/api/set-stop-loss/{session_id}")
async def set_stop_loss(session_id: str, request: dict):
    """Set stop loss for a position"""
    if session_id not in results_store:
        raise HTTPException(status_code=404, detail="Results not found")
    
    try:
        from core.stop_loss_manager import stop_loss_manager
        
        # Extract request data
        symbol = request.get("symbol")
        stop_type = request.get("stop_type")  # "amount" or "percentage"
        stop_value = request.get("stop_value")
        
        if not all([symbol, stop_type, stop_value is not None]):
            raise HTTPException(status_code=400, detail="Missing required fields: symbol, stop_type, stop_value")
        
        # Get position data
        stored_data = results_store[session_id]
        open_positions_data = stored_data.get("open_positions", {"positions": []})
        
        # Find the position
        position = None
        for pos in open_positions_data["positions"]:
            if pos["symbol"] == symbol:
                position = pos
                break
        
        if not position:
            raise HTTPException(status_code=404, detail=f"Position not found for symbol: {symbol}")
        
        # Set stop loss
        stop_loss = stop_loss_manager.set_stop_loss(session_id, position, stop_type, float(stop_value))
        
        return JSONResponse({
            "success": True,
            "message": f"Stop loss set for {symbol}",
            "stop_loss": stop_loss.to_dict()
        })
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error setting stop loss for {session_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Error setting stop loss: {str(e)}")


@app.delete("/api/remove-stop-loss/{session_id}/{symbol}")
async def remove_stop_loss(session_id: str, symbol: str):
    """Remove stop loss for a position"""
    if session_id not in results_store:
        raise HTTPException(status_code=404, detail="Results not found")
    
    try:
        from core.stop_loss_manager import stop_loss_manager
        
        success = stop_loss_manager.remove_stop_loss(session_id, symbol)
        
        if success:
            return JSONResponse({
                "success": True,
                "message": f"Stop loss removed for {symbol}"
            })
        else:
            raise HTTPException(status_code=404, detail=f"No stop loss found for {symbol}")
            
    except Exception as e:
        logger.error(f"Error removing stop loss for {session_id}/{symbol}: {e}")
        raise HTTPException(status_code=500, detail=f"Error removing stop loss: {str(e)}")


@app.get("/api/stop-losses/{session_id}")
async def get_stop_losses(session_id: str):
    """Get all stop losses for a session"""
    if session_id not in results_store:
        raise HTTPException(status_code=404, detail="Results not found")
    
    try:
        from core.stop_loss_manager import stop_loss_manager
        
        stop_losses = stop_loss_manager.get_all_stop_losses(session_id)
        
        # Convert to serializable format
        stop_losses_dict = {symbol: stop_loss.to_dict() for symbol, stop_loss in stop_losses.items()}
        
        return JSONResponse({
            "stop_losses": stop_losses_dict
        })
        
    except Exception as e:
        logger.error(f"Error getting stop losses for {session_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting stop losses: {str(e)}")


@app.post("/api/apply-risk-ratio/{session_id}")
async def apply_risk_ratio(session_id: str, request: dict):
    """Apply risk ratio to all open positions based on average gain"""
    if session_id not in results_store:
        raise HTTPException(status_code=404, detail="Results not found")
    
    try:
        from core.stop_loss_manager import stop_loss_manager
        
        # Extract request data
        risk_ratio = request.get("risk_ratio")  # e.g., "2:1", "3:1", "4:1", "none"
        time_period = request.get("time_period", "since_inception")  # Default to since_inception
        
        if not risk_ratio:
            raise HTTPException(status_code=400, detail="Missing risk_ratio parameter")
        
        if risk_ratio == "none":
            # Clear all stop losses
            stop_loss_manager.clear_session(session_id)
            return JSONResponse({
                "success": True,
                "message": "All stop losses cleared. Set individual stop losses as needed."
            })
        
        # Parse risk ratio (e.g., "2:1" means stop loss should be 1/2 of average gain)
        if ":" not in risk_ratio:
            raise HTTPException(status_code=400, detail="Invalid risk ratio format. Use format like '2:1'")
        
        ratio_parts = risk_ratio.split(":")
        if len(ratio_parts) != 2:
            raise HTTPException(status_code=400, detail="Invalid risk ratio format. Use format like '2:1'")
        
        try:
            risk_multiple = float(ratio_parts[0])
            # reward_multiple = float(ratio_parts[1])  # Not used in current calculation
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid risk ratio values. Must be numbers.")
        
        # Get performance data to calculate average gain for specified period
        stored_data = results_store[session_id]
        performance_data = stored_data.get("performance", {})
        average_gain = get_average_gain_for_period(performance_data, time_period)
        
        if average_gain <= 0:
            raise HTTPException(status_code=400, detail=f"No positive average gain found for period '{time_period}'. Cannot calculate risk ratio.")
        
        # Calculate stop loss percentage based on risk ratio
        # If average gain is 6% and ratio is 2:1, stop loss should be 3%
        # For a 2:1 ratio, you risk 1 unit to gain 2 units
        stop_loss_percentage = average_gain / risk_multiple
        
        # Get open positions
        open_positions_data = stored_data.get("open_positions", {"positions": []})
        positions = open_positions_data["positions"]
        
        if not positions:
            raise HTTPException(status_code=400, detail="No open positions found")
        
        # Apply stop loss to all positions based on individual risk calculations
        applied_count = 0
        errors = []
        
        for position in positions:
            try:
                # Calculate individual stop loss based on position-specific risk
                individual_stop_loss_percentage = calculate_individual_stop_loss_percentage(
                    position, risk_ratio, average_gain
                )
                
                stop_loss = stop_loss_manager.set_stop_loss(
                    session_id, 
                    position, 
                    "percentage", 
                    individual_stop_loss_percentage
                )
                applied_count += 1
                logger.info(f"Set stop loss for {position['symbol']}: {individual_stop_loss_percentage:.2f}% = ${stop_loss.stop_loss_price:.4f}, Risk: ${stop_loss.risk_amount:.2f}")
            except Exception as e:
                errors.append(f"{position['symbol']}: {str(e)}")
        
        message = f"Applied {risk_ratio} risk ratio to {applied_count} positions. "
        message += f"Stop loss set to {stop_loss_percentage:.2f}% (based on {average_gain:.2f}% avg gain from {time_period})."
        
        if errors:
            message += f" Errors: {'; '.join(errors)}"
        
        return JSONResponse({
            "success": True,
            "message": message,
            "applied_count": applied_count,
            "stop_loss_percentage": stop_loss_percentage,
            "average_gain": average_gain,
            "errors": errors
        })
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error applying risk ratio for {session_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Error applying risk ratio: {str(e)}")


@app.get("/api/time-periods/{session_id}")
async def get_time_periods(session_id: str):
    """Get available time periods for risk ratio calculations"""
    if session_id not in results_store:
        raise HTTPException(status_code=404, detail="Results not found")
    
    try:
        stored_data = results_store[session_id]
        performance_data = stored_data.get("performance", {})
        periods = get_available_time_periods(performance_data)
        
        return JSONResponse({
            "periods": periods
        })
        
    except Exception as e:
        logger.error(f"Error getting time periods for {session_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting time periods: {str(e)}")


# Background task to clean up old results (run periodically in production)
def cleanup_old_results():
    """Clean up results older than 24 hours"""
    cutoff_time = datetime.now().timestamp() - 24 * 60 * 60  # 24 hours ago

    to_delete = []
    for session_id, data in results_store.items():
        upload_time = datetime.fromisoformat(data["upload_time"])
        if upload_time.timestamp() < cutoff_time:
            to_delete.append(session_id)

    for session_id in to_delete:
        del results_store[session_id]


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)