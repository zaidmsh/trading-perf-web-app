"""
Trading Performance Analyzer - Web Application
FastAPI backend with file upload and results display
"""

import os
import uuid
import json
from datetime import datetime
from typing import Dict, Any
import pandas as pd
from io import BytesIO

from fastapi import FastAPI, UploadFile, File, Request, HTTPException, BackgroundTasks
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import aiofiles

from core.processor import process_ibkr_csv
from core.calculator import calculate_performance


app = FastAPI(title="Trading Performance Analyzer", version="1.0.0")

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Templates
templates = Jinja2Templates(directory="templates")

# In-memory storage for results (in production, use Redis or database)
results_store: Dict[str, Dict[str, Any]] = {}


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
            roundtrips_df = process_ibkr_csv(csv_content)

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