"""
FastAPI Application for ASV-INSPECT
RESTful API for assembly inspection service
"""

import os
import sys
from pathlib import Path
from typing import Optional
import shutil
from datetime import datetime
import io

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
import uvicorn

from inspect_assembly import AssemblyInspector
from utils import generate_timestamp


# Define models
class InspectionResponse(BaseModel):
    """Response model for inspection results"""
    status: str = Field(..., description="PASS or FAIL")
    compliance_score: float = Field(..., description="Compliance score (0-1)")
    expected_count: int = Field(..., description="Number of expected components")
    detected_count: int = Field(..., description="Number of detected components")
    missing_count: int = Field(..., description="Number of missing components")
    missing_components: list = Field(..., description="Details of missing components")
    timestamp: str = Field(..., description="Inspection timestamp")
    image_name: str = Field(..., description="Input image filename")
    annotated_image_url: Optional[str] = Field(None, description="URL to annotated image")
    report_url: Optional[str] = Field(None, description="URL to JSON report")


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    message: str
    detector_loaded: bool
    golden_model_loaded: bool


# Initialize FastAPI app
app = FastAPI(
    title="ASV-INSPECT API",
    description="Automated Statistical Verification for Assembly Inspection",
    version="1.0.0",
    docs_url="/",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global inspector instance
inspector: Optional[AssemblyInspector] = None

# Configuration
DETECTOR_PATH = os.getenv("DETECTOR_PATH", "../models/detector/train/weights/best.pt")
GOLDEN_MODEL_PATH = os.getenv("GOLDEN_MODEL_PATH", "../models/golden_model/golden_model.json")
UPLOAD_DIR = Path("../outputs/uploads")
OUTPUT_DIR = Path("../outputs")
BASE_TOLERANCE = float(os.getenv("BASE_TOLERANCE", "0.05"))
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.25"))

# Create directories
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


@app.on_event("startup")
async def startup_event():
    """Initialize inspector on startup"""
    global inspector
    
    try:
        print("Loading ASV-INSPECT model...")
        inspector = AssemblyInspector(
            detector_path=DETECTOR_PATH,
            golden_model_path=GOLDEN_MODEL_PATH,
            base_tolerance=BASE_TOLERANCE,
            confidence_threshold=CONFIDENCE_THRESHOLD
        )
        print("✓ Inspector loaded successfully")
    except Exception as e:
        print(f"✗ Failed to load inspector: {e}")
        print("API will start but inspection endpoints will not work")
        inspector = None


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint
    
    Returns system status and model availability
    """
    detector_loaded = False
    golden_model_loaded = False
    
    if inspector is not None:
        detector_loaded = True
        golden_model_loaded = True
    
    status = "healthy" if detector_loaded and golden_model_loaded else "degraded"
    message = "All systems operational" if status == "healthy" else "Models not loaded"
    
    return HealthResponse(
        status=status,
        message=message,
        detector_loaded=detector_loaded,
        golden_model_loaded=golden_model_loaded
    )


@app.post("/inspect", response_model=InspectionResponse)
async def inspect_assembly(
    file: UploadFile = File(..., description="Image file to inspect"),
    save_visualization: bool = Query(True, description="Save annotated image"),
    save_report: bool = Query(True, description="Save JSON report"),
    return_image: bool = Query(False, description="Return annotated image in response")
):
    """
    Inspect an assembly image for missing components
    
    - **file**: Image file (JPEG, PNG, BMP)
    - **save_visualization**: Whether to save annotated image
    - **save_report**: Whether to save JSON report
    - **return_image**: Whether to include image URL in response
    
    Returns inspection results with status, compliance score, and missing components
    """
    if inspector is None:
        raise HTTPException(
            status_code=503,
            detail="Inspector not available. Models not loaded."
        )
    
    # Validate file type
    if not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="File must be an image (JPEG, PNG, BMP)"
        )
    
    # Save uploaded file
    timestamp = generate_timestamp()
    file_extension = Path(file.filename).suffix
    temp_filename = f"upload_{timestamp}{file_extension}"
    temp_path = UPLOAD_DIR / temp_filename
    
    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Perform inspection
        report = inspector.inspect(
            str(temp_path),
            output_dir=str(OUTPUT_DIR),
            save_visualization=save_visualization,
            save_report=save_report
        )
        
        # Prepare response
        response_data = {
            "status": report["status"],
            "compliance_score": report["compliance_score"],
            "expected_count": report["expected_count"],
            "detected_count": report["detected_count"],
            "missing_count": report["missing_count"],
            "missing_components": report["missing_components"],
            "timestamp": report["timestamp"],
            "image_name": file.filename
        }
        
        # Add URLs if available
        if return_image and "output_image_path" in report:
            response_data["annotated_image_url"] = f"/outputs/images/{Path(report['output_image_path']).name}"
        
        if "report_path" in report:
            response_data["report_url"] = f"/outputs/reports/{Path(report['report_path']).name}"
        
        return InspectionResponse(**response_data)
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Inspection failed: {str(e)}"
        )
    finally:
        # Clean up uploaded file
        if temp_path.exists():
            temp_path.unlink()


@app.get("/outputs/images/{filename}")
async def get_output_image(filename: str):
    """
    Retrieve an annotated output image
    
    - **filename**: Name of the output image file
    """
    file_path = OUTPUT_DIR / "images" / filename
    
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Image not found")
    
    return FileResponse(
        path=file_path,
        media_type="image/jpeg",
        filename=filename
    )


@app.get("/outputs/reports/{filename}")
async def get_output_report(filename: str):
    """
    Retrieve an inspection report
    
    - **filename**: Name of the report JSON file
    """
    file_path = OUTPUT_DIR / "reports" / filename
    
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Report not found")
    
    return FileResponse(
        path=file_path,
        media_type="application/json",
        filename=filename
    )


@app.get("/model/info")
async def get_model_info():
    """
    Get information about loaded models
    
    Returns configuration and statistics
    """
    if inspector is None:
        raise HTTPException(
            status_code=503,
            detail="Inspector not available"
        )
    
    info = {
        "detector_path": inspector.detector_path,
        "golden_model_path": inspector.golden_model_path,
        "base_tolerance": inspector.base_tolerance,
        "confidence_threshold": inspector.confidence_threshold,
        "expected_components": len(inspector.golden_model["expected_components"]),
        "component_classes": list(set(
            comp["class_name"] 
            for comp in inspector.golden_model["expected_components"]
        )),
        "golden_model_metadata": inspector.golden_model.get("metadata", {})
    }
    
    return info


@app.post("/reload")
async def reload_models():
    """
    Reload detector and golden model
    
    Useful after training or updating models
    """
    global inspector
    
    try:
        inspector = AssemblyInspector(
            detector_path=DETECTOR_PATH,
            golden_model_path=GOLDEN_MODEL_PATH,
            base_tolerance=BASE_TOLERANCE,
            confidence_threshold=CONFIDENCE_THRESHOLD
        )
        return {"status": "success", "message": "Models reloaded successfully"}
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to reload models: {str(e)}"
        )


@app.get("/ui", response_class=HTMLResponse)
async def get_ui():
    """
    Serve the web UI
    """
    ui_path = Path(__file__).parent / "static" / "index.html"
    
    if not ui_path.exists():
        raise HTTPException(status_code=404, detail="UI not found")
    
    with open(ui_path, 'r') as f:
        return HTMLResponse(content=f.read())


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run ASV-INSPECT API server')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Host to bind')
    parser.add_argument('--port', type=int, default=8000, help='Port to bind')
    parser.add_argument('--reload', action='store_true', help='Enable auto-reload')
    
    args = parser.parse_args()
    
    print(f"""
    {'='*60}
    ASV-INSPECT API Server
    {'='*60}
    Starting server at http://{args.host}:{args.port}
    API Documentation: http://{args.host}:{args.port}/
    
    Configuration:
    - Detector: {DETECTOR_PATH}
    - Golden Model: {GOLDEN_MODEL_PATH}
    - Base Tolerance: {BASE_TOLERANCE}
    - Confidence Threshold: {CONFIDENCE_THRESHOLD}
    {'='*60}
    """)
    
    uvicorn.run(
        "app:app",
        host=args.host,
        port=args.port,
        reload=args.reload
    )
