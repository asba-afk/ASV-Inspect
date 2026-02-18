"""
ASV-INSPECT - Assembly Inspection System
"""

import sys
from pathlib import Path
import streamlit as st
from PIL import Image
import tempfile

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from inspect_assembly import AssemblyInspector

# Page config
st.set_page_config(
    page_title="ASV-INSPECT",
    page_icon="🔍",
    layout="wide"
)

# Initialize session state
if 'clear_file' not in st.session_state:
    st.session_state.clear_file = False

# Header
st.title("🔍 ASV-INSPECT - Assembly Inspection")
st.write("Upload an image to detect missing components")

# Configuration
st.sidebar.header("⚙️ Settings")

detector_path = st.sidebar.text_input(
    "Detector Model",
    value="runs/models/detector/train/weights/best.pt"
)

golden_model_path = st.sidebar.text_input(
    "Golden Model",
    value="models/golden_model/golden_model.json"
)

confidence_threshold = st.sidebar.slider(
    "Confidence Threshold",
    0.01, 1.0, 0.45, 0.01,
    help="Minimum confidence for detections (0.45 recommended based on training F1 score)"
)

position_tolerance = st.sidebar.slider(
    "Position Tolerance",
    0.05, 0.80, 0.50, 0.05,
    help="How far components can be from expected positions"
)

use_count_only = st.sidebar.checkbox(
    "Count Only Mode (Ignore Positions)",
    value=False,
    help="Only count components by type, don't check positions"
)

if use_count_only:
    st.sidebar.warning("⚠️ Position checking disabled - counting only")
else:
    st.sidebar.success("✅ Position-based with Affine Transformation")
    st.sidebar.info("🔄 System uses affine transformation - handles rotation, scale, and position!")
    st.sidebar.write("**How it works:**")
    st.sidebar.write("• Detects anchor components (bearings, oil jets)")
    st.sidebar.write("• Estimates assembly rotation/scale/position")
    st.sidebar.write("• Transforms expected positions to match")
    st.sidebar.write("• Works at any angle/position/zoom!")

if st.sidebar.button("🔄 Reload Models"):
    st.cache_resource.clear()
    st.rerun()

# Load inspector
@st.cache_resource
def load_inspector(detector_path, golden_model_path, confidence, tolerance):
    """Load the assembly inspector"""
    inspector = AssemblyInspector(
        detector_path=detector_path,
        golden_model_path=golden_model_path,
        confidence_threshold=confidence,
        base_tolerance=tolerance
    )
    return inspector

# File upload
st.subheader("📤 Upload Image")

# Clear file uploader if requested
if st.session_state.clear_file:
    st.session_state.clear_file = False
    st.rerun()

uploaded_file = st.file_uploader("Choose an image", type=['jpg', 'jpeg', 'png', 'bmp'], key='file_uploader')

if uploaded_file:
    # Load inspector
    try:
        # Use very high tolerance only if count-only checkbox is enabled
        effective_tolerance = 999.0 if use_count_only else position_tolerance
        inspector = load_inspector(detector_path, golden_model_path, confidence_threshold, effective_tolerance)
    except Exception as e:
        st.error(f"Failed to load models: {e}")
        st.stop()
    
    # Save temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp:
        tmp.write(uploaded_file.getvalue())
        temp_path = tmp.name
    
    # Run inspection
    with st.spinner("Inspecting..."):
        output_dir = Path(tempfile.mkdtemp())
        report = inspector.inspect(temp_path, str(output_dir), save_visualization=True, save_report=False)
    
    # Display results
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Original")
        st.image(Image.open(uploaded_file), width="stretch")
    
    with col2:
        st.subheader("Results")
        if 'output_image_path' in report:
            st.image(Image.open(report['output_image_path']), width="stretch")
    
    # Status
    st.divider()
    if report['status'] == 'PASS':
        st.success(f"✅ PASS - All components present ({report['compliance_score']*100:.1f}% compliance)")
    else:
        st.error(f"❌ FAIL - Components missing ({report['compliance_score']*100:.1f}% compliance)")
    
    # Stats
    col1, col2, col3 = st.columns(3)
    col1.metric("Expected", report['expected_count'])
    col2.metric("Detected", report['detected_count'])
    col3.metric("Not Detected", report['missing_count'])
    
    # Detection details
    with st.expander("🔍 View detection details"):
        from collections import Counter
        det_by_type = Counter([d['class_name'] for d in report['detections']])
        st.write("**Detected by type:**")
        for comp_type, count in det_by_type.items():
            st.write(f"- {comp_type}: {count}")
        
        st.write(f"\n**Total detections:** {len(report['detections'])}")
        st.write(f"**Confidence threshold used:** {confidence_threshold}")
    
    # Missing components
    if report['missing_count'] > 0:
        st.subheader("⚠️ Missing Components")
        
        st.write("**Red circles mark expected positions with no matching detection:**")
        
        # Group by type
        from collections import Counter
        missing_by_type = Counter([m['class_name'] for m in report['missing_components']])
        
        for comp_type, count in missing_by_type.items():
            st.write(f"• **{comp_type.upper()}**: {count} missing")
        
        # Show positions
        with st.expander(f"📍 See {report['missing_count']} missing component locations"):
            for i, m in enumerate(report['missing_components'], 1):
                st.write(f"{i}. **{m['class_name']}** at position ({m['expected_x']:.1%}, {m['expected_y']:.1%})")
        
        st.info("💡 Adjust sliders: Lower **Confidence** to detect more | Increase **Tolerance** to accept misaligned components")
        
        # Action buttons
        col1, col2 = st.columns(2)
        with col1:
            if st.button("✓ Mark as Verified", type="primary", use_container_width=True):
                st.success("✅ Inspection verified and logged!")
                st.balloons()
        with col2:
            if st.button("🔄 Check Another Image", use_container_width=True):
                st.session_state.clear_file = True
                st.rerun()
    else:
        st.success("✅ All expected components detected!")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("✓ Mark as Verified", type="primary", use_container_width=True):
                st.success("✅ Assembly passed inspection!")
                st.balloons()
        with col2:
            if st.button("🔄 Check Another Image", use_container_width=True):
                st.session_state.clear_file = True
                st.rerun()
    
    # Cleanup
    Path(temp_path).unlink(missing_ok=True)
else:
    st.info("Upload an assembly image to begin inspection")