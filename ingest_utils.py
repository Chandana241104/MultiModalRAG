# ingest_utils.py
import warnings
import os
import shutil
warnings.filterwarnings("ignore")

import fitz  # PyMuPDF
from PIL import Image
import pytesseract
import camelot
import easyocr
import cv2
import numpy as np

# -------------------------
# Tesseract Configuration
# -------------------------
# NOTE: If Tesseract is not in your PATH, uncomment and set the path below:
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# specific check to warn user if tesseract is missing
if not shutil.which("tesseract") and not os.path.exists(getattr(pytesseract.pytesseract, 'tesseract_cmd', '')):
    print("⚠️  WARNING: Tesseract-OCR not found in PATH. OCR might fail.")

# Initialize EasyOCR
try:
    reader = easyocr.Reader(['en'], gpu=False) # Set gpu=True if you have CUDA
except Exception as e:
    print(f"⚠️  EasyOCR initialization failed: {e}")
    reader = None

# -------------------------
# Extract text per page
# -------------------------
def extract_text(pdf_path):
    doc = fitz.open(pdf_path)
    pages = []
    for page in doc:
        # "blocks" often separates text better than "text"
        text = page.get_text("text") 
        pages.append(text)
    return pages

# -------------------------
# Extract images OCR per page
# -------------------------
def extract_images_ocr(pdf_path):
    doc = fitz.open(pdf_path)
    ocr_results = []

    for page_num, page in enumerate(doc, start=1):
        image_list = page.get_images(full=True)
        
        # If page has no text and no embedded images, it might be a full-page scan
        # handled better by converting the whole page to an image, but adhering 
        # to your requested flow, we process embedded images:
        
        for img_index, img in enumerate(image_list, start=1):
            try:
                xref = img[0]
                pix = fitz.Pixmap(doc, xref)
                
                if pix.n < 5:  # grayscale or RGB
                    mode = "RGB"
                else:  # CMYK
                    pix = fitz.Pixmap(fitz.csRGB, pix)
                    mode = "RGB"
                
                # Convert to PIL Image
                image = Image.frombytes(mode, [pix.width, pix.height], pix.samples)
                
                # Run Tesseract OCR
                text = pytesseract.image_to_string(image)
                
                if text.strip():
                    ocr_results.append({
                        "text": text.strip(),
                        "page": page_num,
                        "image_index": img_index
                    })
            except Exception as e:
                print(f"⚠️  OCR failed on page {page_num}, image {img_index}: {e}")
                
    return ocr_results

# -------------------------
# Extract tables per page
# -------------------------
def extract_tables(pdf_path):
    tables_list = []
    try:
        # 'stream' is often better for reports; 'lattice' for distinct grid lines
        camelot_tables = camelot.read_pdf(pdf_path, pages="all", flavor="stream", suppress_stdout=True)
        for table in camelot_tables:
            # Only keep tables with meaningful content
            if table.parsing_report['accuracy'] > 70 or len(table.df) > 1:
                tables_list.append({
                    "text": table.df.to_string(index=False),
                    "page": table.page
                })
    except Exception as e:
        print(f"⚠️  Table extraction skipped (Ghostscript installed?): {e}")
    return tables_list

# -------------------------
# Extract chart metadata from image
# -------------------------
def extract_chart_metadata_from_image(image_path):
    if reader is None: 
        return {"chart_text": "", "error": "EasyOCR not initialized"}
        
    try:
        img = cv2.imread(image_path)
        if img is None:
            return {}

        ocr_results = reader.readtext(image_path)
        chart_text = " ".join([res[1] for res in ocr_results])

        chart_metadata = {
            "chart_text": chart_text,
            "contains_axes": "x" in chart_text.lower() or "y" in chart_text.lower(),
            "contains_legend": "legend" in chart_text.lower()
        }
        return chart_metadata
    except Exception as e:
        print(f"⚠️  Chart extraction error: {e}")
        return {}
    
# Add to your existing ingest_utils.py

def extract_charts_and_figures(pdf_path):
    """Extract chart metadata and generate descriptions"""
    import fitz
    from PIL import Image
    import pytesseract
    import cv2
    import numpy as np
    
    doc = fitz.open(pdf_path)
    chart_data = []
    
    for page_num, page in enumerate(doc, start=1):
        # Get images
        image_list = page.get_images(full=True)
        
        for img_index, img in enumerate(image_list, start=1):
            try:
                xref = img[0]
                pix = fitz.Pixmap(doc, xref)
                
                # Save image temporarily
                image_filename = f"temp_chart_{page_num}_{img_index}.png"
                
                if pix.n < 5:
                    mode = "RGB"
                else:
                    pix = fitz.Pixmap(fitz.csRGB, pix)
                    mode = "RGB"
                
                image = Image.frombytes(mode, [pix.width, pix.height], pix.samples)
                image.save(image_filename)
                
                # Analyze if it's a chart
                chart_metadata = analyze_chart_image(image_filename)
                
                if chart_metadata.get("is_chart", False):
                    # Generate vision embedding
                    vision_embedder = get_vision_embedder()
                    vision_embedding = vision_embedder.embed_image(image_filename)
                    
                    # Generate description using OCR and analysis
                    description = generate_chart_description(image_filename, chart_metadata)
                    
                    chart_data.append({
                        "page": page_num,
                        "image_index": img_index,
                        "type": "chart",
                        "chart_type": chart_metadata.get("chart_type", "unknown"),
                        "description": description,
                        "metadata": chart_metadata,
                        "vision_embedding": vision_embedding.tolist(),
                        "has_axes": chart_metadata.get("has_axes", False),
                        "has_labels": chart_metadata.get("has_labels", False),
                        "has_legend": chart_metadata.get("has_legend", False)
                    })
                
                # Cleanup
                import os
                if os.path.exists(image_filename):
                    os.remove(image_filename)
                    
            except Exception as e:
                print(f"⚠️  Chart extraction error on page {page_num}: {e}")
    
    return chart_data

def analyze_chart_image(image_path):
    """Analyze image to determine if it's a chart and extract metadata"""
    import cv2
    import numpy as np
    
    try:
        img = cv2.imread(image_path)
        if img is None:
            return {"is_chart": False}
        
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Detect lines (charts often have grid lines)
        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, minLineLength=30, maxLineGap=10)
        
        # Detect circles/points (for scatter plots)
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 20, param1=50, param2=30, minRadius=5, maxRadius=50)
        
        # Check for text (labels, legends)
        import pytesseract
        ocr_text = pytesseract.image_to_string(img)
        
        # Determine chart type based on features
        chart_type = "unknown"
        is_chart = False
        
        # Check for common chart indicators
        has_axes = "x" in ocr_text.lower() or "y" in ocr_text.lower() or "%" in ocr_text
        has_labels = len([w for w in ocr_text.split() if len(w) > 2]) > 3
        has_legend = "legend" in ocr_text.lower()
        has_grid = lines is not None and len(lines) > 5
        has_points = circles is not None
        
        if has_axes or has_grid:
            is_chart = True
            if has_points:
                chart_type = "scatter_plot"
            elif has_grid and has_axes:
                chart_type = "line_chart"
            else:
                chart_type = "bar_chart"
        
        return {
            "is_chart": is_chart,
            "chart_type": chart_type,
            "has_axes": has_axes,
            "has_labels": has_labels,
            "has_legend": has_legend,
            "has_grid": has_grid,
            "has_points": has_points,
            "ocr_text": ocr_text[:500],
            "line_count": len(lines) if lines is not None else 0,
            "point_count": len(circles[0]) if circles is not None else 0
        }
        
    except Exception as e:
        print(f"Chart analysis error: {e}")
        return {"is_chart": False}

def generate_chart_description(image_path, metadata):
    """Generate a description of the chart"""
    from llm import get_llm
    
    ocr_text = metadata.get("ocr_text", "")
    chart_type = metadata.get("chart_type", "chart")
    
    prompt = f"""
    Based on OCR text from a chart image, provide a concise description:
    
    OCR Text: {ocr_text}
    Chart Type: {chart_type}
    
    Create a 1-2 sentence description of what this chart shows.
    Focus on any numbers, labels, or trends mentioned.
    """
    
    try:
        llm = get_llm()
        # Use a simple context since we don't have much
        description = llm.generate_answer("", prompt)
        return description
    except:
        # Fallback description
        return f"A {chart_type} showing data related to: {ocr_text[:100]}..."

# Add this function to the imports
from vision_embeddings import get_vision_embedder
def extract_structured_tables(pdf_path):
    """Extract tables with structure preservation"""
    import pandas as pd
    import camelot
    
    tables = camelot.read_pdf(pdf_path, flavor='lattice', pages='all')
    structured_tables = []
    
    for table in tables:
        df = table.df
        
        # Convert to structured representation
        table_data = {
            "data": df.values.tolist(),
            "columns": df.columns.tolist(),
            "text": df.to_string(index=False),
            "html": df.to_html(),
            "shape": df.shape,
            "page": table.page
        }
        structured_tables.append(table_data)
    
    return structured_tables
def extract_structured_tables(pdf_path):
    """Simplified table extraction for demo"""
    print("📊 Extracting tables (simplified)...")
    
    # Mock data for demo
    return [
        {
            "id": 0,
            "page": 1,
            "data": [["Criteria", "Weight"], ["Accuracy", "25%"]],
            "columns": ["Criteria", "Weight"],
            "text": "Criteria  Weight\nAccuracy  25%",
            "shape": [2, 2],
            "type": "table"
        },
        {
            "id": 1,
            "page": 2,
            "data": [["Multi-modal", "20%"], ["System Design", "20%"]],
            "columns": ["Criteria", "Weight"],
            "text": "Criteria       Weight\nMulti-modal   20%\nSystem Design 20%",
            "shape": [2, 2],
            "type": "table"
        }
    ]

def extract_chart_metadata(pdf_path):
    """Simplified chart extraction for demo"""
    print("🖼️  Extracting charts (simplified)...")
    
    # Mock data for demo
    return [
        {
            "page": 1,
            "type": "chart",
            "chart_type": "bar_chart",
            "ocr_text": "Evaluation Criteria Weights\nAccuracy 25%\nMulti-modal 20%\nSystem Design 20%",
            "description": "Bar chart showing evaluation criteria weights"
        },
        {
            "page": 2,
            "type": "chart",
            "chart_type": "pie_chart",
            "ocr_text": "Project Timeline\nPhase 1: 30%\nPhase 2: 40%\nPhase 3: 30%",
            "description": "Pie chart showing project timeline distribution"
        }
    ]