import streamlit as st
import cv2
import numpy as np
import pandas as pd
from pdf2image import convert_from_bytes

st.set_page_config(page_title="OMR Evaluation Portal", layout="wide")
st.title("OMR Sheet Scanner & Parser")

# 1. Coordinate Definitions (Pixel coordinates based on your fixed scan DPI)
# Format: (x, y, width, height)
PAPER_CODE_BOXES = {
    'A': (1420, 1020, 30, 30),
    'B': (1460, 1020, 30, 30),
    'C': (1500, 1020, 30, 30),
    'D': (1540, 1020, 30, 30),
    'E': (1580, 1020, 30, 30),
    'F': (1620, 1020, 30, 30)
}

# Example grid layout parameters for 200 MCQs (4 columns of 50)
COLUMNS_X_START = [120, 520, 920, 1320]
ROW_Y_START = 180
ROW_STEP = 35
OPTION_STEP_X = 40
OPTIONS = ['A', 'B', 'C', 'D', 'E']

def evaluate_bubble(thresh_img, x, y, w, h):
    roi = thresh_img[y:y+h, x:x+w]
    return cv2.countNonZero(roi)

def parse_omr(image_cv):
    gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 170, 255, cv2.THRESH_BINARY_INV)

    # 1. Parse Paper Code (A-F)
    paper_code_scores = {code: evaluate_bubble(thresh, *box) for code, box in PAPER_CODE_BOXES.items()}
    paper_code = max(paper_code_scores, key=paper_code_scores.get)

    # 2. Parse 200 MCQ Questions
    results = {}
    for col_idx, start_x in enumerate(COLUMNS_X_START):
        for row_idx in range(50):
            q_num = (col_idx * 50) + row_idx + 1
            y = ROW_Y_START + (row_idx * ROW_STEP)
            
            option_counts = []
            for opt_idx, opt_label in enumerate(OPTIONS):
                x = start_x + (opt_idx * OPTION_STEP_X)
                fill_density = evaluate_bubble(thresh, x, y, 25, 25)
                option_counts.append((opt_label, fill_density))
            
            # Pick highest filled bubble; flag blank if below minimal pixel threshold
            best_opt, max_pixels = max(option_counts, key=lambda item: item[1])
            results[f"Q{q_num}"] = best_opt if max_pixels > 200 else "BLANK"

    return paper_code, results

# --- UI & Upload ---
uploaded_files = st.file_uploader("Upload OMR PDF Sheets", type=["pdf"], accept_multiple_files=True)

if uploaded_files:
    data_records = []
    for file in uploaded_files:
        # Convert first page of PDF to image at fixed 300 DPI
        images = convert_from_bytes(file.read(), dpi=300)
        img_np = np.array(images[0])
        img_cv = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

        code, answers = parse_omr(img_cv)
        
        record = {
            "File Name": file.name,
            "Paper Code": code,
            **answers
        }
        data_records.append(record)

    # Display Table
    df = pd.DataFrame(data_records)
    st.subheader("Processed Results")
    st.dataframe(df, use_container_width=True)

    # Download CSV
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button("Download Results as CSV", data=csv, file_name="omr_results.csv", mime="text/csv")