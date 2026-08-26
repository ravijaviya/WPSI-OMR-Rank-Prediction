import streamlit as st
import pandas as pd
import json
import gspread
from google.oauth2.service_account import Credentials
import cv2
import numpy as np
import math
import pymupdf  # This is PyMuPDF

# ==========================================
# 1. MULTI-PASS ALIGNMENT ENGINE
# ==========================================
def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

# --- METHOD 1: MAIN (The Micro-Scalpel) ---
def align_omr_sheet_main(image):
    padded = cv2.copyMakeBorder(image, 50, 50, 50, 50, cv2.BORDER_CONSTANT, value=[255, 255, 255])
    gray = cv2.cvtColor(padded, cv2.COLOR_BGR2GRAY)
    
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 51, 15)
    kernel = np.ones((2, 2), np.uint8)
    thresh = cv2.erode(thresh, kernel, iterations=1)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    img_h, img_w = padded.shape[:2]
    best_corners = {0: None, 1: None, 2: None, 3: None}
    min_dists = {0: float('inf'), 1: float('inf'), 2: float('inf'), 3: float('inf')}
    targets = {0: (0, 0), 1: (img_w, 0), 2: (img_w, img_h), 3: (0, img_h)}

    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if 35 < w < 250 and 35 < h < 250:
            aspect_ratio = w / float(h)
            if 0.5 <= aspect_ratio <= 1.9:
                area = cv2.contourArea(c)
                hull = cv2.convexHull(c)
                hull_area = cv2.contourArea(hull)
                if hull_area > 0 and (area / hull_area) > 0.35:
                    M = cv2.moments(c)
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                        
                        q = -1
                        if cx < img_w * 0.40 and cy < img_h * 0.40: q = 0
                        elif cx > img_w * 0.60 and cy < img_h * 0.40: q = 1
                        elif cx > img_w * 0.60 and cy > img_h * 0.60: q = 2
                        elif cx < img_w * 0.40 and cy > img_h * 0.60: q = 3
                        
                        if q != -1:
                            dist = np.linalg.norm(np.array([cx, cy]) - np.array(targets[q]))
                            if dist < min_dists[q]:
                                min_dists[q] = dist
                                best_corners[q] = [cx - 50, cy - 50] 

    matched = {k: v for k, v in best_corners.items() if v is not None}
    best_markers = []
    
    if len(matched) == 4:
        best_markers = [matched[0], matched[1], matched[2], matched[3]]
    elif len(matched) == 3:
        missing_idx = [i for i in range(4) if i not in matched][0]
        TL = np.array(matched.get(0, [0, 0]))
        TR = np.array(matched.get(1, [0, 0]))
        BR = np.array(matched.get(2, [0, 0]))
        BL = np.array(matched.get(3, [0, 0]))
        if missing_idx == 0:   TL = TR + BL - BR
        elif missing_idx == 1: TR = TL + BR - BL
        elif missing_idx == 2: BR = TR + BL - TL
        elif missing_idx == 3: BL = TL + BR - TR
        best_markers = [TL, TR, BR, BL]

    CANVAS_W, CANVAS_H, MARGIN = 3400, 4700, 200
    
    if len(best_markers) == 4:
        rect = order_points(np.array(best_markers))
        dst = np.array([[MARGIN, MARGIN], [CANVAS_W - MARGIN, MARGIN], [CANVAS_W - MARGIN, CANVAS_H - MARGIN], [MARGIN, CANVAS_H - MARGIN]], dtype="float32")
        M = cv2.getPerspectiveTransform(rect, dst)
        return cv2.warpPerspective(image, M, (CANVAS_W, CANVAS_H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
    
    return cv2.resize(image, (CANVAS_W, CANVAS_H))


# --- METHOD 2: FALLBACK (Whitespace Stripping) ---
def align_omr_sheet_fallback(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    valid_points = []
    img_area = image.shape[0] * image.shape[1]
    
    for c in contours:
        area = cv2.contourArea(c)
        if 100 < area < (img_area * 0.5): 
            valid_points.append(c)
            
    if valid_points:
        all_points = np.vstack(valid_points)
        x, y, w, h = cv2.boundingRect(all_points)
        x, y = max(0, x - 10), max(0, y - 10)
        w, h = min(image.shape[1] - x, w + 20), min(image.shape[0] - y, h + 20)
        cropped_img = image[y:y+h, x:x+w]
        crop_offset_x, crop_offset_y = x, y
    else:
        cropped_img = image
        crop_offset_x, crop_offset_y = 0, 0

    c_gray = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY)
    _, c_thresh = cv2.threshold(c_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    crop_contours, _ = cv2.findContours(c_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    img_h, img_w = cropped_img.shape[:2]
    best_corners = {0: None, 1: None, 2: None, 3: None}
    min_dists = {0: float('inf'), 1: float('inf'), 2: float('inf'), 3: float('inf')}
    targets = {0: (0, 0), 1: (img_w, 0), 2: (img_w, img_h), 3: (0, img_h)}
    
    for c in crop_contours:
        cx, cy, cw, ch = cv2.boundingRect(c)
        if 30 < cw < 300 and 30 < ch < 300:
            M = cv2.moments(c)
            if M["m00"] != 0:
                px, py = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
                q = -1
                if px < img_w * 0.20 and py < img_h * 0.20: q = 0
                elif px > img_w * 0.80 and py < img_h * 0.20: q = 1
                elif px > img_w * 0.80 and py > img_h * 0.80: q = 2
                elif px < img_w * 0.20 and py > img_h * 0.80: q = 3
                
                if q != -1:
                    dist = np.linalg.norm(np.array([px, py]) - np.array(targets[q]))
                    if dist < min_dists[q]:
                        min_dists[q] = dist
                        best_corners[q] = [px + crop_offset_x, py + crop_offset_y]

    matched = {k: v for k, v in best_corners.items() if v is not None}
    best_markers = []
    
    if len(matched) == 4:
        best_markers = [matched[0], matched[1], matched[2], matched[3]]
    elif len(matched) == 3:
        missing_idx = [i for i in range(4) if i not in matched][0]
        TL = np.array(matched.get(0, [0, 0]))
        TR = np.array(matched.get(1, [0, 0]))
        BR = np.array(matched.get(2, [0, 0]))
        BL = np.array(matched.get(3, [0, 0]))
        if missing_idx == 0:   TL = TR + BL - BR
        elif missing_idx == 1: TR = TL + BR - BL
        elif missing_idx == 2: BR = TR + BL - TL
        elif missing_idx == 3: BL = TL + BR - TR
        best_markers = [TL, TR, BR, BL]
        
    CANVAS_W, CANVAS_H, MARGIN = 3400, 4700, 200

    if len(best_markers) == 4:
        rect = order_points(np.array(best_markers))
        dst = np.array([[MARGIN, MARGIN], [CANVAS_W - MARGIN, MARGIN], [CANVAS_W - MARGIN, CANVAS_H - MARGIN], [MARGIN, CANVAS_H - MARGIN]], dtype="float32")
        M = cv2.getPerspectiveTransform(rect, dst)
        return cv2.warpPerspective(image, M, (CANVAS_W, CANVAS_H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
    
    fallback_canvas = np.ones((CANVAS_H, CANVAS_W, 3), dtype=np.uint8) * 255
    resized_crop = cv2.resize(cropped_img, (CANVAS_W - (MARGIN * 2), CANVAS_H - (MARGIN * 2)))
    fallback_canvas[MARGIN:CANVAS_H - MARGIN, MARGIN:CANVAS_W - MARGIN] = resized_crop
    return fallback_canvas

def get_bubble_density(thresh_img, center_x, center_y, radius):
    mask = np.zeros(thresh_img.shape, dtype="uint8")
    cv2.circle(mask, (int(round(center_x)), int(round(center_y))), radius, 255, -1)
    bubble_pixels = cv2.bitwise_and(thresh_img, thresh_img, mask=mask)
    return cv2.countNonZero(bubble_pixels) / (np.pi * (radius ** 2))

# ==========================================
# 2. MASTER CALIBRATED COORDINATES
# ==========================================
BUBBLE_RADIUS = 25
ROW_GAP = 81.1
OPT_GAP = 71
START_Y = 445

MCQ_COLS_X = [385, 895, 1400, 1905]
MCQ_OPTIONS = ['A', 'B', 'C', 'D', 'E']

ROLL_START_X = 2365
ROLL_START_Y = 1745
ROLL_COL_GAP = 71.4
ROLL_ROW_GAP = 81.2

CODE_X = 2650
CODE_START_Y = 3205
CODE_ROW_GAP = 81.2
CODE_LABELS = ['A', 'B', 'C', 'D', 'E', 'F']

# ==========================================
# 3. OMR PARSER & VALIDATION
# ==========================================
def extract_data_from_aligned(img):
    debug_img = img.copy() 
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # 1. Roll Number
    roll_options = []
    for col in range(8):
        scores = []
        cx = ROLL_START_X + (col * ROLL_COL_GAP)
        for digit in range(10):
            cy = ROLL_START_Y + (digit * ROLL_ROW_GAP)
            density = get_bubble_density(thresh, cx, cy, BUBBLE_RADIUS)
            scores.append(density)
            if density > 0.35:
                cv2.circle(debug_img, (int(round(cx)), int(round(cy))), BUBBLE_RADIUS + 2, (0, 255, 0), 3)
        
        marked = [str(i) for i, score in enumerate(scores) if score > 0.35]
        roll_options.append(marked if marked else ["?"])

    # 2. Paper Code
    code_scores = []
    for i in range(6):
        cx = CODE_X
        cy = CODE_START_Y + (i * CODE_ROW_GAP)
        density = get_bubble_density(thresh, cx, cy, BUBBLE_RADIUS)
        code_scores.append(density)
        if density > 0.35:
            cv2.circle(debug_img, (int(round(cx)), int(round(cy))), BUBBLE_RADIUS + 2, (0, 255, 0), 3)
            
    marked_codes = [CODE_LABELS[i] for i, score in enumerate(code_scores) if score > 0.35]
    paper_options = marked_codes if marked_codes else ["BLANK"]

    # 3. MCQs
    answers = {}
    for col_idx, start_x in enumerate(MCQ_COLS_X):
        for row in range(50):
            q_num = (col_idx * 50) + row + 1
            cy = START_Y + (row * ROW_GAP)
            
            scores = []
            for opt_idx in range(5):
                cx = start_x + (opt_idx * OPT_GAP)
                density = get_bubble_density(thresh, cx, cy, BUBBLE_RADIUS)
                scores.append(density)
                if density > 0.35:
                    cv2.circle(debug_img, (int(round(cx)), int(round(cy))), BUBBLE_RADIUS + 2, (0, 255, 0), 3)
                    
            marked = [i for i, score in enumerate(scores) if score > 0.35]
            
            if len(marked) == 1:
                answers[f"Q{q_num}"] = MCQ_OPTIONS[marked[0]]
            elif len(marked) > 1:
                answers[f"Q{q_num}"] = "MULTIPLE"
            else:
                answers[f"Q{q_num}"] = "BLANK"

    return roll_options, paper_options, answers, debug_img

def parse_omr_image(cv_img, expected_roll):
    # Pass 1: Attempt the Main Method
    img_main = align_omr_sheet_main(cv_img)
    roll_options, paper_options, answers, debug_img = extract_data_from_aligned(img_main)
    
    # Validation Check: Does the extracted roll perfectly match the user's input?
    roll_match = all(m_digit in opts for m_digit, opts in zip(expected_roll, roll_options))
    
    # Pass 2: If validation fails, trigger the Fallback Method
    if not roll_match:
        img_fb = align_omr_sheet_fallback(cv_img)
        roll_options, paper_options, answers, debug_img = extract_data_from_aligned(img_fb)
        
    return roll_options, paper_options, answers, debug_img

# ==========================================
# 4. GOOGLE SHEETS & GRADING BACKEND
# ==========================================
@st.cache_resource
def get_gspread_client():
    creds_dict = st.secrets["gcp_service_account"]
    scopes = ["https://www.googleapis.com/auth/spreadsheets"]
    creds = Credentials.from_service_account_info(creds_dict, scopes=scopes)
    return gspread.authorize(creds)

@st.cache_data(ttl=6000)
def fetch_answer_key(paper_code):
    try:
        client = get_gspread_client()
        sheet = client.open_by_key("1cRNQiZQRuvBzlsHKynvRJF7AD7Vg0628lNq6Ko6Bsic").worksheet(f"Answer{paper_code}")
        return {str(row['Question']): str(row['Answer']) for row in sheet.get_all_records()}
    except Exception:
        return {}

@st.cache_data(ttl=600)
def fetch_leaderboard_data():
    client = get_gspread_client()
    sheet = client.open_by_key("1cRNQiZQRuvBzlsHKynvRJF7AD7Vg0628lNq6Ko6Bsic").worksheet("Leaderboard")
    return sheet.get_all_records()

def calculate_marks(parsed_answers, paper_code):
    part_a_score, part_b_score = 0.0, 0.0
    key = fetch_answer_key(paper_code)
    
    if not key:
        return None, None, None, None
        
    for q_num_str, user_ans in parsed_answers.items():
        q_num = int(q_num_str.replace("Q", ""))
        correct_ans = str(key.get(q_num_str, "")).strip().upper()
        user_ans = str(user_ans).strip().upper()
        
        q_mark = 0.0
        if user_ans in ["BLANK", "MULTIPLE", "?", ""]:
            q_mark = -0.25
        elif user_ans == "E":
            q_mark = 1.0 if "E" in correct_ans else 0.0
        elif user_ans in ["A", "B", "C", "D"]:
            q_mark = 1.0 if user_ans in correct_ans else -0.25

        if q_num <= 80:
            part_a_score += q_mark
        else:
            part_b_score += q_mark

    total_score = part_a_score + part_b_score
    status = "PASS" if (part_a_score >= 32.0 and part_b_score >= 48.0) else "FAIL"
    return round(part_a_score, 2), round(part_b_score, 2), round(total_score, 2), status

def save_submission(roll_number, paper_code, gender, category, part_a, part_b, total, status, raw_answers):
    client = get_gspread_client()
    sheet = client.open_by_key("1cRNQiZQRuvBzlsHKynvRJF7AD7Vg0628lNq6Ko6Bsic").worksheet("Leaderboard")
    records = sheet.get_all_records()
    
    row_data = [str(roll_number), paper_code, gender, category, float(part_a), float(part_b), float(total), status, json.dumps(raw_answers)]
    
    row_idx = next((i + 2 for i, r in enumerate(records) if str(r.get('Roll Number', '')) == str(roll_number)), None)
    
    if row_idx:
        sheet.update(values=[row_data], range_name=f"A{row_idx}:I{row_idx}")
    else:
        sheet.append_row(row_data)

def recalculate_entire_leaderboard():
    st.cache_data.clear()
    client = get_gspread_client()
    sheet = client.open_by_key("1cRNQiZQRuvBzlsHKynvRJF7AD7Vg0628lNq6Ko6Bsic").worksheet("Leaderboard")
    records = sheet.get_all_records()
    if not records: return
        
    updated_scores = []
    for record in records:
        paper_code = record.get('Paper Code')
        raw_ans = json.loads(record.get('Raw Answers', '{}'))
        
        part_a, part_b, total, status = calculate_marks(raw_ans, paper_code)
        
        if status is None:
            part_a, part_b, total, status = 0.0, 0.0, 0.0, "KEY ERROR"
            
        updated_scores.append([part_a, part_b, total, status]) 
        
    sheet.update(values=updated_scores, range_name=f"E2:H{len(records) + 1}")

def mask_roll_number(roll_no):
    roll_str = str(roll_no)
    return roll_str[:2] + "****" + roll_str[-2:] if len(roll_str) == 8 else "****"

# ==========================================
# 5. STREAMLIT UI
# ==========================================
st.set_page_config(page_title="Wireless PSI - OMR Portal", layout="centered")

if 'page' not in st.session_state:
    st.session_state.page = 'Upload OMR'
# --- MOBILE FRIENDLY NAVIGATION ---
nav_options = ["Upload OMR", "Leaderboard", "Answer Keys"]
current_index = nav_options.index(st.session_state.page) if st.session_state.page in nav_options else 0

page_selection = st.radio(
    "📌 Navigation", 
    nav_options, 
    index=current_index, 
    horizontal=True,
    label_visibility="collapsed"
)

if page_selection != st.session_state.page:
    st.session_state.page = page_selection
    st.rerun()
st.markdown("---")

# --- PAGE 1: UPLOAD OMR ---
if st.session_state.page == 'Upload OMR':
    st.title("📄 Wireless PSI - OMR Upload")
    st.write("Enter your details and upload your scanned OMR sheet (PDF) to evaluate your score.")
    
    st.markdown(
        """
        <style>
        @keyframes blinker {
            50% { opacity: 0; }
        }
        .blinking-text {
            animation: blinker 1.5s linear infinite;
            color: red;
            font-weight: bold;
            font-size: 16px;
            margin-bottom: 15px;
        }
        </style>
        <div class="blinking-text">NOTE: DO NOT ENTER PERSONAL INFORMATION LIKE MOBILE, OTP, ETC.</div>
        <div class="blinking-text">તમારી કોઈ પણ વ્યક્તિગત માહિતી ભરવી નહીં. ફક્ત તમારી વાયરલેસ પીએસઆઇ એક્ઝામ નો સીટ નંબર, કેટેગરી અને પેપર સેટ જેવી સામાન્ય માહિતી જ ભરવી.</div>
        """,
        unsafe_allow_html=True
    )
    
    st.subheader("1. Enter Your Details")
    
    col1, col2 = st.columns(2)
    with col1:
        manual_roll = st.text_input("8-digit Roll Number (Must start with 300)", max_chars=8)
        gender = st.selectbox("Gender", ["Male", "Female"])
    with col2:
        manual_code = st.selectbox("Select Paper Set", ['A', 'B', 'C', 'D', 'E', 'F'])
        category = st.selectbox("Category", ["GEN", "EWS", "OBC", "SC", "ST"])
        
    st.subheader("2. Upload OMR Sheet")
    uploaded_file = st.file_uploader("Upload OMR PDF (Single File Only)", type=["pdf"], accept_multiple_files=False)
    
    if uploaded_file is not None:
        if not manual_roll.startswith("300") or len(manual_roll) != 8:
            st.error("⚠️ Invalid Roll Number. It must be exactly 8 digits long and start with '300'.")
        else:
            if 'processed_file_id' not in st.session_state: st.session_state.processed_file_id = None
            if 'result_part_a' not in st.session_state: st.session_state.result_part_a = None
            if 'result_part_b' not in st.session_state: st.session_state.result_part_b = None
            if 'result_total' not in st.session_state: st.session_state.result_total = None
            if 'result_status' not in st.session_state: st.session_state.result_status = None
            if 'result_img' not in st.session_state: st.session_state.result_img = None
                
            if st.session_state.processed_file_id != uploaded_file.file_id:
                if st.button("Submit & Evaluate OMR", type="primary"):
                    try:
                        with st.spinner("Processing OMR Sheet & Verifying Data... Please wait."):
                            pdf_bytes = uploaded_file.read()
                            
                            doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
                            if doc.page_count == 0:
                                raise ValueError("The uploaded PDF contains no readable pages.")
                            
                            page = doc.load_page(0) 
                            pix = page.get_pixmap(dpi=300)
                            img_array = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
                            
                            if pix.n == 4:
                                img_cv = cv2.cvtColor(img_array, cv2.COLOR_RGBA2BGR)
                            else:
                                img_cv = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
                            
                            doc.close()
                            
                            # MULTI-PASS PARSING (Passing manual_roll to verify alignment)
                            roll_options, paper_options, answers, annotated_img = parse_omr_image(img_cv, manual_roll)
                            
                            roll_match = all(m_digit in opts for m_digit, opts in zip(manual_roll, roll_options))
                            paper_match = manual_code in paper_options
                            
                            display_img = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
                            
                            if not roll_match or not paper_match:
                                scanned_r = "".join([opts[0] if len(opts)==1 else f"[{','.join(opts)}]" for opts in roll_options])
                                scanned_p = paper_options[0] if len(paper_options)==1 else f"[{','.join(paper_options)}]"
                                
                                st.error("❌ **Data Mismatch Detected! Upload Rejected. Contact Admin with Error screenshot.**")
                                st.write(f"- **Scanned Roll No:** `{scanned_r}` | **Entered Roll No:** `{manual_roll}`")
                                st.write(f"- **Scanned Paper Set:** `{scanned_p}` | **Entered Paper Set:** `{manual_code}`")
                                st.info("The system could not verify your manual input against the darkened bubbles. Please check the image below.")
                                st.image(display_img, caption="Detected bubbles are marked in green.")
                            
                            else:
                                roll_number = manual_roll
                                paper_code = manual_code
                                
                                if roll_number == "30010843":
                                    st.info("Admin Hook Triggered: Recalculating all leaderboard scores & statuses...")
                                    recalculate_entire_leaderboard()
                                
                                part_a, part_b, total, status = calculate_marks(answers, paper_code)
                                
                                if status is None:
                                    st.warning(f"⚠️ The Answer Key for Paper Set '{paper_code}' is not available yet. Please try again later.")
                                else:
                                    save_submission(roll_number, paper_code, gender, category, part_a, part_b, total, status, answers)
                                    fetch_leaderboard_data.clear()
                                    
                                    st.session_state.processed_file_id = uploaded_file.file_id
                                    st.session_state.result_part_a = part_a
                                    st.session_state.result_part_b = part_b
                                    st.session_state.result_total = total
                                    st.session_state.result_status = status
                                    st.session_state.result_img = display_img
                                    st.rerun()
                                    
                    except Exception as e:
                        st.error("❌ **Failed to process the PDF file.**")
                        st.error(f"Details: {e}")

            if st.session_state.processed_file_id == uploaded_file.file_id:
                st.success("✅ Sheet processed and saved successfully! Here is your result:")
                
                score_cols = st.columns(4)
                score_cols[0].metric("Part A Score", st.session_state.result_part_a)
                score_cols[1].metric("Part B Score", st.session_state.result_part_b)
                score_cols[2].metric("Total Score", st.session_state.result_total)
                score_cols[3].metric("Status", st.session_state.result_status)
                
                st.markdown("---")
                st.image(st.session_state.result_img, caption="Captured OMR Data (Detected bubbles are circled in green).")
                
                if st.button("Proceed to Leaderboard 🏆"):
                    st.session_state.processed_file_id = None 
                    st.session_state.page = 'Leaderboard'
                    st.rerun()

# --- PAGE 2: LEADERBOARD ---
elif st.session_state.page == 'Leaderboard':
    st.title("🏆 Wireless PSI - Leaderboard")
    
    with st.spinner("Fetching live leaderboard..."):
        data = fetch_leaderboard_data()
    
    if data:
        df = pd.DataFrame(data)
        
        if 'Gender' not in df.columns: df['Gender'] = "N/A"
        if 'Category' not in df.columns: df['Category'] = "N/A"
            
        df['Gender'] = df['Gender'].replace('', 'N/A')
        df['Category'] = df['Category'].replace('', 'N/A')

        total_submissions = len(df)
        pass_count = len(df[df['Status'] == 'PASS'])
        fail_count = len(df[df['Status'] == 'FAIL'])
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Submissions", total_submissions)
        col2.metric("PASS", pass_count)
        col3.metric("FAIL", fail_count)
        
        st.markdown("---") 

        df = df.sort_values(by=["Status", "Total"], ascending=[False, False]).reset_index(drop=True)
        df['Rank'] = df[['Status', 'Total']].apply(tuple, axis=1).rank(method='min', ascending=False).astype(int)
        df['Roll Number'] = df['Roll Number'].apply(mask_roll_number)
        
        display_df = df[['Rank', 'Roll Number', 'Paper Code', 'Gender', 'Category', 'Part A', 'Part B', 'Total', 'Status']].head(500)
        
        def style_status(val):
            if val == 'PASS': color = 'green'
            elif val == 'FAIL': color = 'red'
            else: color = 'orange'
            return f'color: {color}; font-weight: bold;'
            
        st.dataframe(display_df.style.map(style_status, subset=['Status']), width='stretch', hide_index=True)
        st.caption("Showing the top 500 results.")
    else:
        st.info("No submissions yet. Be the first to upload!")

    st.markdown("---")
    st.subheader("📊 Official Vacancy Details")
    st.write("Category-wise seat distribution based on the official notification:")
    
    vacancy_data = {
        "Post Name": ["Police Sub Inspector (Wireless)", "Technical Operator"],
        "Total Seats": [172, 698],
        "GEN": [74, 278],
        "EWS": [25, 74],
        "OBC": [38, 165],
        "SC": [12, 48],
        "ST": [23, 133],
        "Women (GEN)": [24, 91],
        "Women (EWS)": [8, 24],
        "Women (OBC)": [12, 54],
        "Women (SC)": [3, 15],
        "Women (ST)": [7, 43],
        "Ex-Army": [17, 69],
        "PH": [8, 36]
    }
    
    vacancy_df = pd.DataFrame(vacancy_data)
    st.dataframe(vacancy_df, hide_index=True, width='stretch')

# --- PAGE 3: ANSWER KEYS ---
elif st.session_state.page == 'Answer Keys':
    st.title("🔑 Official Answer Keys")
    st.write("View the official answer keys used to grade the OMR sheets.")
    
    selected_set = st.selectbox("Select Paper Set", ['A', 'B', 'C', 'D', 'E', 'F'])
    
    with st.spinner(f"Fetching Answer Key for Set {selected_set}..."):
        ans_key = fetch_answer_key(selected_set)
        
    if ans_key:
        sorted_qnums = sorted(ans_key.keys(), key=lambda x: int(x.replace("Q", "")))
        
        cols = st.columns(4)
        chunk_size = math.ceil(len(sorted_qnums) / 4)
        
        for i in range(4):
            start_idx = i * chunk_size
            end_idx = min((i + 1) * chunk_size, len(sorted_qnums))
            chunk_qnums = sorted_qnums[start_idx:end_idx]
            
            if chunk_qnums:
                table_data = [{"Q No": q, "Answer": ans_key[q]} for q in chunk_qnums]
                df_chunk = pd.DataFrame(table_data)
                cols[i].dataframe(df_chunk, width='stretch', hide_index=True)
    else:
        st.warning(f"⚠️ The Answer Key for Paper Set '{selected_set}' is not available yet.")

# --- SUPPORT EXPANDER (Compact & Non-Intrusive) ---
# --- SUPPORT EXPANDER (Compact & Non-Intrusive) ---
with st.expander("☕ Support this free tool (Optional)", expanded=False):
    st.write("If this portal saved you time, consider a small tip to help keep the servers running!")
    
    # 1. Define your list of supporters here
    supporters = ["Raj", "Vijay", "Zeel", "Gaurang"]
    
    # 2. Dynamically generate the CSS keyframes based on the list length
    num_supporters = len(supporters)
    keyframes_css = ""
    
    for i, name in enumerate(supporters):
        start_pct = int((i / num_supporters) * 100)
        end_pct = int(((i + 1) / num_supporters) * 100) - 1
        
        # Ensure the last frame perfectly ends at 100%
        if i == num_supporters - 1:
            end_pct = 100
            
        keyframes_css += f"{start_pct}%, {end_pct}% {{ content: '🎉 Thank you, {name} for your support!'; }}\n        "

    # 3. Inject the dynamically built CSS
    st.markdown(f"""
        <style>
        @keyframes changeSupporter {{
            {keyframes_css}
        }}
        .supporter-ticker::after {{
            content: '🎉 Thank you, {supporters[0]} for your support!'; 
            animation: changeSupporter {num_supporters * 2}s infinite; /* 2 seconds per person */
            color: #28a745;
            font-weight: 600;
        }}
        .ticker-box {{
            text-align: center;
            padding: 8px;
            margin-bottom: 15px;
            background-color: #f8fff8;
            border-radius: 5px;
            border: 1px dashed #28a745;
        }}
        </style>
        
        <div class="ticker-box">
            <span class="supporter-ticker"></span>
        </div>
    """, unsafe_allow_html=True)
    
    st.write("Note: If direct app redirect does not work, take screenshot and use it. Thanks!!")
    qr_col, text_col = st.columns([1, 2])
    
    with qr_col:
        st.image("QRCode.jpeg", use_container_width=True)
        
    with text_col:
        upi_id = "paytmqr5irfbx@ptys"
        payee_name = "Javiya%20Ravi"
        transaction_note = "Support%20OMR%20Project"
        upi_link = f"upi://pay?pa={upi_id}&pn={payee_name}&tn={transaction_note}&cu=INR"
        
        st.write("**UPI ID:** `paytmqr5irfbx@ptys`")
        st.link_button("Tap to Pay via UPI App (Mobile) 💸", upi_link)

# ==========================================
# 4. GLOBAL FOOTER
# ==========================================
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray; padding-top: 20px;'>
        <p>Developed by <b>RJ</b></p>
        </div>
    """,
    unsafe_allow_html=True
)