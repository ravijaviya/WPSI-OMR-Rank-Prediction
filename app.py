import streamlit as st
import pandas as pd
import json
import gspread
from google.oauth2.service_account import Credentials
import cv2
import numpy as np
import math
import pymupdf  # This is PyMuPDF
from log_helper import log_download_event

# ==========================================
# 1. FIXED CANVAS ALIGNMENT ENGINE
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

def deskew_omr_fallback(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    img_h, img_w = image.shape[:2]
    left_blocks, right_blocks = [], []
    
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if 20 < w < 80 and 10 < h < 40:
            cx, cy = x + (w // 2), y + (h // 2)
            if x < 250:
                left_blocks.append((cx, cy))
            elif x > (img_w - 250):
                right_blocks.append((cx, cy))
                
    angles = []
    if len(left_blocks) >= 2:
        left_blocks = sorted(left_blocks, key=lambda b: b[1])
        dx = left_blocks[-1][0] - left_blocks[0][0]
        dy = left_blocks[-1][1] - left_blocks[0][1]
        angles.append(math.degrees(math.atan2(dx, dy)))
        
    if len(right_blocks) >= 2:
        right_blocks = sorted(right_blocks, key=lambda b: b[1])
        dx = right_blocks[-1][0] - right_blocks[0][0]
        dy = right_blocks[-1][1] - right_blocks[0][1]
        angles.append(math.degrees(math.atan2(dx, dy)))

    if not angles:
        return image

    avg_angle = sum(angles) / len(angles)
    center = (img_w // 2, img_h // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, -avg_angle, 1.0)
    
    return cv2.warpAffine(image, rotation_matrix, (img_w, img_h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

def align_omr_sheet(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Aggressive thresholding to guarantee finding black squares
    # OLD: _, thresh = cv2.threshold(blurred, 150, 255, cv2.THRESH_BINARY_INV)
    
    # NEW:
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    blocks = []

    for c in contours:
        area = cv2.contourArea(c)
        if area < 1000 or area > 150000: 
            continue
            
        x, y, w, h = cv2.boundingRect(c)
        aspect_ratio = w / float(h)
        
        if 0.5 <= aspect_ratio <= 2.0:
            hull = cv2.convexHull(c)
            hull_area = cv2.contourArea(hull)
            if hull_area > 0 and (area / hull_area) > 0.70:
                M = cv2.moments(c)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    blocks.append([cx, cy])

    img_h, img_w = image.shape[:2]
    corners_of_image = np.array([
        [0, 0], [img_w, 0], [img_w, img_h], [0, img_h]
    ])
    
    if len(blocks) >= 4:
        best_markers = []
        for corner in corners_of_image:
            distances = [np.linalg.norm(corner - np.array(b)) for b in blocks]
            best_markers.append(blocks[np.argmin(distances)])

        rect = order_points(np.array(best_markers))
        
        CANVAS_W = 3400
        CANVAS_H = 4700
        MARGIN = 200
        
        dst = np.array([
            [MARGIN, MARGIN],                     
            [CANVAS_W - MARGIN, MARGIN],          
            [CANVAS_W - MARGIN, CANVAS_H - MARGIN], 
            [MARGIN, CANVAS_H - MARGIN]           
        ], dtype="float32")

        M = cv2.getPerspectiveTransform(rect, dst)
        return cv2.warpPerspective(image, M, (CANVAS_W, CANVAS_H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
    
    return deskew_omr_fallback(image)

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
# 3. OMR PARSER (WITH GREEN HIGHLIGHTS)
# ==========================================
def parse_omr_image(cv_img):
    img = align_omr_sheet(cv_img)
    debug_img = img.copy()  # Create a copy to draw on
    
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

def allocate_post(df, vacancies):
    """
    Simulates Horizontal Reservation for a specific post.
    Returns the cutoffs, allocated indices, and demographic composition.
    """
    cutoffs = {
        "GEN Male": "N/A", "GEN Female": "N/A",
        "EWS Male": "N/A", "EWS Female": "N/A",
        "OBC Male": "N/A", "OBC Female": "N/A",
        "SC Male": "N/A", "SC Female": "N/A",
        "ST Male": "N/A", "ST Female": "N/A"
    }
    
    allocated_indices = []
    composition = {}  # Tracks the category breakdown for each allocated seat pool
    eligible_df = df.copy()
    
    if eligible_df.empty:
        return cutoffs, allocated_indices, composition

    # --- STEP 1: GENERAL (UR) MERIT ALLOCATION ---
    gen_seats = vacancies.get("GEN", 0)
    gen_candidates = eligible_df.head(gen_seats).copy()
    eligible_df = eligible_df.drop(gen_candidates.index) 
    
    women_gen_needed = vacancies.get("Women (GEN)", 0)
    women_in_gen = len(gen_candidates[gen_candidates['Gender'] == 'Female'])
    
    if women_in_gen < women_gen_needed:
        shortfall = women_gen_needed - women_in_gen
        swap_in = eligible_df[eligible_df['Gender'] == 'Female'].head(shortfall)
        
        if not swap_in.empty:
            swap_out_indices = gen_candidates[gen_candidates['Gender'] == 'Male'].tail(len(swap_in)).index
            gen_candidates = gen_candidates.drop(swap_out_indices)
            gen_candidates = pd.concat([gen_candidates, swap_in])
            
            swapped_out_males = df.loc[swap_out_indices]
            eligible_df = pd.concat([eligible_df, swapped_out_males]).sort_values(by='Total', ascending=False)
            eligible_df = eligible_df.drop(swap_in.index)

    allocated_indices.extend(gen_candidates.index.tolist())
    
    composition["GEN"] = {"Male": {}, "Female": {}}

    if not gen_candidates.empty:
        males = gen_candidates[gen_candidates['Gender'] == 'Male']
        females = gen_candidates[gen_candidates['Gender'] == 'Female']
        if not males.empty: 
            cutoffs["GEN Male"] = round(males['Total'].min(), 2)
            composition["GEN"]["Male"] = males['Category'].value_counts().to_dict()
        if not females.empty: 
            cutoffs["GEN Female"] = round(females['Total'].min(), 2)
            composition["GEN"]["Female"] = females['Category'].value_counts().to_dict()

    # --- STEP 2: CATEGORY ALLOCATIONS ---
    categories = ["EWS", "OBC", "SC", "ST"]
    
    for cat in categories:
        composition[cat] = {"Male": {}, "Female": {}}
        cat_seats = vacancies.get(cat, 0)
        cat_women_needed = vacancies.get(f"Women ({cat})", 0)
        
        cat_pool = eligible_df[eligible_df['Category'] == cat].copy()
        cat_candidates = cat_pool.head(cat_seats).copy()
        
        women_in_cat = len(cat_candidates[cat_candidates['Gender'] == 'Female'])
        
        if women_in_cat < cat_women_needed:
            shortfall = cat_women_needed - women_in_cat
            remaining_cat_pool = cat_pool.drop(cat_candidates.index)
            swap_in = remaining_cat_pool[remaining_cat_pool['Gender'] == 'Female'].head(shortfall)
            
            if not swap_in.empty:
                swap_out_indices = cat_candidates[cat_candidates['Gender'] == 'Male'].tail(len(swap_in)).index
                cat_candidates = cat_candidates.drop(swap_out_indices)
                cat_candidates = pd.concat([cat_candidates, swap_in])

        allocated_indices.extend(cat_candidates.index.tolist())

        if not cat_candidates.empty:
            males = cat_candidates[cat_candidates['Gender'] == 'Male']
            females = cat_candidates[cat_candidates['Gender'] == 'Female']
            if not males.empty: 
                cutoffs[f"{cat} Male"] = round(males['Total'].min(), 2)
                composition[cat]["Male"] = males['Category'].value_counts().to_dict()
            if not females.empty: 
                cutoffs[f"{cat} Female"] = round(females['Total'].min(), 2)
                composition[cat]["Female"] = females['Category'].value_counts().to_dict()

    return cutoffs, allocated_indices, composition
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
    # Keeps the first 2 digits and the last 2 digits, masking the middle 4
    return roll_str[:2] + "****" + roll_str[-2:] if len(roll_str) == 8 else "****"

# ==========================================
# 5. STREAMLIT UI
# ==========================================
st.set_page_config(page_title="Wireless PSI - OMR Portal", layout="centered")

if 'page' not in st.session_state:
    st.session_state.page = 'Upload OMR'
# --- MOBILE FRIENDLY NAVIGATION ---
nav_options = ["Upload OMR", "Leaderboard", "Live Cut-Offs 🔴", "Answer Keys"]
current_index = nav_options.index(st.session_state.page) if st.session_state.page in nav_options else 0

# Places a horizontal menu at the top of the screen instead of hiding it in a sidebar
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
st.markdown("---") # Adds a clean dividing line under the menu

# --- PAGE 1: UPLOAD OMR ---
if st.session_state.page == 'Upload OMR':
    st.title("📄 Wireless PSI - OMR Upload")
    st.write("Enter your details and upload your scanned OMR sheet (PDF) to evaluate your score.")
    # --- NEW BLINKING WARNING ---
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
    # -----------------------------
    
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
            # Safely initialize session state keys for the file processing
            if 'processed_file_id' not in st.session_state: st.session_state.processed_file_id = None
            if 'result_part_a' not in st.session_state: st.session_state.result_part_a = None
            if 'result_part_b' not in st.session_state: st.session_state.result_part_b = None
            if 'result_total' not in st.session_state: st.session_state.result_total = None
            if 'result_status' not in st.session_state: st.session_state.result_status = None
            if 'result_img' not in st.session_state: st.session_state.result_img = None
                
            # If the current file hasn't been successfully processed yet, show the Submit button
            if st.session_state.processed_file_id != uploaded_file.file_id:
                if st.button("Submit & Evaluate OMR", type="primary"):
                    try:
                        with st.spinner("Processing OMR Sheet & Verifying Data... Please wait."):
                            pdf_bytes = uploaded_file.read()
                            
                            # --- MEMORY OPTIMIZED PDF CONVERSION ---
                            doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
                            if doc.page_count == 0:
                                raise ValueError("The uploaded PDF contains no readable pages.")
                            
                            page = doc.load_page(0) 
                            
                            # Use native DPI scaling matching your test script
                            pix = page.get_pixmap(dpi=300)
                            
                            img_array = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
                            
                            # Safely handle alpha channels matching your test script
                            if pix.n == 4:
                                img_cv = cv2.cvtColor(img_array, cv2.COLOR_RGBA2BGR)
                            else:
                                img_cv = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
                            
                            doc.close()
                            # ---------------------------------------
                            
                            roll_options, paper_options, answers, annotated_img = parse_omr_image(img_cv)
                            
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
                                    # Save full score breakdown to session state
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

            # If the file HAS been successfully processed, show the scorecard and Proceed button
            if st.session_state.processed_file_id == uploaded_file.file_id:
                st.success("✅ Sheet processed and saved successfully! Here is your result:")
                
                # Draw the visual Scorecard
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

        # --- SUMMARY STATISTICS ---
        total_submissions = len(df)
        pass_count = len(df[df['Status'] == 'PASS'])
        fail_count = len(df[df['Status'] == 'FAIL'])
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Submissions", total_submissions)
        col2.metric("PASS", pass_count)
        col3.metric("FAIL", fail_count)
        
        st.markdown("---") # Adds a visual divider before the table

        # --- RANKING AND FORMATTING ---
        df = df.sort_values(by=["Status", "Total"], ascending=[False, False]).reset_index(drop=True)
        df['Rank'] = df[['Status', 'Total']].apply(tuple, axis=1).rank(method='min', ascending=False).astype(int)
        df['Roll Number'] = df['Roll Number'].apply(mask_roll_number)
        
        # --- LIMIT TO TOP 500 ---
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

    # --- NEW VACANCY DETAILS TABLE ---
    st.markdown("---")
    st.subheader("📊 Official Vacancy Details")
    st.write("Category-wise seat distribution based on the official notification:")
    
    # Hardcoding the extracted data from the official image
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
    
    # Displaying the dataframe cleanly
    st.dataframe(vacancy_df, hide_index=True, width='stretch')

# --- PAGE 3: LIVE CUT-OFF PREDICTOR ---
elif st.session_state.page == 'Live Cut-Offs 🔴':
    st.title("🔥 Live Cut-Off Predictor")
    st.write("WPSI seats are allocated to top rankers first. Remaining candidates are then evaluated for TO seats. *These predictions update automatically as more candidates upload their OMR sheets.*")
    
    with st.spinner("Calculating live cut-offs based on current leaderboard..."):
        data = fetch_leaderboard_data()
        
        if data:
            df = pd.DataFrame(data)
            
            # 1. Clean missing data (Default: GEN / Male)
            simulation_df = df.copy()
            simulation_df['Category'] = simulation_df['Category'].replace(["", "N/A", None], "GEN").fillna("GEN")
            simulation_df['Gender'] = simulation_df['Gender'].replace(["", "N/A", None], "Male").fillna("Male")
            
            # 2. Filter strictly for PASSING candidates, sorted by Total Score
            passing_pool = simulation_df[simulation_df['Status'] == 'PASS'].sort_values(by='Total', ascending=False)

            wpsi_vacancies = {
                "GEN": 74, "EWS": 25, "OBC": 38, "SC": 12, "ST": 23,
                "Women (GEN)": 24, "Women (EWS)": 8, "Women (OBC)": 12, "Women (SC)": 3, "Women (ST)": 7
            }
            
            to_vacancies = {
                "GEN": 278, "EWS": 74, "OBC": 165, "SC": 48, "ST": 133,
                "Women (GEN)": 91, "Women (EWS)": 24, "Women (OBC)": 54, "Women (SC)": 15, "Women (ST)": 43
            }

            # 3. Waterfall Step A: Allocate WPSI (Now catching composition)
            wpsi_cutoffs, wpsi_allocated_indices, wpsi_comp = allocate_post(passing_pool, wpsi_vacancies)
            
            # 4. Waterfall Step B: Remove WPSI winners and allocate TO (Now catching composition)
            to_pool = passing_pool.drop(wpsi_allocated_indices, errors='ignore')
            to_cutoffs, _, to_comp = allocate_post(to_pool, to_vacancies)

            # Helper function to format the composition dict into a clean string (e.g., "GEN: 30, OBC: 20")
            def format_comp(comp_dict):
                if not comp_dict: return "-"
                # Sorts highest count first
                return ", ".join([f"{k}: {v}" for k, v in sorted(comp_dict.items(), key=lambda item: item[1], reverse=True)])

            # Build Display Tables
            categories = ["GEN", "EWS", "OBC", "SC", "ST"]
            
            wpsi_display = pd.DataFrame({
                "Category": categories,
                "Male Cut-off": [wpsi_cutoffs[f"{cat} Male"] for cat in categories],
                "Male Make-up": [format_comp(wpsi_comp.get(cat, {}).get("Male", {})) for cat in categories],
                "Female Cut-off": [wpsi_cutoffs[f"{cat} Female"] for cat in categories],
                "Female Make-up": [format_comp(wpsi_comp.get(cat, {}).get("Female", {})) for cat in categories]
            })
            
            to_display = pd.DataFrame({
                "Category": categories,
                "Male Cut-off": [to_cutoffs[f"{cat} Male"] for cat in categories],
                "Male Make-up": [format_comp(to_comp.get(cat, {}).get("Male", {})) for cat in categories],
                "Female Cut-off": [to_cutoffs[f"{cat} Female"] for cat in categories],
                "Female Make-up": [format_comp(to_comp.get(cat, {}).get("Female", {})) for cat in categories]
            })
            
            st.write("If there is N/A means there are no sufficient candidates and all who are eligible will get the post.")
            st.write("The horizontal reservation is calculated dynamically as data grows. Also, we don't have Ex-Army and Specially Abled candidate data which can make cut-off +1 or +2.")
            
            # Tables stacked vertically so the new composition strings fit beautifully
            st.markdown("### 👮 Police Sub Inspector (Wireless)")
            st.dataframe(wpsi_display, hide_index=True, use_container_width=True)
            
            st.markdown("### 📡 Technical Operator (TO)")
            st.dataframe(to_display, hide_index=True, use_container_width=True)
            
            st.markdown("---")
            st.info("✨ **Loving this real-time magic?** \n\nIt takes countless hours of coding and server power to keep this engine running smoothly. If this tool gave you clarity on your rank, consider dropping some motivation in the **Support** section below! 👇")
            
        else:
            st.info("Not enough data to calculate cut-offs yet. Upload an OMR sheet to get started!")
# --- PAGE 4: ANSWER KEYS ---
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
with st.expander("☕ Support this free tool (Optional)", expanded=False):
    st.write("If this portal saved you time, consider a small tip to help keep the servers running!")
    
    # 1. Define your list of supporters here
    supporters = ["Raj", "Vijay", "Zeel", "Gaurang","Dhaval","Jigar"]
    
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

    # --- CSV DOWNLOAD FEATURE ---
    st.markdown("---")
    st.write("🎁 **Bonus:** As you have come here to support, you can download the complete leaderboard list for your own analysis!")
    
    with st.spinner("Preparing your file..."):
        all_data = fetch_leaderboard_data()
        
        if all_data:
            df_dl = pd.DataFrame(all_data)
            
            # Ensure missing columns don't break the script
            if 'Gender' not in df_dl.columns: df_dl['Gender'] = "N/A"
            if 'Category' not in df_dl.columns: df_dl['Category'] = "N/A"
            
            # Mask the roll numbers for privacy
            if 'Roll Number' in df_dl.columns:
                df_dl['Roll Number'] = df_dl['Roll Number'].apply(mask_roll_number)
            
            # Filter strictly to the requested columns
            requested_cols = ['Roll Number', 'Category', 'Gender', 'Paper Code', 'Part A', 'Part B', 'Total']
            # Intersection just in case the sheet is empty or missing a header
            final_cols = [col for col in requested_cols if col in df_dl.columns]
            
            df_dl = df_dl[final_cols]
            
            # Convert to CSV bytes
            csv_bytes = df_dl.to_csv(index=False).encode('utf-8')
            
            st.download_button(
                label="📥 Download Data (CSV)",
                data=csv_bytes,
                file_name="wireless_psi_analysis_data.csv",
                mime="text/csv",
                type="primary",
                on_click=log_download_event,
                args=(get_gspread_client(),)
            )
        else:
            st.info("No data is available to download yet.")

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