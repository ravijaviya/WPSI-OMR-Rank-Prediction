import streamlit as st
import pandas as pd
import json
import gspread
from google.oauth2.service_account import Credentials
import cv2
import numpy as np
import math
from pdf2image import convert_from_bytes

# ==========================================
# 1. OMR PARSING ENGINE (OPENCV)
# ==========================================
BUBBLE_RADIUS = 25
ROW_GAP = 77.4
OPT_GAP = 78.0
START_Y = 515
MCQ_COLS_X = [385, 925, 1475, 2025]
MCQ_OPTIONS = ['A', 'B', 'C', 'D', 'E']

ROLL_START_X = 2520
ROLL_START_Y = 1752
ROLL_COL_GAP = 78
ROLL_ROW_GAP = 78

CODE_X = 2830
CODE_START_Y = 3145
CODE_ROW_GAP = 78.0
CODE_LABELS = ['A', 'B', 'C', 'D', 'E', 'F']

def deskew_omr(image):
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

def get_bubble_density(thresh_img, center_x, center_y, radius):
    mask = np.zeros(thresh_img.shape, dtype="uint8")
    cv2.circle(mask, (int(round(center_x)), int(round(center_y))), radius, 255, -1)
    bubble_pixels = cv2.bitwise_and(thresh_img, thresh_img, mask=mask)
    return cv2.countNonZero(bubble_pixels) / (np.pi * (radius ** 2))

def parse_omr_image(cv_img):
    img = deskew_omr(cv_img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # 1. Roll Number (Detects MULTIPLE bubbles per column for manual error correction)
    roll_options = []
    for col in range(8):
        scores = []
        cx = ROLL_START_X + (col * ROLL_COL_GAP)
        for digit in range(10):
            cy = ROLL_START_Y + (digit * ROLL_ROW_GAP)
            scores.append(get_bubble_density(thresh, cx, cy, BUBBLE_RADIUS))
        
        marked = [str(i) for i, score in enumerate(scores) if score > 0.35]
        roll_options.append(marked if marked else ["?"])

    # 2. Paper Code (Detects MULTIPLE bubbles)
    code_scores = [get_bubble_density(thresh, CODE_X, CODE_START_Y + (i * CODE_ROW_GAP), BUBBLE_RADIUS) for i in range(6)]
    marked_codes = [CODE_LABELS[i] for i, score in enumerate(code_scores) if score > 0.35]
    paper_options = marked_codes if marked_codes else ["BLANK"]

    # 3. MCQs
    answers = {}
    for col_idx, start_x in enumerate(MCQ_COLS_X):
        for row in range(50):
            q_num = (col_idx * 50) + row + 1
            cy = START_Y + (row * ROW_GAP)
            scores = [get_bubble_density(thresh, start_x + (opt * OPT_GAP), cy, BUBBLE_RADIUS) for opt in range(5)]
            marked = [i for i, score in enumerate(scores) if score > 0.35]
            
            if len(marked) == 1:
                answers[f"Q{q_num}"] = MCQ_OPTIONS[marked[0]]
            elif len(marked) > 1:
                answers[f"Q{q_num}"] = "MULTIPLE"
            else:
                answers[f"Q{q_num}"] = "BLANK"

    return roll_options, paper_options, answers

# ==========================================
# 2. GOOGLE SHEETS & GRADING BACKEND
# ==========================================
@st.cache_resource
def get_gspread_client():
    creds_dict = st.secrets["gcp_service_account"]
    scopes = ["https://www.googleapis.com/auth/spreadsheets"]
    creds = Credentials.from_service_account_info(creds_dict, scopes=scopes)
    return gspread.authorize(creds)

@st.cache_data(ttl=600)
def fetch_answer_key(paper_code):
    try:
        client = get_gspread_client()
        sheet = client.open_by_key("1cRNQiZQRuvBzlsHKynvRJF7AD7Vg0628lNq6Ko6Bsic").worksheet(f"Answer{paper_code}")
        return {str(row['Question']): str(row['Answer']) for row in sheet.get_all_records()}
    except Exception:
        return {}

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

def save_submission(roll_number, paper_code, part_a, part_b, total, status, raw_answers):
    client = get_gspread_client()
    sheet = client.open_by_key("1cRNQiZQRuvBzlsHKynvRJF7AD7Vg0628lNq6Ko6Bsic").worksheet("Leaderboard")
    records = sheet.get_all_records()
    
    row_data = [str(roll_number), paper_code, float(part_a), float(part_b), float(total), status, json.dumps(raw_answers)]
    
    row_idx = next((i + 2 for i, r in enumerate(records) if str(r.get('Roll Number', '')) == str(roll_number)), None)
    
    if row_idx:
        sheet.update(values=[row_data], range_name=f"A{row_idx}:G{row_idx}")
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
        
    sheet.update(values=updated_scores, range_name=f"C2:F{len(records) + 1}")

def mask_roll_number(roll_no):
    roll_str = str(roll_no)
    return roll_str[:4] + "****" if len(roll_str) == 8 else "****"

# ==========================================
# 3. STREAMLIT UI
# ==========================================
st.set_page_config(page_title="Wireless PSI - OMR Portal", layout="centered")

# Set up default page state
if 'page' not in st.session_state:
    st.session_state.page = 'Upload OMR'

# Sidebar Navigation Menu
st.sidebar.title("📌 Navigation")
page_selection = st.sidebar.radio(
    "Go to:",
    ["Upload OMR", "Leaderboard", "Answer Keys"],
    key="page"
)

# --- PAGE 1: UPLOAD OMR ---
if st.session_state.page == 'Upload OMR':
    st.title("📄 Wireless PSI - OMR Upload")
    st.write("Enter your details and upload your scanned OMR sheet (PDF) to evaluate your score.")
    
    st.subheader("1. Enter Your Details")
    col1, col2 = st.columns(2)
    with col1:
        manual_roll = st.text_input("8-digit Roll Number (Must start with 300)", max_chars=8)
    with col2:
        manual_code = st.selectbox("Select Paper Set", ['A', 'B', 'C', 'D', 'E', 'F'])
        
    st.subheader("2. Upload OMR Sheet")
    uploaded_file = st.file_uploader("Upload OMR PDF", type=["pdf"])
    
    if uploaded_file is not None:
        if not manual_roll.startswith("300") or len(manual_roll) != 8:
            st.error("⚠️ Invalid Roll Number. It must be exactly 8 digits long and start with '300'.")
        else:
            with st.spinner("Processing OMR Sheet & Verifying Data... Please wait."):
                images = convert_from_bytes(uploaded_file.read(), dpi=300)
                img_cv = cv2.cvtColor(np.array(images[0]), cv2.COLOR_RGB2BGR)
                
                # Fetch the options detected by the OpenCV engine
                roll_options, paper_options, answers = parse_omr_image(img_cv)
                
                # Validation: Does the manual input exist inside the darkened bubbles?
                roll_match = all(m_digit in opts for m_digit, opts in zip(manual_roll, roll_options))
                paper_match = manual_code in paper_options
                
                if not roll_match or not paper_match:
                    # Format what the scanner actually saw for the error message
                    scanned_r = "".join([opts[0] if len(opts)==1 else f"[{','.join(opts)}]" for opts in roll_options])
                    scanned_p = paper_options[0] if len(paper_options)==1 else f"[{','.join(paper_options)}]"
                    
                    st.error("❌ **Data Mismatch Detected! Upload Rejected.**")
                    st.write(f"- **Scanned Roll No:** `{scanned_r}` | **Entered Roll No:** `{manual_roll}`")
                    st.write(f"- **Scanned Paper Set:** `{scanned_p}` | **Entered Paper Set:** `{manual_code}`")
                    st.info("The system could not verify your manual input against the darkened bubbles on the sheet.")
                
                else:
                    # If it matches, we trust the user's manual input as the absolute truth
                    roll_number = manual_roll
                    paper_code = manual_code
                    
                    if roll_number == "30010843":
                        st.info("Admin Hook Triggered: Recalculating all leaderboard scores & statuses...")
                        recalculate_entire_leaderboard()
                    
                    part_a, part_b, total, status = calculate_marks(answers, paper_code)
                    
                    if status is None:
                        st.warning(f"⚠️ The Answer Key for Paper Set '{paper_code}' is not available yet. Please try again later.")
                    else:
                        save_submission(roll_number, paper_code, part_a, part_b, total, status, answers)
                        st.session_state.page = 'Leaderboard'
                        st.rerun()

# --- PAGE 2: LEADERBOARD ---
elif st.session_state.page == 'Leaderboard':
    st.title("🏆 Wireless PSI - Leaderboard")
    
    with st.spinner("Fetching live leaderboard..."):
        sheet = get_gspread_client().open_by_key("1cRNQiZQRuvBzlsHKynvRJF7AD7Vg0628lNq6Ko6Bsic").worksheet("Leaderboard")
        data = sheet.get_all_records()
    
    if data:
        df = pd.DataFrame(data)
        df = df.sort_values(by=["Status", "Total"], ascending=[False, False]).reset_index(drop=True)
        df['Rank'] = df[['Status', 'Total']].apply(tuple, axis=1).rank(method='min', ascending=False).astype(int)
        df['Roll Number'] = df['Roll Number'].apply(mask_roll_number)
        
        display_df = df[['Rank', 'Roll Number', 'Paper Code', 'Part A', 'Part B', 'Total', 'Status']]
        
        def style_status(val):
            if val == 'PASS': color = 'green'
            elif val == 'FAIL': color = 'red'
            else: color = 'orange' # For KEY ERROR
            return f'color: {color}; font-weight: bold;'
            
        st.dataframe(display_df.style.map(style_status, subset=['Status']), use_container_width=True, hide_index=True)
    else:
        st.info("No submissions yet. Be the first to upload!")

# --- PAGE 3: ANSWER KEYS ---
elif st.session_state.page == 'Answer Keys':
    st.title("🔑 Official Answer Keys")
    st.write("View the official answer keys used to grade the OMR sheets.")
    
    selected_set = st.selectbox("Select Paper Set", ['A', 'B', 'C', 'D', 'E', 'F'])
    
    with st.spinner(f"Fetching Answer Key for Set {selected_set}..."):
        ans_key = fetch_answer_key(selected_set)
        
    if ans_key:
        # Sort keys numerically (Q1, Q2, ..., Q200) instead of alphabetically
        sorted_qnums = sorted(ans_key.keys(), key=lambda x: int(x.replace("Q", "")))
        
        # Divide into 4 columns to prevent excessive scrolling (matches the physical OMR layout)
        cols = st.columns(4)
        
        # Calculate how many questions go into each column (e.g., 200 / 4 = 50)
        chunk_size = math.ceil(len(sorted_qnums) / 4)
        
        for i in range(4):
            # Get the slice of questions for this specific column
            start_idx = i * chunk_size
            end_idx = min((i + 1) * chunk_size, len(sorted_qnums))
            chunk_qnums = sorted_qnums[start_idx:end_idx]
            
            if chunk_qnums:
                # Build DataFrame for just this chunk
                table_data = [{"Q No": q, "Answer": ans_key[q]} for q in chunk_qnums]
                df_chunk = pd.DataFrame(table_data)
                
                # Display in the respective column
                cols[i].dataframe(df_chunk, use_container_width=True, hide_index=True)
    else:
        st.warning(f"⚠️ The Answer Key for Paper Set '{selected_set}' is not available yet.")

# ==========================================
# 4. GLOBAL FOOTER
# ==========================================
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray; padding-top: 20px;'>
        <p>Developed by <b>RJ</b></p>
        <p>Join our Telegram for updates & support: <a href='https://t.me/WirelessPSI2026' target='_blank'>t.me/WirelessPSI2026</a></p>
    </div>
    """,
    unsafe_allow_html=True
)