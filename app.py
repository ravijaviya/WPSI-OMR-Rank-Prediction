import io
import os
import sys
import zipfile
import requests
import streamlit as st

# Set page config for a clean layout
st.set_page_config(
    page_title="Under Maintenance", page_icon="🍲", layout="centered"
)

def maintainance():
    st.markdown(
        """
        <div style="text-align: center; padding: 40px 20px;">
            <h1 style="font-size: 5rem; margin-bottom: 0px;">🍲 👨‍💻 ⚙️</h1>
            <h2 style="margin-top: 10px; color: #f6ad55;">Engineers Out for Dinner!</h2>
            <p style="font-size: 1.15rem; opacity: 0.85; max-width: 550px; margin: 15px auto;">
                The servers are taking a quick power nap while we feed the developers and patch a few final bugs.
            </p>
            <div style="
                display: inline-block; 
                background: rgba(255, 255, 255, 0.05); 
                border: 1px solid rgba(255, 255, 255, 0.15); 
                border-radius: 12px; 
                padding: 12px 24px; 
                margin-top: 20px;
            ">
                <span style="font-size: 0.95rem; font-weight: 600;">
                    ⏳ Status: Under scheduled maintenance • <strong>Please check back shortly after dinner!</strong>
                </span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    
    st.balloons()
    
@st.cache_resource(show_spinner="Bootstrapping secure core engine...")
def initialize_private_modules():
    pat = st.secrets["GITHUB_PAT"]
    repo = st.secrets["GITHUB_REPO"]
    branch = st.secrets.get("GITHUB_BRANCH", "main")

    url = f"https://api.github.com/repos/{repo}/zipball/{branch}"
    headers = {
        "Authorization": f"Bearer {pat}",
        "Accept": "application/vnd.github+json",
    }

    response = requests.get(url, headers=headers, timeout=25)
    if response.status_code != 200:
        st.error(f"Error loading system modules (HTTP {response.status_code})")
        st.stop()

    target_dir = "/tmp/omr_core_src"
    os.makedirs(target_dir, exist_ok=True)

    with zipfile.ZipFile(io.BytesIO(response.content)) as zip_ref:
        zip_ref.extractall(target_dir)

    extracted_roots = [
        os.path.join(target_dir, d)
        for d in os.listdir(target_dir)
        if os.path.isdir(os.path.join(target_dir, d))
    ]

    if extracted_roots:
        root_path = extracted_roots[0]
        if root_path not in sys.path:
            sys.path.insert(0, root_path)

    return True


# 1. Pull the private code
initialize_private_modules()

# 2. Import the unified runner
from main_runner import start_app

# 3. Launch or Maintenance Screen
start_app()
#maintainance()

