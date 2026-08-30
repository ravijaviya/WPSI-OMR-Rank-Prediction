import io
import os
import sys
import zipfile
import requests
import streamlit as st

@st.cache_resource(show_spinner="Bootstrapping secure core engine...")
def initialize_private_modules():
    pat = st.secrets["GITHUB_PAT"]
    repo = st.secrets["GITHUB_REPO"]
    branch = st.secrets.get("GITHUB_BRANCH", "main")
    
    url = f"https://api.github.com/repos/{repo}/zipball/{branch}"
    headers = {
        "Authorization": f"Bearer {pat}",
        "Accept": "application/vnd.github+json"
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

# 3. Launch!
start_app()
