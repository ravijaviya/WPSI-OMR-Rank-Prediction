import streamlit as st
import requests
from datetime import datetime
from user_agents import parse

def get_client_ip():
    """Extracts the client's public IP from Streamlit headers."""
    try:
        headers = st.context.headers
        if "X-Forwarded-For" in headers:
            return headers["X-Forwarded-For"].split(",")[0].strip()
        elif "X-Real-Ip" in headers:
            return headers["X-Real-Ip"].strip()
    except Exception:
        pass
    return "Unknown"

def get_ip_location(ip_address):
    """Fetches approximate location details for a given IP."""
    if not ip_address or ip_address in ["Unknown", "127.0.0.1", "localhost"]:
        return {"city": "Unknown", "region": "Unknown", "country": "Unknown", "isp": "Unknown"}
    
    try:
        response = requests.get(f"http://ip-api.com/json/{ip_address}?fields=status,country,regionName,city,isp", timeout=3)
        data = response.json()
        if data.get("status") == "success":
            return {
                "city": data.get("city", "N/A"),
                "region": data.get("regionName", "N/A"),
                "country": data.get("country", "N/A"),
                "isp": data.get("isp", "N/A")
            }
    except Exception:
        pass
    return {"city": "Error", "region": "Error", "country": "Error", "isp": "Error"}

def get_device_info():
    """Extracts OS, Device, and Browser info from Streamlit request headers."""
    try:
        user_agent_str = st.context.headers.get("User-Agent", "")
        if user_agent_str:
            user_agent = parse(user_agent_str)
            
            device_type = "Desktop"
            if user_agent.is_mobile:
                device_type = "Mobile"
            elif user_agent.is_tablet:
                device_type = "Tablet"
                
            return {
                "os": f"{user_agent.os.family} {user_agent.os.version_string}".strip(),
                "browser": f"{user_agent.browser.family} {user_agent.browser.version_string}".strip(),
                "device_type": device_type,
                "device_model": user_agent.device.model or "Generic"
            }
    except Exception:
        pass
    return {"os": "Unknown", "browser": "Unknown", "device_type": "Unknown", "device_model": "Unknown"}

def log_download_event(gspread_client):
    """Compiles all data and pushes a new row to the 'Logs' Google Sheet."""
    client_ip = get_client_ip()
    loc = get_ip_location(client_ip)
    device = get_device_info()
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Matches the exact row structure you want in your "Logs" sheet
    log_entry = [
        timestamp,
        client_ip,
        loc["city"],
        loc["region"],
        loc["country"],
        loc["isp"],
        device["os"],
        device["browser"],
        device["device_type"],
        device["device_model"]
    ]
    
    try:
        sheet = gspread_client.open_by_key("1cRNQiZQRuvBzlsHKynvRJF7AD7Vg0628lNq6Ko6Bsic")
        log_ws = sheet.worksheet("Logs")
        log_ws.append_row(log_entry)
        print(f"✅ Successfully logged download: {log_entry}")
    except Exception as e:
        print(f"❌ Failed to log download event to Google Sheets: {e}")