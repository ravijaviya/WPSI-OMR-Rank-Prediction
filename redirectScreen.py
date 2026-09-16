import streamlit as st

def start_app():
    """
    Renders a professional decommissioning and migration notice 
    while preserving the standard application header and authentication layout.
    """
    # Standard header auth check matching your active app layout
    user_obj = getattr(st, "user", getattr(st, "experimental_user", None))
    is_logged_in = bool(user_obj and hasattr(user_obj, "is_logged_in") and user_obj.is_logged_in)

    col_title, col_auth = st.columns([3, 1.2])

    with col_title:
        st.title("OMR Evaluation & Analysis Portal")

    with col_auth:
        if is_logged_in:
            st.caption(f"👤 `{user_obj.email}`")
            if hasattr(st, "logout") and st.button("🚪 Logout", type="secondary"):
                st.logout()
        else:
            if hasattr(st, "login") and st.button("🔑 Login with Google", type="primary"):
                st.login("google")

    st.markdown("---")

    # Professional Decommissioning & Redirection Notice
    st.error("⚠️ **Portal Link Decommissioned & Migrated**")
    
    st.markdown(
        """
        ### This specific entry URL has been officially retired.
        
        To provide enhanced security, automated fraud prevention, and faster rank analytics, our infrastructure has been fully migrated to a centralized main platform. 
        
        **Please transition to the new active portal below:**
        """
    )

    # Replace with your actual second app URL
    new_portal_url = "https://your-second-app-url.streamlit.app/"

    col_space1, col_cta, col_space2 = st.columns([1, 2, 1])
    with col_cta:
        st.link_button(
            "🚀 Open New Main Portal", 
            new_portal_url, 
            type="primary", 
            use_container_width=True
        )

    st.markdown("---")
    st.info(
        "💡 **Note for Candidates:** All your previously verified roll numbers, evaluation logs, and leaderboard data are completely safe and active on the new platform. Please update your bookmarks accordingly."
    )
    
    st.caption("Wireless PSI Recruitment Cell • Automated Evaluation System")