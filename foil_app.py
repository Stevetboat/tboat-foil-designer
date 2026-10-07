import streamlit as st
import requests

st.set_page_config(page_title="Tboats Foil Designer", layout="wide")
WORDPRESS_ACCESS_URL = "https://tboat.com/wp-json/tboat/v1/app-access"

def check_wordpress_access(email):
    try:
        response = requests.post(
            WORDPRESS_ACCESS_URL,
            headers={
                "Content-Type": "application/json",
                "X-Tboat-API-Key": st.secrets["TBOAT_API_KEY"]
            },
            json={"email": email.strip().lower()},
            timeout=10,
        )

        if response.status_code == 200:
            data = response.json()
            if data.get("valid"):
                return data.get("access", "standard")

        st.error(f"WordPress returned status {response.status_code}: {response.text}")
        return None

    except Exception as e:
        st.error(f"Connection error: {e}")
        return None

if "customer_email" not in st.session_state:
    st.session_state.customer_email = ""
if "app_access" not in st.session_state:
    st.session_state.app_access = None

if not st.session_state.app_access:
    st.title("Tboats Foil Designer")
    st.write("Enter the email address registered with Tboats to open the design app.")
    with st.form("tboat_login"):
        email = st.text_input("Email address")
        submitted = st.form_submit_button("Open Tboat Design App", use_container_width=True)
    if submitted:
        if not email.strip():
            st.error("Please enter your email address.")
        else:
            with st.spinner("Checking access..."):
                access = check_wordpress_access(email)
            if access:
                st.session_state.customer_email = email.strip().lower()
                st.session_state.app_access = access
                st.rerun()
            else:
                st.error("We could not confirm app access for that email address.")
    st.stop()

c1,c2,c3 = st.columns([5,1.5,1.5])
with c1:
    st.caption(f"Signed in: {st.session_state.customer_email}  •  Access: {st.session_state.app_access.title()}")
with c2:
    if st.button("Check access", use_container_width=True,
                 help="Use this after upgrading to Advanced. Your current design is retained."):
        refreshed = check_wordpress_access(st.session_state.customer_email)
        if refreshed:
            st.session_state.app_access = refreshed
            st.rerun()
        else:
            st.error("Access could not be checked.")
with c3:
    if st.button("Sign out", use_container_width=True):
        st.session_state.app_access = None
        st.session_state.customer_email = ""
        st.rerun()

keel = st.Page("keel_app.py", title="Keel Design", icon=":material/straighten:", default=True)
bulb = st.Page("bulb_app.py", title="Bulb Design", icon=":material/egg_alt:")
combined = st.Page("combined_app.py", title="Keel + Bulb", icon=":material/join:")
pg = st.navigation([keel, bulb, combined], position="top")
pg.run()
