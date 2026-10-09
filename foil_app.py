import streamlit as st
import requests

st.set_page_config(page_title="Tboats Foil Designer", layout="wide")

WORDPRESS_VERIFY_URL = "https://tboat.com/wp-json/tboat/v1/verify-login"
WORDPRESS_ACCESS_URL = "https://tboat.com/wp-json/tboat/v1/app-access"
WORDPRESS_ACCOUNT_URL = "https://tboat.com/my-account/"


def api_headers():
    return {
        "Content-Type": "application/json",
        "X-Tboat-API-Key": st.secrets["TBOAT_API_KEY"],
        "User-Agent": "Tboat-Streamlit-App/1.0",
    }


def verify_wordpress_ticket(ticket):
    try:
        response = requests.post(
            WORDPRESS_VERIFY_URL,
            headers=api_headers(),
            json={"ticket": ticket},
            timeout=10,
        )
        if response.status_code == 200:
            data = response.json()
            if data.get("valid") and data.get("email") and data.get("access") in ("standard", "advanced"):
                return data
    except (requests.RequestException, KeyError, ValueError):
        pass
    return None


def refresh_wordpress_access(email):
    # Only call this after the customer's identity has been verified with a ticket.
    try:
        response = requests.post(
            WORDPRESS_ACCESS_URL,
            headers=api_headers(),
            json={"email": email},
            timeout=10,
        )
        if response.status_code == 200:
            data = response.json()
            if data.get("valid") and data.get("access") in ("standard", "advanced"):
                return data["access"]
    except (requests.RequestException, KeyError, ValueError):
        pass
    return None


if "customer_email" not in st.session_state:
    st.session_state.customer_email = ""
if "app_access" not in st.session_state:
    st.session_state.app_access = None

# A WordPress ticket is short-lived and single-use. Remove it from the URL after
# verification, including if verification fails.
ticket = st.query_params.get("tboat_ticket")
if ticket:
    st.query_params.clear()
    with st.spinner("Verifying your Tboat login..."):
        verified = verify_wordpress_ticket(ticket)
    if verified:
        st.session_state.customer_email = verified["email"].strip().lower()
        st.session_state.app_access = verified["access"]
        st.rerun()
    else:
        st.session_state.customer_email = ""
        st.session_state.app_access = None
        st.error("Your login link has expired or has already been used. Please open a new link from Tboats.")

if not st.session_state.app_access:
    st.title("Tboats Foil Designer")
    st.write("Please sign in to your Tboat account, then open the design app from the Tboats website.")
    st.link_button("Sign in to Tboats", WORDPRESS_ACCOUNT_URL)
    st.stop()

c1, c2, c3 = st.columns([5, 1.5, 1.5])
with c1:
    st.caption(f"Signed in: {st.session_state.customer_email}  •  Access: {st.session_state.app_access.title()}")
with c2:
    if st.button("Check access", use_container_width=True,
                 help="Use this after upgrading to Advanced. Your current design is retained."):
        refreshed = refresh_wordpress_access(st.session_state.customer_email)
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
