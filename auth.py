import streamlit as st
from ui_components import apply_custom_styles

PASSWORD = "nopalganteng"  # Password default admin

def check_password():

    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False

    if not st.session_state.authenticated:
        apply_custom_styles()

        st.markdown("""
            <div class="custom-card" style="max-width: 480px; margin: 40px auto 20px auto; text-align: center;">
                <div style="font-size: 3rem; margin-bottom: 8px;">🔒</div>
                <h3 style="margin-top: 0; color: #f8fafc;">Autentikasi Hak Akses</h3>
                <p style="color: #94a3b8; font-size: 0.9rem;">
                    Halaman ini dilindungi password administratif. Silakan masukkan password untuk melanjutkan.
                </p>
            </div>
        """, unsafe_allow_html=True)

        col_l1, col_l2, col_l3 = st.columns([1, 2, 1])

        with col_l2:
            password = st.text_input(
                "Password Admin:",
                type="password",
                placeholder="Masukkan password admin...",
                key="admin_pwd_input"
            )

            if st.button("🔑 Masuk / Login", type="primary", use_container_width=True):
                if password == PASSWORD:
                    st.session_state.authenticated = True
                    st.toast("✅ Login berhasil!")
                    st.rerun()
                else:
                    st.error("❌ Password yang Anda masukkan salah!")

        st.stop()