import streamlit as st
import pandas as pd
import joblib
import numpy as np
import time
import os
import plotly.express as px
from stmol import showmol
from data import get_database
from utils import fetch_pdb, render_3d_molecule, check_lipinski, create_admet_radar, classify_potential

# 1. Cấu hình trang
st.set_page_config(
    page_title="AlkaLotus Predictor | Alzheimer Research",
    layout="wide",
    page_icon="🪷",
    initial_sidebar_state="expanded"
)

# 2. Giao diện CSS
st.markdown("""
    <style>
    .stApp { background-color: #FFFFFF !important; }
    h1, h2, h3, h4, h5, h6, p, span, label, .stMarkdown { color: #262730 !important; }
    .card {
        background-color: #F8F9FA; 
        padding: 20px; 
        border-radius: 15px; 
        margin-bottom: 20px;
        border: 1px solid #E0E0E0;
        box-shadow: 2px 2px 8px rgba(0,0,0,0.05);
    }
    .xai-box {
        background-color: #FFF0F5;
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #FF69B4;
    }
    [data-testid="stMetricValue"] { color: #FF69B4 !important; font-weight: bold; }
    .stButton>button { 
        width: 100%; 
        border-radius: 20px; 
        background-color: #FF69B4; 
        color: white;
        font-weight: bold;
        border: none;
    }
    .stButton>button:hover { background-color: #FF1493; color: white; }
    </style>
    """, unsafe_allow_html=True)

# 3. Khởi tạo dữ liệu
if 'selected_compound' not in st.session_state:
    st.session_state.selected_compound = "Roemerine"

df = get_database()
selected_data = df[df['Name'] == st.session_state.selected_compound].iloc[0]

# --- 4. SIDEBAR
st.sidebar.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)

github_logo_url = "https://raw.githubusercontent.com/giaan9417-rgb/AlkaLotus-Predictor/main/AlkaLotus/Logo_HungVuong.png.png"
st.sidebar.image(github_logo_url, width=130)

st.sidebar.markdown(
    """
    <p style='font-size: 1em; font-weight: bold; color: #2E2E2E; margin-top: 5px; margin-bottom: 0px;'>
        TRƯỜNG THPT CHUYÊN HÙNG VƯƠNG
    </p>
    <p style='font-size: 0.8em; color: #666;'>TỈNH BÌNH DƯƠNG</p>
    """, 
    unsafe_allow_html=True
)
st.sidebar.markdown("</div>", unsafe_allow_html=True)
st.sidebar.divider()

st.sidebar.title("🪷 ALKALOTUS PREDICTOR")
st.sidebar.markdown("<div style='text-align: justify; font-size: 0.9em;'><b>Hệ thống tích hợp Machine Learning</b> để tối ưu hóa quy trình sàng lọc ảo và dự đoán dược tính.</div>", unsafe_allow_html=True)

st.sidebar.divider()
page = st.sidebar.radio(
    "Danh mục hệ thống",
    ["1. Thư viện Alkaloid", "2. Mô phỏng Docking 3D", "3. Phân tích & Xuất báo cáo", "4. AI Predictor (ML)"]
)
st.sidebar.divider()
st.sidebar.caption("👨 Học sinh: **Quách Gia An & Nguyễn Lê Bách Hợp**")
st.sidebar.caption("🏫 Đơn vị: **Lớp 10-K30 - THPT Chuyên Hùng Vương**")

# --- MODULE 1: DATABASE EXPLORER ---
if page == "1. Thư viện Alkaloid":
    st.title("📚 Thư viện số hóa Alkaloid")
    with st.expander("🔍 Bộ lọc sàng lọc thuốc (Lipinski Rule of 5)"):
        c1, c2 = st.columns(2)
        mw_f = c1.checkbox("Khối lượng (MW) < 500", value=True)
        lp_f = c2.checkbox("Độ ưa dầu (LogP) < 5", value=True)
    
    filtered_df = df.copy()
    if mw_f: filtered_df = filtered_df[filtered_df['MW'] < 500]
    if lp_f: filtered_df = filtered_df[filtered_df['LogP'] < 5]
    
    st.dataframe(filtered_df[['Name', 'Formula', 'MW', 'LogP', 'HBD', 'HBA']], use_container_width=True)
    
    compounds = df['Name'].tolist()
    choice = st.selectbox("Chọn hợp chất mục tiêu:", compounds, index=compounds.index(st.session_state.selected_compound))
    if choice != st.session_state.selected_compound:
        st.session_state.selected_compound = choice
        st.rerun()

# --- MODULE 2: VIRTUAL DOCKING LAB ---
elif page == "2. Mô phỏng Docking 3D":
    st.title("🔬 Virtual Docking Lab (In Silico)")
    target = st.radio("Chọn Enzyme mục tiêu:", ["BACE1 (Protein 4XXS)", "AChE (Protein 7D9O)"], horizontal=True)
    pdb_id = "4XXS" if "BACE1" in target else "7D9O"
    col1, col2 = st.columns([1, 3])
    with col1:
        st.markdown(f"**Protein:** `{pdb_id}` | **Ligand:** `{st.session_state.selected_compound}`")
        hl = st.toggle("Hiện khoang liên kết (Binding Site)", value=True)
    with col2:
        with st.spinner("Đang dựng cấu trúc phân tử 3D..."):
            pdb_string = fetch_pdb(pdb_id)
            if pdb_string:
                showmol(render_3d_molecule(pdb_string, highlight_site=hl), height=550, width=850)
            else:
                st.error("Không thể kết nối Server RCSB PDB.")

# --- MODULE 3: ANALYTICS & REPORT ---
elif page == "3. Phân tích & Xuất báo cáo":
    st.title("📊 Phân tích Kết quả & Xuất báo cáo")
    st.subheader(f"Thông tin chi tiết hợp chất: {selected_data['Name']}")
    st.markdown('<div class="card">', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    c1.metric("Công thức hóa học", selected_data['Formula'])
    c2.metric("Khối lượng (MW)", f"{selected_data['MW']} Da")
    c3.metric("Độ ưa dầu (LogP)", selected_data['LogP'])
    st.write("---")
    st.metric("Đánh giá Drug-likeness", classify_potential(selected_data['dG_BACE1']))
    st.markdown('</div>', unsafe_allow_html=True)

    col_left, col_right = st.columns([1, 1]) 
    with col_left:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("🎯 Năng lượng liên kết (Affinity)")
        st.metric("BACE1 ΔG", f"{selected_data['dG_BACE1']} kcal/mol")
        st.metric("AChE ΔG", f"{selected_data['dG_AChE']} kcal/mol")
        st.markdown('</div>', unsafe_allow_html=True)

    with col_right:
        st.markdown('<div class="card" style="height: 100%;">', unsafe_allow_html=True)
        st.subheader("🕸️ Hồ sơ ADMET Radar")
        st.plotly_chart(create_admet_radar(selected_data), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("---")
    report_text = f"BAO CAO: {selected_data['Name']}\nMW: {selected_data['MW']}\nLogP: {selected_data['LogP']}"
    st.download_button(label="📥 TẢI BÁO CÁO (.TXT)", data=report_text, file_name=f"Report_{selected_data['Name']}.txt")

# --- MODULE 4: AI PREDICTOR ---
elif page == "4. AI Predictor (ML)":
    st.title("🛡️ AI Research Expert")
    try:
        model_ai = joblib.load('AlkaLotus/alkmer_model.pkl')
        st.markdown('<div class="card">', unsafe_allow_html=True)
        v_mw = st.number_input("Khối lượng (MW):", 100.0, 1000.0, 311.40)
        v_logp = st.number_input("LogP (Lipophilicity):", -2.0, 10.0, 3.00)
        v_hbd = st.slider("H-Donor:", 0, 12, 1)
        v_hba = st.slider("H-Acceptor:", 0, 20, 5)
        if st.button("⚡ CHẠY DỰ ĐOÁN"):
            features = np.array([[v_mw, v_logp, v_hbd, v_hba]])
            pred_dg = model_ai.predict(features)[0]
            st.metric("AI Dự đoán ΔG", f"{round(pred_dg, 2)} kcal/mol")
        st.markdown('</div>', unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Lỗi mô hình AI: {e}")
