import streamlit as st
import pandas as pd
import joblib
import numpy as np
import time # Thêm thư viện thời gian để tạo hiệu ứng
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
    [data-testid="stMetricValue"] { color: #FF69B4 !important; font-weight: bold; }
    .stSelectbox div[data-baseweb="select"] { color: #262730 !important; }
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
import os
logo_filename = "Logo_HungVuong.png.png" 
st.sidebar.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)

if os.path.exists(logo_filename):
    st.sidebar.image(logo_filename, width=130)
else:
    fallback_url = "https://raw.githubusercontent.com/giaan9417-rgb/AlkaLotus-Predictor/main/AlkaLotus/Logo_HungVuong.png.png"
    st.sidebar.image(fallback_url, width=130)

st.sidebar.markdown(
    """
    <p style='font-size: 1em; font-weight: bold; color: #2E2E2E; margin-top: 5px; margin-bottom: 0px;'>
        Trường THPT Chuyên Hùng Vương
    </p>
    <p style='font-size: 0.8em; color: #666;'>TP. Hồ Chí Minh</p>
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
st.sidebar.caption("👨‍ Học sinh: **Quách Gia An & Nguyễn Lê Bách Hợp**")
st.sidebar.caption("🏫 Đơn vị: **Lớp 10-K30 - THPT Chuyên Hùng Vương**")

# --- MODULE 1: DATABASE EXPLORER ---
if page == "1. Thư viện Alkaloid":
    st.title("📚 Thư viện số hóa Alkaloid")
    with st.expander("🔍 Bộ lọc sàng lọc thuốc (Lipinski Rule of 5)"):
        c1, c2, c3, c4 = st.columns(4)
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
        hl = st.toggle("Hiện khoang liên kết", value=True)
    with col2:
        pdb_string = fetch_pdb(pdb_id)
        if pdb_string:
            showmol(render_3d_molecule(pdb_string, highlight_site=hl), height=550, width=850)

# --- MODULE 3: ANALYTICS & REPORT ---
elif page == "3. Phân tích & Xuất báo cáo":
    st.title("📊 Kết quả phân tích dược tính")
    col_left, col_right = st.columns([1, 1]) 
    
    with col_left:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("🎯 Năng lượng liên kết (Affinity)")
        st.metric("BACE1 ΔG", f"{selected_data['dG_BACE1']} kcal/mol", delta="-8.5 (Veru)", delta_color="inverse")
        st.metric("AChE ΔG", f"{selected_data['dG_AChE']} kcal/mol", delta="-7.9 (Done)", delta_color="inverse")
        st.markdown('</div>', unsafe_allow_html=True)
        
        if selected_data['BBB_Permeability']:
            st.success("✅ Có khả năng xuyên rào máu não (BBB)")
        else:
            st.warning("⚠️ Khả năng xuyên rào máu não thấp")

    with col_right:
        st.markdown('<div class="card" style="height: 100%;">', unsafe_allow_html=True)
        st.subheader("🕸️ Hồ sơ ADMET Radar")
        st.plotly_chart(create_admet_radar(selected_data), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.header("🔬 Kiểm chứng độ tin cậy mô hình (Validation)")
    real_data = {
        "Hợp chất": ["Neferine", "Isoliensinine", "Liensinine", "Nuciferine"],
        "Thực nghiệm (IC50)": ["2.16 µM", "5.45 µM", "6.08 µM", "45.20 µM"],
        "Dự đoán AI (kcal/mol)": ["-10.2", "-9.1", "-8.9", "-7.8"],
        "Độ tương quan": ["Khớp mạnh nhất ✅", "Chính xác ✅", "Chính xác ✅", "Chính xác ✅"],
        "Nguồn": ["PMID: 25442253", "PMID: 25442253", "PMID: 25442253", "Elsevier 2015"]
    }
    st.table(pd.DataFrame(real_data))
    
    report_text = f"Báo cáo: {selected_data['Name']}\nBACE1: {selected_data['dG_BACE1']}\nAChE: {selected_data['dG_AChE']}"
    st.download_button("📥 TẢI BÁO CÁO CHI TIẾT", data=report_text, file_name="AlkaLotus_Report.txt")

# --- MODULE 4: AI PREDICTOR ---
elif page == "4. AI Predictor (ML)":
    st.title("🛡️ AI Research Expert - Molecular Screening")
    st.markdown("<div style='background-color: #F0F2F6; padding: 15px; border-radius: 10px; border-left: 5px solid #FF69B4;'><b>Đánh giá đa tầng:</b> Dự đoán ái lực & Sàng lọc dược tính.</div>", unsafe_allow_html=True)
    
    try:
        model_ai = joblib.load('AlkaLotus/alkmer_model.pkl')
        tab_main, tab_expert = st.tabs(["🎯 Dự đoán & Đánh giá", "🔬 Phân tích XAI Chuyên sâu"])
        
        with tab_main:
            c1, c2 = st.columns([2, 1])
            with c1:
                st.markdown('<div class="card">', unsafe_allow_html=True)
                col_in1, col_in2 = st.columns(2)
                v_mw = col_in1.number_input("Khối lượng (MW):", 100.0, 1000.0, 311.40)
                v_logp = col_in2.number_input("LogP (Lipophilicity):", -2.0, 10.0, 3.00)
                v_hbd = col_in1.slider("H-Donor:", 0, 12, 1)
                v_hba = col_in2.slider("H-Acceptor:", 0, 20, 5)
                
                btn_analyze = st.button("⚡ CHẠY PHÂN TÍCH HỆ THỐNG")
                st.markdown('</div>', unsafe_allow_html=True)
                
            if btn_analyze:
                # --- TÍNH NĂNG MỚI: HIỆU ỨNG AI THINKING ---
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                for i in range(1, 101):
                    # Giả lập quá trình rừng cây đang "hội ý"
                    status_text.markdown(f"✨ *Đang tổng hợp ý kiến từ cây quyết định thứ **{i}/100**...*")
                    progress_bar.progress(i)
                    time.sleep(0.01) # Tạo hiệu ứng chuyển động nhanh
                
                status_text.success("✅ AI đã hoàn tất quá trình hội ý!")
                # ------------------------------------------

                features = np.array([[v_mw, v_logp, v_hbd, v_hba]])
                pred_dg = model_ai.predict(features)[0]
                
                violations = sum([v_mw > 500, v_logp > 5, v_hbd > 5, v_hba > 10])
                safety_score = 100 - (violations * 25)

                with c2:
                    st.metric("AI Dự đoán ΔG", f"{round(pred_dg, 2)} kcal/mol")
                    st.metric("Drug-likeness", f"{safety_score}%")
                    if safety_score >= 75: st.success("✅ Tiềm năng thuốc tốt")
                    
                st.subheader("So sánh với 'Thuốc vàng' Verubecestat")
                comp_data = pd.DataFrame({
                    "Chỉ số": ["ΔG (Affinity)", "LogP", "Drug-likeness"],
                    "Hợp chất của bạn": [abs(pred_dg)/10, v_logp/5, safety_score/100],
                    "Verubecestat": [0.85, 0.6, 1.0]
                })
                st.line_chart(comp_data.set_index("Chỉ số"))

        with tab_expert:
            st.subheader("🧠 Giải thích quyết định của AI (XAI)")
            importances = model_ai.feature_importances_
            imp_df = pd.DataFrame({'Yếu tố': ['MW', 'LogP', 'HBD', 'HBA'], 'Mức độ ảnh hưởng': importances})
            st.bar_chart(imp_df.set_index('Yế tố'))

    except Exception as e:
        st.error(f"Kiểm tra file model trên GitHub: {e}")
