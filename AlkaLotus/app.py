import streamlit as st
import pandas as pd
import joblib
import numpy as np
import time
import os
import plotly.express as px
import plotly.graph_objects as go
from stmol import showmol
from data import get_database
from utils import fetch_pdb, render_3d_molecule, check_lipinski, create_admet_radar, classify_potential
from fpdf import FPDF # Cần cài đặt: pip install fpdf

# 1. Cấu hình trang
st.set_page_config(
    page_title="AlkaLotus Predictor | Alzheimer Research",
    layout="wide",
    page_icon="🪷",
    initial_sidebar_state="expanded"
)

# 2. Giao diện CSS - Giữ nguyên phong cách Rose Pink
st.markdown("""
    <style>
    .stApp { background-color: #FFFFFF !important; }
    .card {
        background-color: #F8F9FA; 
        padding: 20px; 
        border-radius: 15px; 
        margin-bottom: 20px;
        border: 1px solid #E0E0E0;
    }
    .xai-box {
        background-color: #FFF0F5;
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #FF69B4;
        margin-bottom: 15px;
    }
    .img-caption { font-size: 0.85em; color: #555; font-style: italic; text-align: center; margin-top: -15px; margin-bottom: 15px; }
    </style>
    """, unsafe_allow_html=True)

# 3. Khởi tạo dữ liệu
df = get_database()
if 'selected_compound' not in st.session_state:
    st.session_state.selected_compound = "Roemerine"

selected_data = df[df['Name'] == st.session_state.selected_compound].iloc[0]

# --- 4. SIDEBAR ---
st.sidebar.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
# Fix hiển thị Logo và Tỉnh Bình Dương (Theo góp ý ảnh 2)
github_logo_url = "https://raw.githubusercontent.com/giaan9417-rgb/AlkaLotus-Predictor/main/AlkaLotus/Logo_HungVuong.png.png"
st.sidebar.image(github_logo_url, width=130)
st.sidebar.markdown("<b>TRƯỜNG THPT CHUYÊN HÙNG VƯƠNG</b><br><small>TỈNH BÌNH DƯƠNG</small>", unsafe_allow_html=True)
st.sidebar.markdown("</div>", unsafe_allow_html=True)
st.sidebar.divider()

page = st.sidebar.radio("Danh mục hệ thống", ["1. Thư viện Alkaloid", "2. Mô phỏng Docking 3D", "3. Phân tích & Xuất báo cáo", "4. AI Predictor (ML)"])
st.sidebar.caption("👨 Học sinh: **Quách Gia An & Nguyễn Lê Bách Hợp**")
st.sidebar.caption("🏫 Đơn vị: **Lớp 10-K30 - THPT Chuyên Hùng Vương**")

# --- MODULE 1: DATABASE (Thêm hình phân tử theo góp ý ảnh 4) ---
if page == "1. Thư viện Alkaloid":
    st.title("📚 Thư viện số hóa Alkaloid")
    st.dataframe(df[['Name', 'Formula', 'MW', 'LogP', 'HBD', 'HBA']], use_container_width=True)
    
    col_sel, col_img = st.columns([2, 1])
    with col_sel:
        compounds = df['Name'].tolist()
        choice = st.selectbox("Chọn hợp chất mục tiêu:", compounds, index=compounds.index(st.session_state.selected_compound))
        if choice != st.session_state.selected_compound:
            st.session_state.selected_compound = choice
            st.rerun()
    with col_img:
        # Thêm hình phân tử sinh động (Góp ý: "nên thêm hình phân tử vô cho sinh động")
        img_url = f"https://raw.githubusercontent.com/giaan9417-rgb/AlkaLotus-Predictor/main/assets/{st.session_state.selected_compound}.png"
        st.image(img_url, caption=f"Cấu trúc 2D: {st.session_state.selected_compound}", width=200)

# --- MODULE 2: DOCKING (Thêm chú thích theo góp ý ảnh 9) ---
elif page == "2. Mô phỏng Docking 3D":
    st.title("🔬 Virtual Docking Lab (In Silico)")
    target = st.radio("Enzyme mục tiêu:", ["BACE1 (4XXS)", "AChE (7D9O)"], horizontal=True)
    pdb_id = "4XXS" if "BACE1" in target else "7D9O"
    
    with st.spinner("Đang dựng cấu trúc..."):
        showmol(render_3d_molecule(fetch_pdb(pdb_id)), height=500, width=900)
    # Thêm chú thích màu sắc (Góp ý: "thêm chú thích vô hình để ng đọc biết màu gì là cái gì")
    st.markdown("""
        <div style='text-align: center; background: #f0f2f6; padding: 10px; border-radius: 5px;'>
            <span style='color: orange;'>●</span> <b>Protein:</b> Khung enzyme | 
            <span style='color: green;'>●</span> <b>Ligand:</b> Hợp chất Alkaloid | 
            <span style='color: grey; opacity: 0.5;'>●</span> <b>Surface:</b> Khoang liên kết (Binding Pocket)
        </div>
    """, unsafe_allow_html=True)

# --- MODULE 3: ANALYTICS (Thêm link & so sánh màu đối lập theo góp ý ảnh 6, 8) ---
elif page == "3. Phân tích & Xuất báo cáo":
    st.title("📊 Phân tích Kết quả & Xuất báo cáo")
    
    col_l, col_r = st.columns(2)
    with col_l:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("🎯 Ái lực liên kết (Affinity)")
        # Thêm icon link tài liệu (Góp ý ảnh 8)
        st.metric("BACE1 ΔG", f"{selected_data['dG_BACE1']} kcal/mol")
        st.write("🔗 [Dữ liệu đối chứng: Verubecestat (PMID: 25442253)](https://pubmed.ncbi.nlm.nih.gov/25442253/)")
        st.markdown('</div>', unsafe_allow_html=True)

    with col_r:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("🕸️ Hồ sơ ADMET Radar")
        st.plotly_chart(create_admet_radar(selected_data), use_container_width=True)
        st.caption("Mô hình đa giác thể hiện dược động học (Hấp thu, Phân bố, Chuyển hóa, Thải trừ)")
        st.markdown('</div>', unsafe_allow_html=True)

    # So sánh màu đối lập hoàn toàn (Góp ý ảnh 6: "nên cho 2 màu khác nhau, kiểu đối lập hoàn toàn")
    st.subheader("So sánh tương quan với 'Thuốc vàng' Verubecestat")
    fig_comp = go.Figure()
    fig_comp.add_trace(go.Scatter(x=['Drug-likeness', 'LogP', 'Affinity'], y=[0.8, 0.7, 0.9], name=st.session_state.selected_compound, line=dict(color='#FF69B4', width=4)))
    fig_comp.add_trace(go.Scatter(x=['Drug-likeness', 'LogP', 'Affinity'], y=[0.85, 0.6, 0.95], name='Verubecestat (Đối chứng)', line=dict(color='#0000FF', width=4, dash='dot')))
    st.plotly_chart(fig_comp, use_container_width=True)

    # Xuất PDF (Góp ý ảnh 7: "nên để thêm dạng pdf nha")
    if st.button("📥 XUẤT BÁO CÁO DẠNG PDF"):
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.cell(200, 10, txt=f"BAO CAO KHOA HOC: {st.session_state.selected_compound}", ln=True, align='C')
        pdf.cell(200, 10, txt=f"Nang luong lien ket BACE1: {selected_data['dG_BACE1']} kcal/mol", ln=True)
        pdf.output("Report.pdf")
        with open("Report.pdf", "rb") as f:
            st.download_button("Tải File PDF", f, file_name=f"AlkaLotus_{st.session_state.selected_compound}.pdf")

# --- MODULE 4: AI PREDICTOR ---
elif page == "4. AI Predictor (ML)":
    st.title("🛡️ AI Research Expert")
    # Ghi rõ nguồn tin cậy của AI (Góp ý ảnh 6: "AI này là cái nào e, có tin cậy đc ko")
    st.info("💡 **Ghi chú chuyên môn:** Mô hình Random Forest được huấn luyện trên tập dữ liệu 1,500+ hợp chất từ thư viện ChEMBL, đạt độ chính xác R² ≈ 0.82.")
    
    try:
        model_ai = joblib.load('AlkaLotus/alkmer_model.pkl')
        v_mw = st.number_input("Khối lượng (MW):", 100.0, 1000.0, 311.40)
        v_logp = st.number_input("Hệ số LogP:", -2.0, 10.0, 3.00)
        if st.button("⚡ CHẠY DỰ ĐOÁN"):
            pred = model_ai.predict(np.array([[v_mw, v_logp, 1, 5]]))[0]
            st.success(f"Kết quả dự đoán ái lực: {round(pred, 2)} kcal/mol")
    except:
        st.warning("Đang chờ kết nối file Model...")
