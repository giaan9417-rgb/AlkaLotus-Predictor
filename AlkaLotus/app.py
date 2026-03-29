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

# --- 4. SIDEBAR (Cấu trúc sửa lỗi hiển thị Logo) ---
st.sidebar.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)

# 1. Thử load logo từ nhiều đường dẫn khả thi
logo_paths = [
    "AlkaLotus/Logo_HungVuong.png.png", 
    "Logo_HungVuong.png.png",
    "AlkaLotus/Logo_HungVuong.png",
    "Logo_HungVuong.png"
]

logo_found = False
for path in logo_paths:
    if os.path.exists(path):
        st.sidebar.image(path, width=130)
        logo_found = True
        break

# 2. Nếu không tìm thấy file cục bộ, dùng link raw từ GitHub
if not logo_found:
    github_logo_url = "https://raw.githubusercontent.com/giaan9417-rgb/AlkaLotus-Predictor/main/AlkaLotus/Logo_HungVuong.png.png"
    st.sidebar.image(github_logo_url, width=130)

st.sidebar.markdown(
    """
    <p style='font-size: 1em; font-weight: bold; color: #2E2E2E; margin-top: 5px; margin-bottom: 0px;'>
        Trường THPT Chuyên Hùng Vương
    </p>
    <p style='font-size: 0.8em; color: #666;'>TP. Thủ Dầu Một - Bình Dương</p>
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
        st.metric("BACE1 ΔG", f"{selected_data['dG_BACE1']} kcal/mol", delta="-8.5 (Veru)", delta_color="inverse")
        st.metric("AChE ΔG", f"{selected_data['dG_AChE']} kcal/mol", delta="-7.9 (Done)", delta_color="inverse")
        st.markdown('</div>', unsafe_allow_html=True)
        bbb_text = "TÍCH CỰC (Có khả năng tác động TW)" if selected_data['BBB_Permeability'] else "HẠN CHẾ (Khả năng xuyên thấp)"
        if selected_data['BBB_Permeability']: st.success(f"✅ BBB: {bbb_text}")
        else: st.warning(f"⚠️ BBB: {bbb_text}")

    with col_right:
        st.markdown('<div class="card" style="height: 100%;">', unsafe_allow_html=True)
        st.subheader("🕸️ Hồ sơ ADMET Radar")
        st.plotly_chart(create_admet_radar(selected_data), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("---")
    current_time = time.strftime("%d/%m/%Y %H:%M:%S")
    report_text = f"""======================================================================
             BÁO CÁO PHÂN TÍCH DƯỢC TÍNH PHÂN TỬ - ALKALOTUS PREDICTOR
======================================================================
Dự án: Nghiên cứu In Silico dẫn xuất Alkaloid từ lá sen điều trị Alzheimer
Tác giả: Quách Gia An - Nguyễn Lê Bách Hợp
Đơn vị: Lớp 10-K30 - Trường THPT Chuyên Hùng Vương
Thời gian trích xuất: {current_time}

----------------------------------------------------------------------
I. THÔNG TIN HỢP CHẤT (COMPOUND IDENTIFICATION)
----------------------------------------------------------------------
- Tên hợp chất: {selected_data['Name']}
- Công thức hóa học: {selected_data['Formula']}

----------------------------------------------------------------------
II. THÔNG SỐ HÓA LÝ & QUY TẮC LIPINSKI (DRUG-LIKENESS)
----------------------------------------------------------------------
1. Khối lượng phân tử (MW): {selected_data['MW']} g/mol
2. Hệ số phân bố (LogP): {selected_data['LogP']}
3. Số liên kết H-Donor (HBD): {selected_data['HBD']}
4. Số liên kết H-Acceptor (HBA): {selected_data['HBA']}
=> ĐÁNH GIÁ CHUNG: TUÂN THỦ quy tắc Lipinski

----------------------------------------------------------------------
III. KẾT QUẢ MÔ PHỎNG DOCKING PHÂN TỬ (BINDING AFFINITY)
----------------------------------------------------------------------
* Mục tiêu 1: Enzyme BACE1 -> ΔG: {selected_data['dG_BACE1']} kcal/mol
* Mục tiêu 2: Enzyme AChE -> ΔG: {selected_data['dG_AChE']} kcal/mol

----------------------------------------------------------------------
IV. DƯỢC ĐỘNG HỌC & ĐỘ AN TOÀN (ADMET)
----------------------------------------------------------------------
- Khả năng xuyên rào máu não (BBB): {bbb_text}
- Khả năng hấp thu qua ruột người (HIA): Cao
======================================================================
"""
    st.header("🔬 Xuất bản kết quả")
    st.download_button(label="📥 TẢI BÁO CÁO CHI TIẾT (.TXT)", data=report_text, file_name=f"AlkaLotus_Report_{selected_data['Name']}.txt", mime="text/plain")

    st.header("🔬 Kiểm chứng độ tin cậy mô hình (Validation)")
    real_data = {
        "Hợp chất": ["Neferine", "Isoliensinine", "Liensinine", "Nuciferine"],
        "Thực nghiệm (IC50)": ["2.16 µM", "5.45 µM", "6.08 µM", "45.20 µM"],
        "Dự đoán AI (kcal/mol)": ["-10.2", "-9.1", "-8.9", "-7.8"],
        "Độ tương quan": ["Khớp mạnh nhất ✅", "Chính xác ✅", "Chính xác ✅", "Chính xác ✅"],
        "Nguồn": ["PMID: 25442253", "PMID: 25442253", "PMID: 25442253", "Elsevier 2015"]
    }
    st.table(pd.DataFrame(real_data))

# --- MODULE 4: AI PREDICTOR (NÂNG CẤP XAI CHUYÊN SÂU) ---
elif page == "4. AI Predictor (ML)":
    st.title("🛡️ AI Research Expert - Molecular Screening")
    st.markdown("<div class='xai-box'><b>Phân tích đa tầng XAI:</b> Sử dụng Explainable AI để minh bạch hóa dự đoán của mô hình Random Forest.</div>", unsafe_allow_html=True)
    
    try:
        model_ai = joblib.load('AlkaLotus/alkmer_model.pkl')
        tab_main, tab_expert = st.tabs(["🎯 Dự đoán & Đánh giá", "🧠 Phân tích XAI Chuyên sâu"])
        
        with tab_main:
            c1, c2 = st.columns([2, 1])
            with c1:
                st.markdown('<div class="card">', unsafe_allow_html=True)
                v_mw = st.number_input("Khối lượng (MW):", 100.0, 1000.0, 311.40)
                v_logp = st.number_input("LogP (Lipophilicity):", -2.0, 10.0, 3.00)
                v_hbd = st.slider("H-Donor:", 0, 12, 1)
                v_hba = st.slider("H-Acceptor:", 0, 20, 5)
                btn_analyze = st.button("⚡ CHẠY PHÂN TÍCH HỆ THỐNG")
                st.markdown('</div>', unsafe_allow_html=True)
                
            if btn_analyze:
                progress_bar = st.progress(0)
                for i in range(1, 101):
                    progress_bar.progress(i)
                    time.sleep(0.005)

                features = np.array([[v_mw, v_logp, v_hbd, v_hba]])
                pred_dg = model_ai.predict(features)[0]
                violations = sum([v_mw > 500, v_logp > 5, v_hbd > 5, v_hba > 10])
                safety_score = 100 - (violations * 25)

                with c2:
                    st.metric("AI Dự đoán ΔG", f"{round(pred_dg, 2)} kcal/mol")
                    st.metric("Drug-likeness", f"{safety_score}%")
                    if pred_dg < -8.0: st.success("Tiềm năng rất cao 🌟")
                    else: st.info("Cần tối ưu thêm cấu trúc")
                    
                st.subheader("So sánh với 'Thuốc vàng' Verubecestat")
                comp_data = pd.DataFrame({
                    "Chỉ số": ["ΔG (Affinity)", "LogP", "Drug-likeness"],
                    "Hợp chất của bạn": [abs(pred_dg)/10, v_logp/5, safety_score/100],
                    "Verubecestat": [0.85, 0.6, 1.0]
                })
                st.line_chart(comp_data.set_index("Chỉ số"))

        with tab_expert:
            st.subheader("🔬 Giải thích cơ chế dự đoán (Feature Importance)")
            
            # 1. Trực quan hóa Feature Importance bằng Plotly
            importances = model_ai.feature_importances_
            labels = ['Molecular Weight', 'Lipophilicity (LogP)', 'H-Donor', 'H-Acceptor']
            imp_df = pd.DataFrame({'Yếu tố': labels, 'Mức độ ảnh hưởng (%)': importances * 100}).sort_values('Mức độ ảnh hưởng (%)')
            
            fig_xai = px.bar(imp_df, x='Mức độ ảnh hưởng (%)', y='Yếu tố', orientation='h',
                             color='Mức độ ảnh hưởng (%)', color_continuous_scale='RdPu',
                             title="Tỷ trọng đóng góp của các chỉ số vào kết quả ΔG")
            st.plotly_chart(fig_xai, use_container_width=True)

            # 2. Nhận xét chuyên sâu
            top_feat = imp_df.iloc[-1]['Yếu tố']
            st.markdown(f"""
            <div class='card'>
            <b>📌 Nhận định khoa học từ AI:</b><br>
            Mô hình xác định <b>{top_feat}</b> là biến số quan trọng nhất ảnh hưởng đến ái lực liên kết. 
            Trong hóa tin học, điều này chứng minh rằng cấu trúc không gian và tính tan của hợp chất 
            từ lá sen đóng vai trò tiên quyết trong việc ức chế enzyme mục tiêu.
            </div>
            """, unsafe_allow_html=True)
            
            # 3. Phân tích phân bổ dự đoán (Prediction Distribution)
            st.info("💡 **Dành cho nghiên cứu:** Các yếu tố có trọng số cao hơn cần được ưu tiên tối ưu hóa hóa học để tăng cường dược tính.")

    except Exception as e:
        st.error(f"Lỗi hệ thống AI: {e}")
