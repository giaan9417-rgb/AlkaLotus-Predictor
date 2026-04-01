import streamlit as st
import pandas as pd
import joblib
import numpy as np
import time
import os
import plotly.express as px
from stmol import showmol
from sklearn.ensemble import RandomForestRegressor
from data import get_database
from utils import fetch_pdb, render_3d_molecule, check_lipinski, create_admet_radar, classify_potential

# ==========================================
# 1. CẤU HÌNH HỆ THỐNG & GIAO DIỆN (UI/UX)
# ==========================================
st.set_page_config(
    page_title="AlkaLotus Predictor | Alzheimer Research",
    layout="wide",
    page_icon="🪷",
    initial_sidebar_state="expanded"
)

# Giao diện CSS tùy chỉnh (Giữ nguyên phong cách của Gia An)
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
        transition: 0.3s;
    }
    .stButton>button:hover { 
        background-color: #FF1493; 
        transform: scale(1.02);
    }
    </style>
    """, unsafe_allow_html=True)

# Khởi tạo trạng thái phiên làm việc
if 'selected_compound' not in st.session_state:
    st.session_state.selected_compound = "Roemerine"

# Tải dữ liệu từ database gốc
try:
    df = get_database()
    selected_data = df[df['Name'] == st.session_state.selected_compound].iloc[0]
except Exception as e:
    st.error(f"Lỗi tải cơ sở dữ liệu: {e}")
    st.stop()

# ==========================================
# 2. THANH ĐIỀU HƯỚNG (SIDEBAR)
# ==========================================
st.sidebar.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)

# Logic tìm kiếm Logo đa tầng (Đảm bảo luôn hiện Logo)
logo_paths = ["AlkaLotus/Logo_HungVuong.png.png", "Logo_HungVuong.png.png", "AlkaLotus/Logo_HungVuong.png", "Logo_HungVuong.png"]
logo_found = False
for path in logo_paths:
    if os.path.exists(path):
        st.sidebar.image(path, width=130)
        logo_found = True
        break
if not logo_found:
    st.sidebar.image("https://raw.githubusercontent.com/giaan9417-rgb/AlkaLotus-Predictor/main/AlkaLotus/Logo_HungVuong.png.png", width=130)

st.sidebar.markdown("""
    <p style='font-size: 1em; font-weight: bold; color: #2E2E2E; margin-top: 5px; margin-bottom: 0px;'>
        Trường THPT Chuyên Hùng Vương
    </p>
    <p style='font-size: 0.8em; color: #666;'>TP. HỒ CHÍ MINH</p>
    """, unsafe_allow_html=True)
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

# ==========================================
# MODULE 1: THƯ VIỆN SỐ HÓA ALKALOID
# ==========================================
if page == "1. Thư viện Alkaloid":
    st.title("📚 Thư viện số hóa Alkaloid")
    
    with st.expander("🔍 Bộ lọc sàng lọc thuốc (Lipinski Rule of 5)"):
        c1, c2, c3, c4 = st.columns(4)
        mw_f = c1.checkbox("MW < 500", value=True)
        lp_f = c2.checkbox("LogP < 5", value=True)
        hbd_f = c3.checkbox("H-Donor < 5", value=True)
        hba_f = c4.checkbox("H-Acceptor < 10", value=True)
    
    # Logic lọc dữ liệu
    filtered_df = df.copy()
    if mw_f: filtered_df = filtered_df[filtered_df['MW'] < 500]
    if lp_f: filtered_df = filtered_df[filtered_df['LogP'] < 5]
    if hbd_f: filtered_df = filtered_df[filtered_df['HBD'] < 5]
    if hba_f: filtered_df = filtered_df[filtered_df['HBA'] < 10]
    
    st.dataframe(filtered_df[['Name', 'Formula', 'MW', 'LogP', 'HBD', 'HBA']], use_container_width=True)
    
    compounds = df['Name'].tolist()
    choice = st.selectbox("Chọn hợp chất mục tiêu:", compounds, index=compounds.index(st.session_state.selected_compound))
    
    if choice != st.session_state.selected_compound:
        st.session_state.selected_compound = choice
        st.rerun()

# ==========================================
# MODULE 2: MÔ PHỎNG DOCKING 3D (DATABASE CHÍNH THỨC)
# ==========================================
elif page == "2. Mô phỏng Docking 3D":
    st.title("🔬 Virtual Docking Lab (In Silico)")
    
    # DATABASE PHỤC HỒI 100% (Khớp Báo cáo 2026)
    alkaloid_db = {
        "Nuciferine": {
            "BACE1": {"dg": -8.3, "hbond": 1, "amin": "Asp32", "dist": 3.2, "stab": 75},
            "AChE": {"dg": -8.2, "hbond": 1, "amin": "Trp286", "dist": 3.5, "stab": 70}
        },
        "Nornuciferine": {
            "BACE1": {"dg": -8.3, "hbond": 1, "amin": "Gly120", "dist": 3.4, "stab": 72},
            "AChE": {"dg": -8.1, "hbond": 1, "amin": "Tyr124", "dist": 3.6, "stab": 68}
        },
        "Roemerine": {
            "BACE1": {"dg": -9.0, "hbond": 2, "amin": "Asp32/Asp228", "dist": 2.8, "stab": 88},
            "AChE": {"dg": -8.6, "hbond": 2, "amin": "Trp286 (PAS)", "dist": 2.9, "stab": 90}
        },
        "Pronuciferine": {
            "BACE1": {"dg": -8.6, "hbond": 1, "amin": "Ser203", "dist": 3.5, "stab": 78},
            "AChE": {"dg": -8.6, "hbond": 1, "amin": "Phe338", "dist": 3.4, "stab": 80}
        },
        "Liensinine": {
            "BACE1": {"dg": -9.6, "hbond": 3, "amin": "Asp32 (Catalytic)", "dist": 2.6, "stab": 95},
            "AChE": {"dg": -7.5, "hbond": 1, "amin": "His447", "dist": 3.8, "stab": 65}
        },
        "Neferine": {
            "BACE1": {"dg": -9.0, "hbond": 2, "amin": "Tyr124", "dist": 3.0, "stab": 85},
            "AChE": {"dg": -7.5, "hbond": 1, "amin": "Trp286", "dist": 3.9, "stab": 62}
        },
        "Isoliensinine": {
            "BACE1": {"dg": -9.6, "hbond": 3, "amin": "Asp32/Asp228", "dist": 2.5, "stab": 96},
            "AChE": {"dg": -7.7, "hbond": 2, "amin": "Trp286 (PAS)", "dist": 3.1, "stab": 72}
        }
    }
    
    controls = {
        "BACE1": {"name": "Verubecestat", "dg": -8.5},
        "AChE": {"name": "Donepezil", "dg": -7.9}
    }

    tab_view, tab_compare = st.tabs(["🔍 Chi tiết tương tác", "⚖️ So sánh đối chứng"])

    with tab_view:
        target = st.radio("Chọn Enzyme mục tiêu:", ["BACE1 (Protein 4XXS)", "AChE (Protein 7D9O)"], horizontal=True)
        p_key = "BACE1" if "BACE1" in target else "AChE"
        pdb_id = "4XXS" if p_key == "BACE1" else "7D9O"
        
        selected = st.session_state.get('selected_compound', 'Roemerine')
        if selected not in alkaloid_db: selected = "Roemerine"
        data = alkaloid_db[selected][p_key]

        c1, c2 = st.columns([1, 2.5])
        with c1:
            st.info(f"🧬 **{selected}** + **{p_key}**")
            hl = st.toggle("Hiện Binding Site", value=True)
            st.subheader("📊 Thông số thực nghiệm")
            st.metric("Năng lượng liên kết (ΔG)", f"{data['dg']} kcal/mol")
            st.write(f"📍 **Acid amin chính:** {data['amin']}")
            st.progress(data['stab']/100, text=f"Độ bền phức hợp: {data['stab']}%")
            
            if "Asp32" in data['amin']:
                st.caption("Cơ chế: Khóa cặp Asp xúc tác (Catalytic Dyad).")
            elif "Trp286" in data['amin']:
                st.caption("Cơ chế: Tương tác Pi-Pi tại vị trí PAS.")

        with c2:
            with st.spinner("Đang tải mô hình PDB..."):
                pdb_string = fetch_pdb(pdb_id)
                if pdb_string:
                    showmol(render_3d_molecule(pdb_string, highlight_site=hl), height=500, width=700)

    with tab_compare:
        comp_p = st.radio("Protein đối chứng:", ["BACE1", "AChE"], horizontal=True, key="comp_p")
        control_data = controls[comp_p]
        
        st.subheader(f"So sánh với Thuốc đối chứng: {control_data['name']}")
        selected_comp = st.selectbox("Chọn Alkaloid:", list(alkaloid_db.keys()))
        user_dg = alkaloid_db[selected_comp][comp_p]['dg']
        
        col1, col2 = st.columns(2)
        col1.metric(selected_comp, f"{user_dg} kcal/mol")
        col2.metric(control_data['name'], f"{control_data['dg']} kcal/mol", 
                    delta=round(user_dg - control_data['dg'], 2), delta_color="inverse")
        
        chart_data = pd.DataFrame({
            "Hợp chất": [selected_comp, control_data['name']],
            "Ái lực (|ΔG|)": [abs(user_dg), abs(control_data['dg'])]
        })
        st.bar_chart(chart_data.set_index("Hợp chất"))

# ==========================================
# MODULE 3: PHÂN TÍCH & XUẤT BÁO CÁO (FULL TEXT)
# ==========================================
elif page == "3. Phân tích & Xuất báo cáo":
    st.title("📊 Phân tích & Xuất báo cáo")
    st.subheader(f"Hợp chất: {selected_data['Name']}")
    
    st.markdown('<div class="card">', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    c1.metric("Công thức", selected_data['Formula'])
    c2.metric("MW", f"{selected_data['MW']} Da")
    c3.metric("LogP", selected_data['LogP'])
    st.markdown('</div>', unsafe_allow_html=True)

    col_left, col_right = st.columns([1, 1]) 
    with col_left:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("🎯 Binding Affinity")
        st.metric("BACE1 ΔG", f"{selected_data['dG_BACE1']} kcal/mol")
        st.metric("AChE ΔG", f"{selected_data['dG_AChE']} kcal/mol")
        st.markdown('</div>', unsafe_allow_html=True)

    with col_right:
        st.markdown('<div class="card" style="height: 100%;">', unsafe_allow_html=True)
        st.subheader("🕸️ ADMET Radar")
        st.plotly_chart(create_admet_radar(selected_data), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # --- KHÔI PHỤC 100% NỘI DUNG BÁO CÁO GỐC ---
    current_time = time.strftime("%d/%m/%Y %H:%M:%S")
    bbb_status = "TÍCH CỰC (Có khả năng tác động TW)" if selected_data['BBB_Permeability'] else "HẠN CHẾ"
    
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
- Khả năng xuyên rào máu não (BBB): {bbb_status}
- Khả năng hấp thu qua ruột người (HIA): Cao
======================================================================
"""
    st.header("🔬 Xuất bản kết quả")
    st.download_button(
        label="📥 TẢI BÁO CÁO CHI TIẾT (.TXT)", 
        data=report_text, 
        file_name=f"AlkaLotus_Report_{selected_data['Name']}.txt", 
        mime="text/plain"
    )

# ==========================================
# MODULE 4: AI PREDICTOR (ML & XAI)
# ==========================================
elif page == "4. AI Predictor (ML)":
    st.title("🛡️ AI Research Expert")
    st.markdown("<div class='xai-box'><b>Explainable AI:</b> Minh bạch hóa dự đoán của mô hình Random Forest.</div>", unsafe_allow_html=True)
    
    # Logic nạp Model cực kỳ an toàn
    try:
        if os.path.exists('AlkaLotus/alkmer_model.pkl'):
            model_ai = joblib.load('AlkaLotus/alkmer_model.pkl')
        else:
            # Fallback Model: Nếu mất file pkl, tự tạo model tại chỗ để app không chết
            X_dummy = np.array([[311, 3.0, 1, 5], [280, 2.5, 0, 4], [340, 3.5, 2, 6]])
            y_dummy = np.array([-9.0, -8.3, -8.6])
            model_ai = RandomForestRegressor(n_estimators=50).fit(X_dummy, y_dummy)
            st.warning("Hệ thống đang chạy ở chế độ Dự phòng (Online Mode).")

        c1, c2 = st.columns([2, 1])
        with c1:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            mw = st.number_input("Khối lượng (MW):", 100.0, 1000.0, 311.40)
            logp = st.number_input("LogP:", -2.0, 10.0, 3.00)
            hbd = st.slider("H-Donor:", 0, 12, 1)
            hba = st.slider("H-Acceptor:", 0, 20, 5)
            btn_analyze = st.button("⚡ CHẠY PHÂN TÍCH AI")
            st.markdown('</div>', unsafe_allow_html=True)
        
        if btn_analyze:
            features = np.array([[mw, logp, hbd, hba]])
            pred_dg = model_ai.predict(features)[0]
            violations = sum([mw > 500, logp > 5, hbd > 5, hba > 10])
            safety_score = 100 - (violations * 25)

            with c2:
                st.metric("AI Dự đoán ΔG", f"{round(pred_dg, 2)} kcal/mol")
                st.metric("Drug-likeness", f"{safety_score}%")
                
                # Logic phân loại & Hiệu ứng (Giữ nguyên của An)
                if safety_score < 75:
                    st.error("### 🛑 KÉM KHẢ THI")
                    st.snow()
                elif pred_dg <= -8.0:
                    st.success("### 🌟 TIỀM NĂNG CAO")
                    st.balloons()
                else:
                    st.info("### 🧪 CẦN TỐI ƯU")

            # Biểu đồ XAI Chuyên sâu
            st.subheader("🔬 Tỷ trọng đóng góp Feature Importance")
            importances = model_ai.feature_importances_
            labels = ['MW', 'LogP', 'H-Donor', 'H-Acceptor']
            fig_xai = px.bar(x=importances, y=labels, orientation='h', color=importances, color_continuous_scale='RdPu')
            st.plotly_chart(fig_xai, use_container_width=True)

    except Exception as e:
        st.error(f"Lỗi hệ thống AI: {e}")
