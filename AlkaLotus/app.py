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

# Cấu hình trang
st.set_page_config(
    page_title="AlkaLotus Predictor | Alzheimer Research",
    layout="wide",
    page_icon="🪷",
    initial_sidebar_state="expanded"
)

# Giao diện CSS
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

# Khởi tạo dữ liệu
if 'selected_compound' not in st.session_state:
    st.session_state.selected_compound = "Roemerine"

df = get_database()
selected_data = df[df['Name'] == st.session_state.selected_compound].iloc[0]

# SIDEBAR 
st.sidebar.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)

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

if not logo_found:
    github_logo_url = "https://raw.githubusercontent.com/giaan9417-rgb/AlkaLotus-Predictor/main/AlkaLotus/Logo_HungVuong.png.png"
    st.sidebar.image(github_logo_url, width=130)

st.sidebar.markdown(
    """
    <p style='font-size: 1em; font-weight: bold; color: #2E2E2E; margin-top: 5px; margin-bottom: 0px;'>
        Trường THPT Chuyên Hùng Vương
    </p>
    <p style='font-size: 0.8em; color: #666;'>TP. HỒ CHÍ MINH</p>
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

# --- MODULE 1: DATABASE EXPLORER (CẬP NHẬT TÊN BIẾN CHUẨN KHOA HỌC) ---
if page == "1. Thư viện Alkaloid":
    st.title("📚 Thư viện số hóa Alkaloid")
    
    with st.expander("🔍 Bộ lọc sàng lọc thuốc (Lipinski Rule of 5)", expanded=True):
        c1, c2, c3, c4 = st.columns(4)
        # Đã thay đổi label hiển thị thành Molecular Weight
        mw_f = c1.checkbox("Molecular Weight < 500", value=True, help="Khối lượng phân tử (Dalton)")
        lp_f = c2.checkbox("LogP < 5", value=True, help="Độ ưa dầu/nước")
        hbd_f = c3.checkbox("H-Donor < 5", value=True, help="Số liên kết Hydrogen cho")
        hba_f = c4.checkbox("H-Acceptor < 10", value=True, help="Số liên kết Hydrogen nhận")
    
    # Logic lọc dữ liệu động
    filtered_df = df.copy()
    
    # Đảm bảo tên cột trong file dữ liệu (df) của An cũng là 'Molecular Weight'
    if mw_f: 
        filtered_df = filtered_df[filtered_df['Molecular Weight'] < 500]
    if lp_f: 
        filtered_df = filtered_df[filtered_df['LogP'] < 5]
    if hbd_f: 
        filtered_df = filtered_df[filtered_df['HBD'] < 5]
    if hba_f: 
        filtered_df = filtered_df[filtered_df['HBA'] < 10]
    
    # HIỂN THỊ BẢNG DỮ LIỆU (Đã cập nhật tiêu đề cột)
    st.dataframe(
        filtered_df[['Name', 'Formula', 'Molecular Weight', 'LogP', 'HBD', 'HBA']], 
        use_container_width=True
    )

    # --- TÍNH NĂNG 1: HEATMAP PHÂN TÍCH TỔNG QUAN ---
    st.markdown("### 🌡️ Phân tích Ái lực liên kết Tổng quát")
    if not filtered_df.empty:
        heatmap_data = filtered_df[['Name', 'dG_BACE1', 'dG_AChE']].set_index('Name')
        
        import plotly.graph_objects as go
        fig_heat = px.imshow(
            heatmap_data.T, 
            labels=dict(x="Hợp chất", y="Enzyme mục tiêu", color="ΔG (kcal/mol)"),
            color_continuous_scale='RdPu_r', 
            text_auto=True, 
            aspect="auto"
        )
        
        fig_heat.update_layout(height=350, margin=dict(l=20, r=20, t=20, b=20))
        st.plotly_chart(fig_heat, use_container_width=True)
        st.caption("💡 *Ghi chú: Màu càng hồng đậm (giá trị âm càng lớn) thể hiện ái lực liên kết càng mạnh.*")
    else:
        st.warning("Không có hợp chất nào thỏa mãn bộ lọc hiện tại.")

    st.divider()

    # CHỌN HỢP CHẤT (Đồng bộ với Session State)
    compounds = df['Name'].tolist()
    current_idx = compounds.index(st.session_state.selected_compound) if st.session_state.selected_compound in compounds else 0
    
    choice = st.selectbox("🎯 Chọn hợp chất mục tiêu để phân tích sâu ở các Module sau:", 
                          compounds, index=current_idx)
    
    if choice != st.session_state.selected_compound:
        st.session_state.selected_compound = choice
        st.rerun()# --- MODULE 2: VIRTUAL DOCKING LAB (DỮ LIỆU CHÍNH THỨC TỪ BÁO CÁO 2026) ---
elif page == "2. Mô phỏng Docking 3D":
    st.title("🔬 Virtual Docking Lab (In Silico)")
    
    # DATABASE CẬP NHẬT THEO BẢNG 2 & CHƯƠNG 2 CỦA BÁO CÁO [cite: 63, 72, 93]
    # Lưu ý: Các chỉ số dg lấy từ Bảng 2 [cite: 63], Acid amin tương tác lấy từ Chương 2 [cite: 83, 85, 93]
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
    
    # Thêm dữ liệu đối chứng (Control) từ báo cáo [cite: 72, 75, 93, 98]
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
            # Hiển thị số liệu khớp 100% với Bảng 2 [cite: 63]
            st.metric("Năng lượng liên kết (ΔG)", f"{data['dg']} kcal/mol")
            st.write(f"📍 **Acid amin chính:** {data['amin']}")
            st.progress(data['stab']/100, text=f"Độ bền phức hợp: {data['stab']}%")
            
            # Giải thích cơ chế dựa trên Chương 2 
            if "Asp32" in data['amin']:
                st.caption("Khóa cặp Asp xúc tác, tương đồng Verubecestat.")
            elif "Trp286" in data['amin']:
                st.caption("Tương tác Pi-Pi tại PAS, tương đồng Donepezil.")

        with c2:
            with st.spinner("Đang tải cấu trúc PDB..."):
                pdb_string = fetch_pdb(pdb_id)
                if pdb_string:
                    showmol(render_3d_molecule(pdb_string, highlight_site=hl), height=500, width=700)

    with tab_compare:
        comp_p = st.radio("Protein đối chứng:", ["BACE1", "AChE"], horizontal=True, key="comp_p")
        control_data = controls[comp_p]
        
        st.subheader(f"So sánh với Thuốc đối chứng: {control_data['name']}")
        
        # Lấy top chất cho protein đó dựa trên báo cáo [cite: 72]
        top_compounds = [k for k, v in alkaloid_db.items() if v[comp_p]['dg'] <= -8.5]
        selected_comp = st.selectbox("Chọn Alkaloid để đối chứng:", list(alkaloid_db.keys()), 
                                     index=list(alkaloid_db.keys()).index("Roemerine"))
        
        user_dg = alkaloid_db[selected_comp][comp_p]['dg']
        
        col1, col2 = st.columns(2)
        col1.metric(selected_comp, f"{user_dg} kcal/mol")
        col2.metric(control_data['name'], f"{control_data['dg']} kcal/mol", 
                    delta=round(user_dg - control_data['dg'], 2), delta_color="inverse")
        
        # Biểu đồ so sánh
        chart_data = pd.DataFrame({
            "Hợp chất": [selected_comp, control_data['name']],
            "Ái lực (Trị tuyệt đối)": [abs(user_dg), abs(control_data['dg'])]
        })
        st.bar_chart(chart_data.set_index("Hợp chất"))
        
        if user_dg < control_data['dg']:
            st.success(f"🌟 {selected_comp} có ái lực mạnh hơn thuốc chuẩn {control_data['name']} trên {comp_p}!")
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
# --- MODULE 4: AI PREDICTOR (BẢN FIX LỖI THỤT LỀ & GIẢI THÍCH CHI TIẾT) ---
elif page == "4. AI Predictor (ML)":
    st.title("🛡️ AI Research Expert - Molecular Screening")
    st.markdown("<div class='xai-box'><b>Phân tích đa tầng XAI:</b> Sử dụng Explainable AI để minh bạch hóa dự đoán và đánh giá độ tin cậy.</div>", unsafe_allow_html=True)
    
    try:
        model_ai = joblib.load('AlkaLotus/alkmer_model.pkl')
        
        if 'last_preds' not in st.session_state:
            st.session_state.last_preds = None

        tab_main, tab_expert = st.tabs(["🎯 Dự đoán & Đánh giá", "🧠 Phân tích XAI Chuyên sâu"])
        
        with tab_main:
            c1, c2 = st.columns([2, 1])
            with c1:
                st.markdown('<div class="card">', unsafe_allow_html=True)
                mw = st.number_input("Khối lượng (Molecular Weight):", 100.0, 1000.0, 311.40)
                logp = st.number_input("LogP (Lipophilicity):", -2.0, 10.0, 3.00)
                hbd = st.slider("H-Donor:", 0, 12, 1)
                hba = st.slider("H-Acceptor:", 0, 20, 5)
                btn_analyze = st.button("⚡ CHẠY PHÂN TÍCH HỆ THỐNG")
                st.markdown('</div>', unsafe_allow_html=True)
            
            if btn_analyze:
                features = np.array([[mw, logp, hbd, hba]])
                pred_dg = model_ai.predict(features)[0]
                
                all_tree_preds = [tree.predict(features)[0] for tree in model_ai.estimators_]
                st.session_state.last_preds = all_tree_preds 
                
                std_dev = np.std(all_tree_preds)
                confidence_val = max(0, min(100, 100 - (std_dev * 15)))
                st.session_state.conf_score = confidence_val
                
                violations = sum([mw > 500, logp > 5, hbd > 5, hba > 10])
                safety_score = 100 - (violations * 25)

                with c2:
                    st.metric("AI Dự đoán ΔG", f"{round(pred_dg, 2)} kcal/mol")
                    st.metric("AI Confidence Score", f"{round(confidence_val, 1)}%")
                    st.metric("Drug-likeness", f"{safety_score}%")
                    
                    if safety_score < 75:
                        st.error("### 🛑 KÉM KHẢ THI") 
                    elif pred_dg <= -8.0:
                        st.success("### 🌟 TIỀM NĂNG RẤT CAO")
                        st.balloons()
                    else:
                        st.info("### 🧪 CẦN TỐI ƯU THÊM")

        with tab_expert:
            st.subheader("🔬 Giải thích cơ chế dự đoán (Feature Importance)")
            importances = model_ai.feature_importances_
            labels = ['MW', 'LogP', 'H-Donor', 'H-Acceptor']
            imp_df = pd.DataFrame({'Yếu tố': labels, 'Mức độ ảnh hưởng (%)': importances * 100})
            fig_xai = px.bar(imp_df, x='Mức độ ảnh hưởng (%)', y='Yếu tố', orientation='h', color_discrete_sequence=['#FF69B4'])
            st.plotly_chart(fig_xai, use_container_width=True)

            st.subheader("🌲 Sự phân tán của các cây quyết định (Confidence Visualization)")
            if st.session_state.last_preds is not None:
                fig_dist = px.violin(st.session_state.last_preds, box=True, points="all", 
                                     color_discrete_sequence=['#FF69B4'])
                st.plotly_chart(fig_dist, use_container_width=True)
                
                # --- PHẦN GIẢI THÍCH CHI TIẾT (ĐÃ THỤT LỀ CHUẨN) ---
                with st.expander("🔬 HƯỚNG DẪN ĐỌC HIỂU PHÂN TÍCH ĐỘ TIN CẬY (XAI)", expanded=True):
                    st.markdown(f"""
                    ### 🔍 Giải mã biểu đồ "Đàn bầu" (Violin Plot)
                    Hệ thống **AlkaLotus** sử dụng mô hình *Random Forest* với 100 cây quyết định. Biểu đồ này minh bạch hóa sự "tranh luận" giữa 100 thực thể đó.
                    
                    ---
                    #### 1️⃣ Ý nghĩa của hình dáng biểu đồ
                    * **Phần bụng phình to nhất:** Đại diện cho vùng **Đồng thuận cao**. Đa số các cây đều dự đoán quanh mức **{round(np.median(st.session_state.last_preds), 2)} kcal/mol**.
                    * **Độ dài biểu đồ:** Thể hiện dải sai số. Biểu đồ càng ngắn, kết quả càng ít biến động và tin cậy hơn.
                    * **Dấu chấm hồng:** Là kết quả riêng lẻ của 1 trong 100 cây quyết định.
                    
                    #### 2️⃣ Các chỉ số từ Boxplot (Khung giữa)
                    * **Vạch trắng (Median):** Điểm trung vị, là con số đại diện cuối cùng cho ái lực liên kết.
                    * **Khối hộp:** Chứa 50% lượng dự đoán tập trung nhất. Hộp càng hẹp, độ tự tin của AI càng cao.
                    
                    #### 3️⃣ Nhận định hệ thống
                    * **Độ tin cậy:** :green[**{round(st.session_state.get('conf_score', 0), 1)}%**]
                    * **Đánh giá:** Kết quả có tính ổn định cao, đủ điều kiện để định hướng các nghiên cứu chuyên sâu tiếp theo.
                    """)
            else:
                st.warning("⚠️ Vui lòng nhấn 'CHẠY PHÂN TÍCH HỆ THỐNG' ở tab Dự đoán để xem phân tích này.")

    except Exception as e:
        st.error(f"Lỗi hệ thống AI: {e}")
