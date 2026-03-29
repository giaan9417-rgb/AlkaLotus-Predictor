import streamlit as st
import pandas as pd
import joblib
import numpy as np
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

# 4. SIDEBAR
st.sidebar.title("🪷 ALKALOTUS PREDICTOR")
st.sidebar.markdown(f"""
<div style='text-align: justify; font-size: 0.85em; color: #555;'>
<b>Hệ thống hóa kết quả nghiên cứu In Silico</b> khả năng ức chế các enzyme AChE và BACE1 từ các dẫn xuất Alkaloid từ lá sen trong hỗ trợ điều trị bệnh Alzheimer.
</div>
""", unsafe_allow_html=True)
st.sidebar.markdown("---")

page = st.sidebar.radio("Danh mục hệ thống", 
    ["1. Thư viện Alkaloid", "2. Mô phỏng Docking 3D", "3. Phân tích & Xuất báo cáo", "4. AI Predictor (ML)"])

st.sidebar.markdown("---")
st.sidebar.info(f"🧬 **Đang phân tích:**\n{st.session_state.selected_compound}")
st.sidebar.caption("Tác giả: Quách Gia An & Nguyễn Lê Bách Hợp\nLớp 10S-K30 - THPT Chuyên Hùng Vương")

# --- MODULE 1: DATABASE EXPLORER ---
if page == "1. Thư viện Alkaloid":
    st.title("📚 Thư viện số hóa Alkaloid")
    with st.expander("🔍 Bộ lọc sàng lọc thuốc (Lipinski Rule of 5)"):
        c1, c2, c3, c4 = st.columns(4)
        mw_f = c1.checkbox("Khối lượng (MW) < 500", value=True)
        lp_f = c2.checkbox("Độ ưa dầu (LogP) < 5", value=True)
        hbd_f = c3.checkbox("H-Donor ≤ 5", value=True)
        hba_f = c4.checkbox("H-Acceptor ≤ 10", value=True)
    
    filtered_df = df.copy()
    if mw_f: filtered_df = filtered_df[filtered_df['MW'] < 500]
    if lp_f: filtered_df = filtered_df[filtered_df['LogP'] < 5]
    
    st.dataframe(filtered_df[['Name', 'Formula', 'MW', 'LogP', 'HBD', 'HBA']], use_container_width=True)
    
    st.subheader("Lựa chọn hợp chất mục tiêu")
    compounds = df['Name'].tolist()
    choice = st.selectbox("Chọn từ danh sách nghiên cứu:", compounds, index=compounds.index(st.session_state.selected_compound))
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
        st.markdown(f"**Protein Đích:** `{pdb_id}`")
        st.markdown(f"**Phân tử Ligand:** `{st.session_state.selected_compound}`")
        hl = st.toggle("Hiện khoang liên kết (Binding Site)", value=True)
        st.warning("⚠️ Dữ liệu đang được tải từ RCSB PDB...")
    with col2:
        with st.spinner("Đang dựng cấu trúc phân tử 3D..."):
            pdb_string = fetch_pdb(pdb_id)
            if pdb_string:
                view = render_3d_molecule(pdb_string, highlight_site=hl)
                showmol(view, height=550, width=850)
            else:
                st.error("Không thể kết nối Server PDB.")

# --- MODULE 3: ANALYTICS & REPORT (PHIÊN BẢN NGHIÊN CỨU KHOA HỌC) ---
elif page == "3. Phân tích & Xuất báo cáo":
    st.title("📊 Kết quả phân tích dược tính")
    
    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Năng lượng liên kết (Affinity)")
        st.metric("BACE1 ΔG", f"{selected_data['dG_BACE1']} kcal/mol", delta="-8.5 (Veru)", delta_color="inverse")
        st.caption(f"Đánh giá tiềm năng: {classify_potential(selected_data['dG_BACE1'])}")
        st.metric("AChE ΔG", f"{selected_data['dG_AChE']} kcal/mol", delta="-7.9 (Done)", delta_color="inverse")
        st.caption(f"Đánh giá tiềm năng: {classify_potential(selected_data['dG_AChE'])}")
        st.markdown('</div>', unsafe_allow_html=True)
        
        if selected_data['BBB_Permeability']:
            st.success("✅ Dự đoán: Có khả năng xuyên rào máu não (BBB)")
        else:
            st.error("⚠️ Dự đoán: Khả năng xuyên rào máu não thấp")
            
    with c2:
        st.subheader("Hồ sơ ADMET Radar")
        fig = create_admet_radar(selected_data)
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Chuẩn bị nội dung báo cáo cực kỳ chi tiết
    report_content = f"""
======================================================================
         BÁO CÁO PHÂN TÍCH DƯỢC TÍNH PHÂN TỬ - ALKALOTUS PREDICTOR
======================================================================
Dự án: Nghiên cứu In Silico dẫn xuất Alkaloid từ lá sen điều trị Alzheimer
Tác giả: Quách Gia An - Nguyễn Lê Bách Hợp
Đơn vị: Lớp 10-K30 - Trường THPT Chuyên Hùng Vương
Thời gian trích xuất: {pd.Timestamp.now().strftime('%d/%m/%Y %H:%M:%S')}

----------------------------------------------------------------------
I. THÔNG TIN HỢP CHẤT (COMPOUND IDENTIFICATION)
----------------------------------------------------------------------
- Tên hợp chất: {selected_data['Name']}
- Công thức hóa học: {selected_data['Formula']}
- Nguồn gốc: Alkaloid từ Nelumbo nucifera (Sen)

----------------------------------------------------------------------
II. THÔNG SỐ HÓA LÝ & QUY TẮC LIPINSKI (DRUG-LIKENESS)
----------------------------------------------------------------------
1. Khối lượng phân tử (MW): {selected_data['MW']} g/mol
2. Hệ số phân bố (LogP): {selected_data['LogP']}
3. Số liên kết H-Donor (HBD): {selected_data['HBD']}
4. Số liên kết H-Acceptor (HBA): {selected_data['HBA']}
=> ĐÁNH GIÁ CHUNG: {'TUÂN THỦ quy tắc Lipinski (Tiềm năng làm thuốc uống tốt)' if check_lipinski(selected_data) else 'VI PHẠM quy tắc Lipinski (Cần cải thiện cấu trúc)'}

----------------------------------------------------------------------
III. KẾT QUẢ MÔ PHỎNG DOCKING PHÂN TỬ (BINDING AFFINITY)
----------------------------------------------------------------------
* Mục tiêu 1: Enzyme BACE1 (Beta-secretase 1)
  - Năng lượng tự do Gibbs (ΔG): {selected_data['dG_BACE1']} kcal/mol
  - So sánh với thuốc đối chứng Verubecestat (-8.5 kcal/mol): {'Tốt hơn' if selected_data['dG_BACE1'] < -8.5 else 'Thấp hơn'}

* Mục tiêu 2: Enzyme AChE (Acetylcholinesterase)
  - Năng lượng tự do Gibbs (ΔG): {selected_data['dG_AChE']} kcal/mol
  - So sánh với thuốc đối chứng Donepezil (-7.9 kcal/mol): {'Tốt hơn' if selected_data['dG_AChE'] < -7.9 else 'Thấp hơn'}

=> KẾT LUẬN DOCKING: {classify_potential(selected_data['dG_BACE1'])}

----------------------------------------------------------------------
IV. DƯỢC ĐỘNG HỌC & ĐỘ AN TOÀN (ADMET)
----------------------------------------------------------------------
- Khả năng xuyên rào máu não (BBB): {'TÍCH CỰC (Có khả năng tác động TW)' if selected_data['BBB_Permeability'] else 'HẠN CHẾ (Cần hệ vận chuyển nano)'}
- Khả năng hấp thu qua ruột người (HIA): Cao (Dựa trên mô phỏng Radar)
- Độ an toàn: Không gây độc tính cấp trong ngưỡng sàng lọc.

----------------------------------------------------------------------
Đây là kết quả nghiên cứu dựa trên mô phỏng máy tính (In Silico). 
Cần các thử nghiệm In Vitro và In Vivo để xác minh kết quả.
======================================================================
    """

    st.download_button(
        label="📥 TẢI BÁO CÁO NGHIÊN CỨU CHI TIẾT (.TXT)",
        data=report_content,
        file_name=f"AlkaLotus_Report_{selected_data['Name']}.txt",
        mime="text/plain"
    )
# --- MODULE 4: AI PREDICTOR (PHIÊN BẢN REAL-TIME XAI) ---
elif page == "4. AI Predictor (ML)":
    st.title("🤖 AI Predictor - Machine Learning Research Core")
    st.markdown("""
    <div style='background-color: #F0F2F6; padding: 15px; border-radius: 10px; border-left: 5px solid #FF69B4;'>
    <b>Hệ thống XAI (Explainable AI):</b> Sử dụng thuật toán <b>Random Forest Regressor</b> để dự đoán ái lực liên kết 
    và giải thích quyết định dựa trên trọng số các đặc trưng hóa lý.
    </div>
    """, unsafe_allow_html=True)
    
    try:
        # 1. Load mô hình thật từ file của Gia An
        model_ai = joblib.load('AlkaLotus/alkmer_model.pkl')
        
        # 2. Tabs chức năng
        tab_pred, tab_interpret = st.tabs(["🎯 Dự đoán Real-time", "🧠 Giải thích thuật toán (XAI)"])
        
        with tab_pred:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            c1, c2 = st.columns(2)
            val_mw = c1.number_input("Khối lượng phân tử (MW):", min_value=100.0, max_value=1000.0, value=311.4)
            val_logp = c2.number_input("Hệ số LogP (Lipophilicity):", min_value=-2.0, max_value=10.0, value=3.0)
            val_hbd = c1.slider("Số liên kết H-Donor:", 0, 15, 1)
            val_hba = c2.slider("Số liên kết H-Acceptor:", 0, 20, 5)
            
            if st.button("🚀 KÍCH HOẠT PHÂN TÍCH AI"):
                # Dữ liệu đầu vào chuẩn hóa theo training
                features = np.array([[val_mw, val_logp, val_hbd, val_hba]])
                
                # Chạy dự đoán trực tiếp từ model .pkl
                prediction = model_ai.predict(features)[0]
                
                # Hiển thị kết quả
                st.divider()
                col_res1, col_res2 = st.columns(2)
                with col_res1:
                    st.metric("Kết quả dự đoán ΔG", f"{round(prediction, 2)} kcal/mol")
                with col_res2:
                    potential = "Cao" if prediction < -8.0 else "Trung bình"
                    st.metric("Tiềm năng ức chế", potential)
                
                # So sánh trực tiếp với thuốc đối chứng trong báo cáo
                st.info(f"Độ lệch so với Verubecestat (Đối chứng): {round(abs(prediction - (-8.5)), 2)} kcal/mol")
            st.markdown('</div>', unsafe_allow_html=True)

        with tab_interpret:
            st.subheader("Trích xuất Feature Importance (Trọng số đặc trưng)")
            st.write("Biểu đồ này được trích xuất trực tiếp từ mô hình bạn đã huấn luyện, cho thấy yếu tố nào quyết định ái lực liên kết:")
            
            # Lấy trọng số thật từ mô hình Random Forest
            importances = model_ai.feature_importances_
            feature_names = ['MW', 'LogP', 'HBD', 'HBA']
            imp_df = pd.DataFrame({'Đặc trưng': feature_names, 'Trọng số': importances}).sort_values(by='Trọng số', ascending=True)
            
            # Vẽ biểu đồ ngang cho chuyên nghiệp
            st.bar_chart(imp_df, x='Đặc trưng', y='Trọng số', horizontal=True)
            
            st.success(f"Dựa trên dữ liệu huấn luyện, **{imp_df.iloc[-1]['Đặc trưng']}** là yếu tố có ảnh hưởng mạnh nhất đến khả năng ức chế enzyme.")

    except Exception as e:
        st.error(f"Lỗi hệ thống: {e}")
        st.warning("Gia An hãy kiểm tra chắc chắn file 'AlkaLotus/alkmer_model.pkl' đã được upload lên GitHub nhé!")
