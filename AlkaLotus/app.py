import streamlit as st
from stmol import showmol
from data import get_database
from utils import fetch_pdb, render_3d_molecule, check_lipinski, create_admet_radar, classify_potential

# 1. Cấu hình trang (Chỉ đặt 1 lần duy nhất ở đầu file)
st.set_page_config(
    page_title="AlkaLotus Predictor", 
    layout="wide", 
    page_icon="🪷",
    initial_sidebar_state="expanded"
)

# 2. CSS để ép giao diện TRẮNG SÁNG và CHỮ ĐEN nổi bật
st.markdown("""
    <style>
    .stApp { background-color: #FFFFFF !important; }
    h1, h2, h3, h4, h5, h6, p, span, label, .stMarkdown { color: #262730 !important; }
    .card {
        background-color: #F8F9FA; 
        padding: 20px; 
        border-radius: 10px; 
        margin-bottom: 20px;
        border: 1px solid #E0E0E0;
    }
    [data-testid="stMetricValue"] { color: #FF69B4 !important; }
    /* Sửa màu chữ trong ô chọn (selectbox) */
    .stSelectbox div[data-baseweb="select"] { color: #262730 !important; }
    </style>
    """, unsafe_allow_html=True)

# 3. Khởi tạo dữ liệu
if 'selected_compound' not in st.session_state:
    st.session_state.selected_compound = "Roemerine"

df = get_database()
selected_data = df[df['Name'] == st.session_state.selected_compound].iloc[0]

# 4. SIDEBAR (Đã dọn dẹp phần chữ bị lặp)
st.sidebar.title("🪷 AlkaLotus Predictor")
st.sidebar.caption("Dự án nghiên cứu Alkaloid từ Lá Sen hỗ trợ điều trị Alzheimer")
st.sidebar.markdown("---")

page = st.sidebar.radio("Điều hướng hệ thống", 
    ["1. Database Explorer", "2. Virtual Docking Lab", "3. Analytics Dashboard"])

st.sidebar.markdown("---")
st.sidebar.write(f"🧬 **Hợp chất mục tiêu:** \n### {st.session_state.selected_compound}")
st.sidebar.info(f"Công thức: {selected_data['Formula']}")

# --- MODULE 1: DATABASE EXPLORER ---
if page == "1. Database Explorer":
    st.title("📚 Thư viện Alkaloid Lá Sen")
    
    # Bộ lọc Lipinski
    st.subheader("Bộ lọc Lipinski Rule of 5")
    col1, col2, col3, col4 = st.columns(4)
    with col1: mw_filter = st.checkbox("MW < 500", value=True)
    with col2: logp_filter = st.checkbox("LogP < 5", value=True)
    with col3: hbd_filter = st.checkbox("HBD ≤ 5", value=True)
    with col4: hba_filter = st.checkbox("HBA ≤ 10", value=True)
    
    filtered_df = df.copy()
    if mw_filter: filtered_df = filtered_df[filtered_df['MW'] < 500]
    if logp_filter: filtered_df = filtered_df[filtered_df['LogP'] < 5]
    if hbd_filter: filtered_df = filtered_df[filtered_df['HBD'] <= 5]
    if hba_filter: filtered_df = filtered_df[filtered_df['HBA'] <= 10]
    
    st.dataframe(filtered_df[['Name', 'Formula', 'MW', 'LogP', 'HBD', 'HBA']], use_container_width=True)
    
    # Lựa chọn hợp chất & Xóa ô trắng thừa
    st.subheader("Phân tích chi tiết hợp chất")
    compounds = df['Name'].tolist()
    
    # Dùng columns để lấp đầy khoảng trống
    c_sel, c_empty = st.columns([2, 1])
    with c_sel:
        choice = st.selectbox("Chọn Alkaloid để xem đặc tính:", compounds, index=compounds.index(st.session_state.selected_compound))
    
    if choice != st.session_state.selected_compound:
        st.session_state.selected_compound = choice
        st.rerun()

    # Card thông tin
    st.markdown('<div class="card">', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    c1.metric("Khối lượng (MW)", f"{selected_data['MW']} g/mol")
    c2.metric("Độ ưa dầu (LogP)", selected_data['LogP'])
    c3.metric("Vi phạm Lipinski", "0" if check_lipinski(selected_data) else "≥2")
    st.markdown('</div>', unsafe_allow_html=True)

# --- MODULE 2: VIRTUAL DOCKING LAB ---
elif page == "2. Virtual Docking Lab":
    st.title("🔬 Virtual Docking Lab")
    target = st.radio("Chọn Protein Mục tiêu:", ["BACE1 (4XXS)", "AChE (7D9O)"], horizontal=True)
    pdb_id = "4XXS" if "BACE1" in target else "7D9O"
    
    col1, col2 = st.columns([1, 3])
    with col1:
        st.markdown(f"**Protein:** {pdb_id}")
        st.markdown(f"**Ligand:** {st.session_state.selected_compound}")
        highlight = st.toggle("Hiện khoang liên kết", value=False)
        st.warning("Đang kết nối PDB API...")
        
    with col2:
        with st.spinner("Đang render cấu trúc 3D..."):
            pdb_string = fetch_pdb(pdb_id)
            if pdb_string:
                view = render_3d_molecule(pdb_string, highlight_site=highlight)
                showmol(view, height=500, width=800)
            else:
                st.error("Lỗi kết nối Server PDB.")

# --- MODULE 3: ANALYTICS DASHBOARD ---
elif page == "3. Analytics Dashboard":
    st.title("📊 Analytics Dashboard")
    st.markdown(f"Kết quả mô phỏng cho **{st.session_state.selected_compound}**")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Năng lượng liên kết (ΔG)")
        bace_dg = selected_data['dG_BACE1']
        st.metric("BACE1 Affinity", f"{bace_dg} kcal/mol", delta_color="inverse")
        st.caption(f"Đánh giá: {classify_potential(bace_dg)}")
        
        ache_dg = selected_data['dG_AChE']
        st.metric("AChE Affinity", f"{ache_dg} kcal/mol", delta_color="inverse")
        st.caption(f"Đánh giá: {classify_potential(ache_dg)}")
            
    with col2:
        st.subheader("Hồ sơ ADMET")
        fig = create_admet_radar(selected_data)
        st.plotly_chart(fig, use_container_width=True)
        
    if st.button("📄 Xuất báo cáo nghiên cứu"):
        report = f"Hợp chất: {selected_data['Name']}\nMW: {selected_data['MW']}\ndG BACE1: {bace_dg}"
        st.download_button("Tải File TXT", report, file_name=f"ALKMER_{selected_data['Name']}.txt")
