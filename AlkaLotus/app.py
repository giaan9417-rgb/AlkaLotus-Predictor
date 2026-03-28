# app.py
import streamlit as st
from stmol import showmol
from data import get_database
from utils import fetch_pdb, render_3d_molecule, check_lipinski, create_admet_radar, classify_potential

st.set_page_config(
    page_title="AlkaLotus Predictor", 
    layout="wide", 
    page_icon="🪷",
    initial_sidebar_state="expanded"
)

# --- XÓA CẢ 2 ĐOẠN CSS CŨ VÀ DÁN ĐOẠN NÀY VÀO ---
st.markdown("""
    <style>
    /* 1. Ép nền toàn bộ web thành màu trắng */
    .stApp {
        background-color: #FFFFFF !important;
    }
    
    /* 2. Ép chữ tiêu đề và chữ nội dung thành màu đen đậm để nổi bật */
    h1, h2, h3, h4, h5, h6, p, span, label, .stMarkdown {
        color: #262730 !important;
    }

    /* 3. Chỉnh cái Card thông tin cho sáng sủa hơn */
    .card {
        background-color: #F8F9FA; 
        padding: 20px; 
        border-radius: 10px; 
        margin-bottom: 20px;
        border: 1px solid #E0E0E0;
    }

    /* 4. Chỉnh màu cho các con số Metric (MW, LogP) */
    [data-testid="stMetricValue"] {
        color: #FF69B4 !important; /* Màu hồng sen cho các con số */
    }
    </style>
    """, unsafe_allow_html=True)
# Khởi tạo Session State
if 'selected_compound' not in st.session_state:
    st.session_state.selected_compound = "Roemerine"

df = get_database()

# --- SIDEBAR ---
st.sidebar.title("🪷 AlkaLotus Predictor")
st.sidebar.markdown("# app.py")
import streamlit as st
from stmol import showmol
from data import get_database
from utils import fetch_pdb, render_3d_molecule, check_lipinski, create_admet_radar, classify_potential

st.set_page_config(
    page_title="AlkaLotus Predictor", 
    layout="wide", 
    page_icon="🪷",
    initial_sidebar_state="expanded"
)

# --- XÓA CẢ 2 ĐOẠN CSS CŨ VÀ DÁN ĐOẠN NÀY VÀO ---
st.markdown("""
    <style>
    /* 1. Ép nền toàn bộ web thành màu trắng */
    .stApp {
        background-color: #FFFFFF !important;
    }
    
    /* 2. Ép chữ tiêu đề và chữ nội dung thành màu đen đậm để nổi bật */
    h1, h2, h3, h4, h5, h6, p, span, label, .stMarkdown {
        color: #262730 !important;
    }

    /* 3. Chỉnh cái Card thông tin cho sáng sủa hơn */
    .card {
        background-color: #F8F9FA; 
        padding: 20px; 
        border-radius: 10px; 
        margin-bottom: 20px;
        border: 1px solid #E0E0E0;
    }

    /* 4. Chỉnh màu cho các con số Metric (MW, LogP) */
    [data-testid="stMetricValue"] {
        color: #FF69B4 !important; /* Màu hồng sen cho các con số */
    }
    </style>
    """, unsafe_allow_html=True)
# Khởi tạo Session State
if 'selected_compound' not in st.session_state:
    st.session_state.selected_compound = "Roemerine"

df = get_database()

# --- SIDEBAR ---
st.sidebar.title("🪷 AlkaLotus Predictor")
st.sidebar.markdown("# app.py
import streamlit as st
from stmol import showmol
from data import get_database
from utils import fetch_pdb, render_3d_molecule, check_lipinski, create_admet_radar, classify_potential

st.set_page_config(
    page_title="AlkaLotus Predictor", 
    layout="wide", 
    page_icon="🪷",
    initial_sidebar_state="expanded"
)

# --- XÓA CẢ 2 ĐOẠN CSS CŨ VÀ DÁN ĐOẠN NÀY VÀO ---
st.markdown("""
    <style>
    /* 1. Ép nền toàn bộ web thành màu trắng */
    .stApp {
        background-color: #FFFFFF !important;
    }
    
    /* 2. Ép chữ tiêu đề và chữ nội dung thành màu đen đậm để nổi bật */
    h1, h2, h3, h4, h5, h6, p, span, label, .stMarkdown {
        color: #262730 !important;
    }

    /* 3. Chỉnh cái Card thông tin cho sáng sủa hơn */
    .card {
        background-color: #F8F9FA; 
        padding: 20px; 
        border-radius: 10px; 
        margin-bottom: 20px;
        border: 1px solid #E0E0E0;
    }

    /* 4. Chỉnh màu cho các con số Metric (MW, LogP) */
    [data-testid="stMetricValue"] {
        color: #FF69B4 !important; /* Màu hồng sen cho các con số */
    }
    </style>
    """, unsafe_allow_html=True)
# Khởi tạo Session State
if 'selected_compound' not in st.session_state:
    st.session_state.selected_compound = "Roemerine"

df = get_database()

# --- SIDEBAR ---
st.sidebar.title("🪷 AlkaLotus Predictor")
st.sidebar.markdown("HỆ THỐNG HÓA KẾT QUẢ NGHIÊN CỨU IN SILICO KHẢ NĂNG ỨC CHẾ ENZYME AChE VÀ BACE1 TỪ CÁC DẪN XUẤT ALKALOID TỪ LÁ SEN TRONG ĐỊNH HƯỚNG HỖ TRỢ ĐIỀU TRỊ BỆNH ALZHEIMER")
st.sidebar.markdown("---")
page = st.sidebar.radio("Điều hướng", ["1. Database Explorer", "2. Virtual Docking Lab", "3. Analytics Dashboard"])

st.sidebar.markdown("---")
st.sidebar.info("Hợp chất đang chọn:\n**{}**".format(st.session_state.selected_compound))

selected_data = df[df['Name'] == st.session_state.selected_compound].iloc[0]

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
    
    # Lựa chọn hợp chất
    st.subheader("Chọn hợp chất để phân tích")
    compounds = df['Name'].tolist()
    choice = st.selectbox("Chọn Alkaloid:", compounds, index=compounds.index(st.session_state.selected_compound))
    
    if choice != st.session_state.selected_compound:
        st.session_state.selected_compound = choice
        st.rerun()

    # Card thông tin
    st.markdown('<div class="card">', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    c1.metric("Khối lượng phân tử (MW)", f"{selected_data['MW']} g/mol")
    c2.metric("Độ ưa dầu (LogP)", selected_data['LogP'])
    c3.metric("Số vi phạm Lipinski", "0" if check_lipinski(selected_data) else "≥2")
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
        highlight = st.toggle("Highlight Binding Site", value=False)
        
        st.markdown("---")
        st.warning("Đang tải dữ liệu tinh thể trực tiếp từ PDB API. Vui lòng đợi trong giây lát.")
        
    with col2:
        with st.spinner("Đang render môi trường 3D..."):
            pdb_string = fetch_pdb(pdb_id)
            if pdb_string:
                view = render_3d_molecule(pdb_string, highlight_site=highlight)
                showmol(view, height=500, width=800)
            else:
                st.error("Không thể kết nối đến PDB Server.")

# --- MODULE 3: ANALYTICS DASHBOARD ---
elif page == "3. Analytics Dashboard":
    st.title("📊 Analytics Dashboard")
    st.markdown(f"Phân tích chuyên sâu cho **{st.session_state.selected_compound}**")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Năng lượng liên kết (ΔG)")
        
        # BACE1 Comparison
        bace_dg = selected_data['dG_BACE1']
        st.metric("BACE1 Affinity", f"{bace_dg} kcal/mol", 
                  delta=f"{round(bace_dg - (-8.5), 2)} vs Verubecestat", delta_color="inverse")
        st.caption(classify_potential(bace_dg))
        
        # AChE Comparison
        ache_dg = selected_data['dG_AChE']
        st.metric("AChE Affinity", f"{ache_dg} kcal/mol", 
                  delta=f"{round(ache_dg - (-7.9), 2)} vs Donepezil", delta_color="inverse")
        st.caption(classify_potential(ache_dg))
        
        st.markdown("---")
        st.subheader("Cảnh báo Dược lý")
        if selected_data['BBB_Permeability']:
            st.success("✅ Khả năng qua hàng rào máu não (BBB): TỐT")
        else:
            st.error("⚠️ Khả năng qua hàng rào máu não (BBB): KÉM (Cần hệ vận chuyển Nano)")
            
    with col2:
        st.subheader("Hồ sơ ADMET (Mô phỏng)")
        fig = create_admet_radar(selected_data)
        st.plotly_chart(fig, use_container_width=True)
        
    if st.button("📄 Export Report (TXT)"):
        report = f"""
        BÁO CÁO PHÂN TÍCH IN SILICO - ALKALOTUS PREDICTOR
        --------------------------------------------------
        Hợp chất: {selected_data['Name']}
        Công thức: {selected_data['Formula']}
        
        [1] Đặc tính hóa lý:
        - MW: {selected_data['MW']}
        - LogP: {selected_data['LogP']}
        
        [2] Tiềm năng ức chế (Docking):
        - BACE1 (4XXS): {selected_data['dG_BACE1']} kcal/mol ({classify_potential(selected_data['dG_BACE1'])})
        - AChE (7D9O): {selected_data['dG_AChE']} kcal/mol ({classify_potential(selected_data['dG_AChE'])})
        
        [3] Cảnh báo hệ thần kinh trung ương:
        - Xuyên rào cản máu não (BBB): {'Có' if selected_data['BBB_Permeability'] else 'Cần hệ Nano'}
        """
        st.download_button("Tải xuống Báo cáo", report, file_name=f"{selected_data['Name']}_Report.txt")
")
st.sidebar.markdown("---")
page = st.sidebar.radio("Điều hướng", ["1. Database Explorer", "2. Virtual Docking Lab", "3. Analytics Dashboard"])

st.sidebar.markdown("---")
st.sidebar.info("Hợp chất đang chọn:\n**{}**".format(st.session_state.selected_compound))

selected_data = df[df['Name'] == st.session_state.selected_compound].iloc[0]

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
    
    # Lựa chọn hợp chất
    st.subheader("Chọn hợp chất để phân tích")
    compounds = df['Name'].tolist()
    choice = st.selectbox("Chọn Alkaloid:", compounds, index=compounds.index(st.session_state.selected_compound))
    
    if choice != st.session_state.selected_compound:
        st.session_state.selected_compound = choice
        st.rerun()

    # Card thông tin
    st.markdown('<div class="card">', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    c1.metric("Khối lượng phân tử (MW)", f"{selected_data['MW']} g/mol")
    c2.metric("Độ ưa dầu (LogP)", selected_data['LogP'])
    c3.metric("Số vi phạm Lipinski", "0" if check_lipinski(selected_data) else "≥2")
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
        highlight = st.toggle("Highlight Binding Site", value=False)
        
        st.markdown("---")
        st.warning("Đang tải dữ liệu tinh thể trực tiếp từ PDB API. Vui lòng đợi trong giây lát.")
        
    with col2:
        with st.spinner("Đang render môi trường 3D..."):
            pdb_string = fetch_pdb(pdb_id)
            if pdb_string:
                view = render_3d_molecule(pdb_string, highlight_site=highlight)
                showmol(view, height=500, width=800)
            else:
                st.error("Không thể kết nối đến PDB Server.")

# --- MODULE 3: ANALYTICS DASHBOARD ---
elif page == "3. Analytics Dashboard":
    st.title("📊 Analytics Dashboard")
    st.markdown(f"Phân tích chuyên sâu cho **{st.session_state.selected_compound}**")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Năng lượng liên kết (ΔG)")
        
        # BACE1 Comparison
        bace_dg = selected_data['dG_BACE1']
        st.metric("BACE1 Affinity", f"{bace_dg} kcal/mol", 
                  delta=f"{round(bace_dg - (-8.5), 2)} vs Verubecestat", delta_color="inverse")
        st.caption(classify_potential(bace_dg))
        
        # AChE Comparison
        ache_dg = selected_data['dG_AChE']
        st.metric("AChE Affinity", f"{ache_dg} kcal/mol", 
                  delta=f"{round(ache_dg - (-7.9), 2)} vs Donepezil", delta_color="inverse")
        st.caption(classify_potential(ache_dg))
        
        st.markdown("---")
        st.subheader("Cảnh báo Dược lý")
        if selected_data['BBB_Permeability']:
            st.success("✅ Khả năng qua hàng rào máu não (BBB): TỐT")
        else:
            st.error("⚠️ Khả năng qua hàng rào máu não (BBB): KÉM (Cần hệ vận chuyển Nano)")
            
    with col2:
        st.subheader("Hồ sơ ADMET (Mô phỏng)")
        fig = create_admet_radar(selected_data)
        st.plotly_chart(fig, use_container_width=True)
        
    if st.button("📄 Export Report (TXT)"):
        report = f"""
        BÁO CÁO PHÂN TÍCH IN SILICO - ALKALOTUS PREDICTOR
        --------------------------------------------------
        Hợp chất: {selected_data['Name']}
        Công thức: {selected_data['Formula']}
        
        [1] Đặc tính hóa lý:
        - MW: {selected_data['MW']}
        - LogP: {selected_data['LogP']}
        
        [2] Tiềm năng ức chế (Docking):
        - BACE1 (4XXS): {selected_data['dG_BACE1']} kcal/mol ({classify_potential(selected_data['dG_BACE1'])})
        - AChE (7D9O): {selected_data['dG_AChE']} kcal/mol ({classify_potential(selected_data['dG_AChE'])})
        
        [3] Cảnh báo hệ thần kinh trung ương:
        - Xuyên rào cản máu não (BBB): {'Có' if selected_data['BBB_Permeability'] else 'Cần hệ Nano'}
        """
        st.download_button("Tải xuống Báo cáo", report, file_name=f"{selected_data['Name']}_Report.txt")
")
st.sidebar.markdown("---")
page = st.sidebar.radio("Điều hướng", ["1. Database Explorer", "2. Virtual Docking Lab", "3. Analytics Dashboard"])

st.sidebar.markdown("---")
st.sidebar.info("Hợp chất đang chọn:\n**{}**".format(st.session_state.selected_compound))

selected_data = df[df['Name'] == st.session_state.selected_compound].iloc[0]

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
    
    # Lựa chọn hợp chất
    st.subheader("Chọn hợp chất để phân tích")
    compounds = df['Name'].tolist()
    choice = st.selectbox("Chọn Alkaloid:", compounds, index=compounds.index(st.session_state.selected_compound))
    
    if choice != st.session_state.selected_compound:
        st.session_state.selected_compound = choice
        st.rerun()

    # Card thông tin
    st.markdown('<div class="card">', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    c1.metric("Khối lượng phân tử (MW)", f"{selected_data['MW']} g/mol")
    c2.metric("Độ ưa dầu (LogP)", selected_data['LogP'])
    c3.metric("Số vi phạm Lipinski", "0" if check_lipinski(selected_data) else "≥2")
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
        highlight = st.toggle("Highlight Binding Site", value=False)
        
        st.markdown("---")
        st.warning("Đang tải dữ liệu tinh thể trực tiếp từ PDB API. Vui lòng đợi trong giây lát.")
        
    with col2:
        with st.spinner("Đang render môi trường 3D..."):
            pdb_string = fetch_pdb(pdb_id)
            if pdb_string:
                view = render_3d_molecule(pdb_string, highlight_site=highlight)
                showmol(view, height=500, width=800)
            else:
                st.error("Không thể kết nối đến PDB Server.")

# --- MODULE 3: ANALYTICS DASHBOARD ---
elif page == "3. Analytics Dashboard":
    st.title("📊 Analytics Dashboard")
    st.markdown(f"Phân tích chuyên sâu cho **{st.session_state.selected_compound}**")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Năng lượng liên kết (ΔG)")
        
        # BACE1 Comparison
        bace_dg = selected_data['dG_BACE1']
        st.metric("BACE1 Affinity", f"{bace_dg} kcal/mol", 
                  delta=f"{round(bace_dg - (-8.5), 2)} vs Verubecestat", delta_color="inverse")
        st.caption(classify_potential(bace_dg))
        
        # AChE Comparison
        ache_dg = selected_data['dG_AChE']
        st.metric("AChE Affinity", f"{ache_dg} kcal/mol", 
                  delta=f"{round(ache_dg - (-7.9), 2)} vs Donepezil", delta_color="inverse")
        st.caption(classify_potential(ache_dg))
        
        st.markdown("---")
        st.subheader("Cảnh báo Dược lý")
        if selected_data['BBB_Permeability']:
            st.success("✅ Khả năng qua hàng rào máu não (BBB): TỐT")
        else:
            st.error("⚠️ Khả năng qua hàng rào máu não (BBB): KÉM (Cần hệ vận chuyển Nano)")
            
    with col2:
        st.subheader("Hồ sơ ADMET (Mô phỏng)")
        fig = create_admet_radar(selected_data)
        st.plotly_chart(fig, use_container_width=True)
        
    if st.button("📄 Export Report (TXT)"):
        report = f"""
        BÁO CÁO PHÂN TÍCH IN SILICO - ALKALOTUS PREDICTOR
        --------------------------------------------------
        Hợp chất: {selected_data['Name']}
        Công thức: {selected_data['Formula']}
        
        [1] Đặc tính hóa lý:
        - MW: {selected_data['MW']}
        - LogP: {selected_data['LogP']}
        
        [2] Tiềm năng ức chế (Docking):
        - BACE1 (4XXS): {selected_data['dG_BACE1']} kcal/mol ({classify_potential(selected_data['dG_BACE1'])})
        - AChE (7D9O): {selected_data['dG_AChE']} kcal/mol ({classify_potential(selected_data['dG_AChE'])})
        
        [3] Cảnh báo hệ thần kinh trung ương:
        - Xuyên rào cản máu não (BBB): {'Có' if selected_data['BBB_Permeability'] else 'Cần hệ Nano'}
        """
        st.download_button("Tải xuống Báo cáo", report, file_name=f"{selected_data['Name']}_Report.txt")
