# utils.py
import requests
import py3Dmol
import plotly.graph_objects as go
import streamlit as st

@st.cache_data
def fetch_pdb(pdb_id):
    """Tải cấu trúc protein từ RCSB PDB"""
    url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
    response = requests.get(url)
    if response.status_code == 200:
        return response.text
    return None

def render_3d_molecule(pdb_data, highlight_site=False):
    """Render 3D viewer sử dụng py3Dmol"""
    view = py3Dmol.view(width=800, height=500)
    view.addModel(pdb_data, "pdb")
    view.setStyle({'cartoon': {'color': 'spectrum'}})
    
    if highlight_site:
        # Highlight mô phỏng túi hoạt động (Active site)
        view.addSurface(py3Dmol.VDW, {'opacity': 0.7, 'color': 'white'}, {'resi': ['32', '228', '286']})
        view.setStyle({'resi': ['32', '228', '286']}, {'stick': {'colorscheme': 'greenCarbon'}})
        
    view.zoomTo()
    return view

def check_lipinski(row):
    """Kiểm tra quy tắc Lipinski Rule of 5"""
    mw_pass = row['MW'] < 500
    logp_pass = row['LogP'] < 5
    hbd_pass = row['HBD'] <= 5
    hba_pass = row['HBA'] <= 10
    violations = 4 - sum([mw_pass, logp_pass, hbd_pass, hba_pass])
    return violations <= 1 # Chấp nhận tối đa 1 vi phạm

def create_admet_radar(compound_data):
    """Vẽ biểu đồ Radar ADMET"""
    categories = ['Absorption', 'Distribution', 'Metabolism', 'Excretion', 'Toxicity']
    # Dữ liệu mô phỏng chuẩn hóa (0-10) dựa trên MW và LogP
    score = 10 - (compound_data['MW']/100) 
    
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=[score, score*0.8 if not compound_data['BBB_Permeability'] else score, 7, 8, 9 if compound_data['LogP'] < 5 else 4],
        theta=categories,
        fill='toself',
        name=compound_data['Name'],
        line_color='#00ff9d'
    ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 10])),
        showlegend=False,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white')
    )
    return fig

def classify_potential(dg):
    """Phân loại tiềm năng ức chế"""
    if dg <= -9.0:
        return "High Potential 🌟"
    elif -9.0 < dg <= -7.0:
        return "Medium Potential ⚡"
    else:
        return "Low Potential 📉"