import streamlit as st
import numpy as np
import trimesh
from PIL import Image
from sklearn.cluster import MiniBatchKMeans
import io

# --- CONFIGURAÇÃO ---
st.set_page_config(page_title="GLB to 3MF v6 (Fixed)", page_icon="🎨", layout="wide")

st.markdown("""
    <style>
    .stButton>button {width: 100%; background-color: #0068c9; color: white;}
    .stApp {background-color: #0e1117; color: white;}
    .color-box {
        width: 100%;
        height: 34px;
        border-radius: 5px;
        border: 1px solid #ffffff50;
    }
    </style>
""", unsafe_allow_html=True)

# --- FUNÇÃO PRINCIPAL ---
def process_glb(file_bytes, n_colors):
    file_obj = io.BytesIO(file_bytes)
    mesh = trimesh.load(file_obj, file_type='glb', force='mesh')
    
    # Detecção de Textura
    texture = None
    try:
        mat = mesh.visual.material
        if isinstance(mat, list): mat = mat[0]
        if hasattr(mat, 'baseColorTexture') and mat.baseColorTexture: texture = mat.baseColorTexture
        elif hasattr(mat, 'image') and mat.image: texture = mat.image
        elif isinstance(mat, dict) and 'image' in mat: texture = mat['image']
    except Exception: pass

    if texture is None: raise ValueError("Sem textura encontrada.")

    # Processamento
    if texture.mode != 'RGB': texture = texture.convert('RGB')
    tex_array = np.array(texture)
    h, w, _ = tex_array.shape

    uvs = mesh.visual.uv
    faces = mesh.faces
    face_uvs = uvs[faces].mean(axis=1)
    
    u = (face_uvs[:, 0] * (w - 1)).astype(int).clip(0, w - 1)
    v = ((1 - face_uvs[:, 1]) * (h - 1)).astype(int).clip(0, h - 1)
    face_colors = tex_array[v, u]

    kmeans = MiniBatchKMeans(n_clusters=n_colors, batch_size=4096, n_init=3)
    labels = kmeans.fit_predict(face_colors)
    centroids = kmeans.cluster_centers_.astype(int)

    # Geração das Partes
    sub_meshes = []
    palette_data = []

    for i in range(n_colors):
        mask = labels == i
        if not np.any(mask): continue
        
        part = mesh.submesh([mask], append=True)
        if isinstance(part, list): part = trimesh.util.concatenate(part)
        
        part_name = f"Cor_{i+1}"
        part.metadata['name'] = part_name
        part.visual.face_colors = np.append(centroids[i], 255)
        sub_meshes.append(part)
        
        # RGB String
        r, g, b = centroids[i]
        rgb_string = f"{r},{g},{b}" 
        hex_color = '#{:02x}{:02x}{:02x}'.format(r, g, b)
        
        palette_data.append({
            "name": part_name,
            "rgb": rgb_string,
            "hex": hex_color
        })

    scene = trimesh.Scene(sub_meshes)
    return scene.export(file_type='3mf'), palette_data

# --- INTERFACE ---
st.title("🎨 Conversor v6.0 (Persistente)") 

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("1. Arquivo e Configuração")
    uploaded_file = st.file_uploader("Arquivo GLB", type="glb")
    colors = st.slider("Quantidade de Cores", 2, 16, 8)
    
    # Botão de processar
    if st.button("Processar Cores"):
        if uploaded_file:
            with st.spinner("Analisando cores..."):
                try:
                    # Processa e SALVA NA MEMÓRIA (Session State)
                    file_data, palette = process_glb(uploaded_file.getvalue(), colors)
                    st.session_state['3mf_data'] = file_data
                    st.session_state['palette_data'] = palette
                    st.session_state['processed'] = True
                except Exception as e:
                    st.error(f"Erro: {e}")

# --- EXIBIÇÃO DOS RESULTADOS (Lê da Memória) ---
if 'processed' in st.session_state and st.session_state['processed']:
    
    # Coluna 1: Download
    with col1:
        st.success("Processado com sucesso!")
        st.download_button(
            label="⬇️ Baixar 3MF", 
            data=st.session_state['3mf_data'], 
            file_name="modelo_rgb.3mf", 
            mime="model/3mf"
        )

    # Coluna 2: Gabarito
    with col2:
        st.subheader("2. Gabarito RGB")
        st.info("Copie os códigos e cole no Orca Slicer.")
        
        for p in st.session_state['palette_data']:
            c1, c2, c3 = st.columns([1, 2, 3])
            with c1:
                st.markdown(f'<div class="color-box" style="background-color: {p["hex"]};"></div>', unsafe_allow_html=True)
            with c2:
                st.write(f"**{p['name']}**")
            with c3:
                st.code(p['rgb'], language="text")
