import streamlit as st
import numpy as np
import trimesh
from PIL import Image
from sklearn.cluster import MiniBatchKMeans
import io

# --- CONFIGURAÇÃO ---
st.set_page_config(page_title="GLB to 3MF v3", page_icon="🎨", layout="centered")

st.markdown("""
    <style>
    .stButton>button {width: 100%; background-color: #0068c9; color: white;}
    .stApp {background-color: #0e1117; color: white;}
    </style>
""", unsafe_allow_html=True)

# --- FUNÇÃO CORRIGIDA (V3) ---
def process_glb(file_bytes, n_colors):
    file_obj = io.BytesIO(file_bytes)
    mesh = trimesh.load(file_obj, file_type='glb', force='mesh')
    
    # --- PARTE 1: DETECÇÃO DE TEXTURA (A que funcionou!) ---
    texture = None
    try:
        mat = mesh.visual.material
        if isinstance(mat, list): mat = mat[0]
            
        if hasattr(mat, 'baseColorTexture') and mat.baseColorTexture:
            texture = mat.baseColorTexture
        elif hasattr(mat, 'image') and mat.image:
            texture = mat.image
        elif isinstance(mat, dict) and 'image' in mat:
            texture = mat['image']
            
    except Exception as e:
        st.warning(f"Aviso: {e}")

    if texture is None:
        raise ValueError("ERRO: Não achei textura. O arquivo pode não ter cor.")

    # --- PARTE 2: PROCESSAMENTO ---
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

    # --- PARTE 3: CRIAÇÃO DAS PEÇAS ---
    sub_meshes = []
    for i in range(n_colors):
        mask = labels == i
        if not np.any(mask): continue
        
        part = mesh.submesh([mask], append=True)
        if isinstance(part, list): part = trimesh.util.concatenate(part)
        
        # Corrige visualização
        part.visual.face_colors = np.append(centroids[i], 255)
        part.metadata['name'] = f"Cor_{i+1}"
        sub_meshes.append(part)

    scene = trimesh.Scene(sub_meshes)

    # --- CORREÇÃO AQUI (O ERRO ESTAVA AQUI) ---
    # Usamos o método direto da cena, que retorna os bytes automaticamente
    return scene.export(file_type='3mf')

# --- INTERFACE ---
st.title("🎨 Conversor v3.0 (Export Fix)") 
st.markdown("Agora com correção de salvamento.")

uploaded_file = st.file_uploader("Arquivo GLB", type="glb")
colors = st.slider("Cores", 2, 16, 8)

if uploaded_file and st.button("Processar"):
    with st.spinner("Processando e salvando..."):
        try:
            result = process_glb(uploaded_file.getvalue(), colors)
            st.success("Sucesso! Baixe agora.")
            st.download_button("Baixar 3MF", result, "modelo_final.3mf", "model/3mf")
        except Exception as e:
            st.error(f"Erro: {e}")
