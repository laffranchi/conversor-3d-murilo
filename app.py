import streamlit as st
import numpy as np
import trimesh
from PIL import Image
from sklearn.cluster import MiniBatchKMeans
import io

# --- CONFIGURAÇÃO DA PÁGINA (TÍTULO E ÍCONE) ---
st.set_page_config(page_title="GLB to Color 3MF", page_icon="🎨", layout="centered")

# --- ESTILO CSS (PARA FICAR BONITO) ---
st.markdown("""
    <style>
    .stButton>button {
        width: 100%;
        background-color: #0068c9;
        color: white;
        border-radius: 10px;
    }
    .stApp {
        background-color: #0e1117;
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

# --- FUNÇÃO DE PROCESSAMENTO (O CÉREBRO) ---
def process_glb(file_bytes, n_colors):
    file_obj = io.BytesIO(file_bytes)
    # Força o carregamento como Scene ou Mesh
    mesh = trimesh.load(file_obj, file_type='glb', force='mesh')
    
    if not hasattr(mesh.visual, 'material') or mesh.visual.material.image is None:
        raise ValueError("O arquivo não tem textura de imagem válida.")

    # Pega a textura
    texture = mesh.visual.material.image
    if texture.mode != 'RGB': texture = texture.convert('RGB')
    tex_array = np.array(texture)
    h, w, _ = tex_array.shape

    # Pega os UVs e Faces
    uvs = mesh.visual.uv
    faces = mesh.faces
    
    # Mapeia cada face para um pixel da textura
    face_uvs = uvs[faces].mean(axis=1)
    u = (face_uvs[:, 0] * (w - 1)).astype(int).clip(0, w - 1)
    v = ((1 - face_uvs[:, 1]) * (h - 1)).astype(int).clip(0, h - 1)
    
    face_colors = tex_array[v, u]

    # Agrupa as cores (K-Means)
    kmeans = MiniBatchKMeans(n_clusters=n_colors, batch_size=4096, n_init=3)
    labels = kmeans.fit_predict(face_colors)
    centroids = kmeans.cluster_centers_.astype(int)

    # Cria as sub-malhas
    sub_meshes = []
    for i in range(n_colors):
        mask = labels == i
        if not np.any(mask): continue
        
        # Cria a peça separada
        part = mesh.submesh([mask], append=True)
        if isinstance(part, list): part = trimesh.util.concatenate(part)
        
        # Pinta a peça para visualização
        part.visual.face_colors = np.append(centroids[i], 255)
        part.metadata['name'] = f"Cor_{i+1}"
        sub_meshes.append(part)

    scene = trimesh.Scene(sub_meshes)
    return trimesh.exchange.export.export_mesh(scene, file_type='3mf')

# --- A INTERFACE VISUAL ---
st.title("🎨 Conversor GLB para 3MF Multicolor")
st.markdown("Converta texturas em peças segmentadas para **Orca Slicer**.")

uploaded_file = st.file_uploader("Arraste seu arquivo GLB aqui", type="glb")
colors = st.slider("Quantidade de Cores", 2, 16, 4)

if uploaded_file and st.button("Processar Cores"):
    with st.spinner("Analisando geometria e texturas... (Isso pode levar alguns segundos)"):
        try:
            result = process_glb(uploaded_file.getvalue(), colors)
            st.success("Conversão concluída! Baixe seu arquivo abaixo.")
            
            st.download_button(
                label="⬇️ Baixar Arquivo 3MF Pronto",
                data=result,
                file_name="modelo_colorido.3mf",
                mime="model/3mf"
            )
        except Exception as e:
            st.error(f"Erro ao processar: {e}")