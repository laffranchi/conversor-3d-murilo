import streamlit as st
import numpy as np
import trimesh
from sklearn.cluster import MiniBatchKMeans
import io
import base64

# --- CONFIGURAÇÃO DA PÁGINA (WIDE MODE) ---
st.set_page_config(page_title="Color 3MF Converter v8", page_icon="🎨", layout="wide")

# --- CSS PRO (DESIGN MODERNO) ---
st.markdown("""
    <style>
    /* Fundo e Fontes */
    .stApp {
        background-color: #0e1117;
    }
    h1, h2, h3 {
        font-family: 'Helvetica Neue', sans-serif;
        font-weight: 700;
        color: #fafafa;
    }
    
    /* Card do Gabarito */
    .palette-card {
        background-color: #262730;
        padding: 12px;
        border-radius: 10px;
        margin-bottom: 8px;
        border: 1px solid #41444e;
        display: flex;
        align-items: center;
        justify-content: space-between;
    }
    
    /* A caixa de cor */
    .color-preview {
        width: 35px;
        height: 35px;
        border-radius: 8px;
        border: 2px solid #ffffff50;
        box-shadow: 0 4px 6px rgba(0,0,0,0.2);
    }
    
    /* Botões */
    .stButton>button {
        background: linear-gradient(90deg, #00C9FF 0%, #92FE9D 100%);
        color: #000;
        font-weight: bold;
        border: none;
        height: 50px;
        border-radius: 12px;
        transition: transform 0.2s;
    }
    .stButton>button:hover {
        transform: scale(1.02);
        color: #000;
    }
    /* Ajuste do viewer para centralizar */
    iframe { display: block; margin: auto; }
    </style>
""", unsafe_allow_html=True)

# --- FUNÇÃO DO VIEWER 3D (HTML/JS) ---
def render_3d_viewer(file_bytes):
    """Gera um visualizador 3D interativo usando model-viewer do Google"""
    b64 = base64.b64encode(file_bytes).decode()
    viewer_html = f"""
        <script type="module" src="https://ajax.googleapis.com/ajax/libs/model-viewer/3.1.1/model-viewer.min.js"></script>
        <model-viewer
            src="data:model/gltf-binary;base64,{b64}"
            camera-controls
            auto-rotate
            shadow-intensity="1"
            style="width: 100%; height: 450px; background-color: #161920; border-radius: 15px; box-shadow: inset 0 0 20px rgba(0,0,0,0.5);"
            ar>
        </model-viewer>
    """
    return viewer_html

# --- FUNÇÃO DE PROCESSAMENTO PRINCIPAL ---
def process_glb(file_bytes, n_colors):
    file_obj = io.BytesIO(file_bytes)
    mesh = trimesh.load(file_obj, file_type='glb', force='mesh')
    
    # 1. Busca Textura
    texture = None
    try:
        mat = mesh.visual.material
        if isinstance(mat, list): mat = mat[0]
        if hasattr(mat, 'baseColorTexture') and mat.baseColorTexture: texture = mat.baseColorTexture
        elif hasattr(mat, 'image') and mat.image: texture = mat.image
        elif isinstance(mat, dict) and 'image' in mat: texture = mat['image']
    except Exception: pass

    if texture is None: raise ValueError("O arquivo não possui texturas suportadas.")

    # 2. Processa Cores e UVs
    if texture.mode != 'RGB': texture = texture.convert('RGB')
    tex_array = np.array(texture)
    h, w, _ = tex_array.shape

    uvs = mesh.visual.uv
    faces = mesh.faces
    face_uvs = uvs[faces].mean(axis=1)
    
    u = (face_uvs[:, 0] * (w - 1)).astype(int).clip(0, w - 1)
    v = ((1 - face_uvs[:, 1]) * (h - 1)).astype(int).clip(0, h - 1)
    face_colors = tex_array[v, u]

    # 3. K-Means (Redução de cores)
    kmeans = MiniBatchKMeans(n_clusters=n_colors, batch_size=4096, n_init=3)
    labels = kmeans.fit_predict(face_colors)
    centroids = kmeans.cluster_centers_.astype(int)

    # 4. Segmentação e Criação da Cena
    sub_meshes = []
    palette_data = []

    for i in range(n_colors):
        mask = labels == i
        if not np.any(mask): continue
        
        part = mesh.submesh([mask], append=True)
        if isinstance(part, list): part = trimesh.util.concatenate(part)
        
        part_name = f"Cor_{i+1}"
        part.metadata['name'] = part_name
        # Pinta a malha com a cor sólida para o preview
        part.visual.face_colors = np.append(centroids[i], 255)
        sub_meshes.append(part)
        
        r, g, b = centroids[i]
        palette_data.append({
            "name": part_name,
            "rgb": f"{r},{g},{b}",
            "hex": '#{:02x}{:02x}{:02x}'.format(r, g, b)
        })

    scene = trimesh.Scene(sub_meshes)
    
    # --- NOVO: GERA DOIS ARQUIVOS ---
    # 1. O 3MF para o Orca Slicer
    export_3mf = scene.export(file_type='3mf')
    # 2. Um GLB colorido apenas para o preview no navegador
    export_glb_preview = scene.export(file_type='glb')

    return export_3mf, export_glb_preview, palette_data

# --- LAYOUT DA INTERFACE ---

# 1. BARRA LATERAL (CONTROLES)
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/5219/5219462.png", width=60)
    st.title("Configurações")
    st.markdown("---")
    
    uploaded_file = st.file_uploader("📂 Carregar GLB Original", type="glb")
    
    if uploaded_file:
        st.markdown("### 🎨 Paleta Desejada")
        colors = st.slider("Número de Cores", 2, 16, 8)
        st.info(f"O modelo será reduzido para {colors} cores.")
        process_btn = st.button("🚀 Processar e Gerar Preview")
    else:
        st.warning("Faça upload de um arquivo para começar.")
        process_btn = False

# 2. ÁREA PRINCIPAL
st.title("Conversor Multicolor 3D Pro")

# Lógica de Estado e Processamento
if process_btn and uploaded_file:
    with st.spinner("Segmentando malha, aplicando cores e gerando preview..."):
        try:
            # Processa e recebe OS TRÊS dados
            file_3mf, preview_glb, palette = process_glb(uploaded_file.getvalue(), colors)
            
            # Salva na memória
            st.session_state['3mf_data'] = file_3mf
            st.session_state['preview_data'] = preview_glb # Guarda o preview colorido
            st.session_state['palette_data'] = palette
            st.session_state['processed'] = True
            st.session_state['input_bytes'] = uploaded_file.getvalue() # Guarda o original
            
        except Exception as e:
            st.error(f"Erro no processamento: {e}")

# --- EXIBIÇÃO DOS VIEWS ---

# CENÁRIO A: Arquivo carregado, mas não processado (Mostra Original)
if uploaded_file and not st.session_state.get('processed'):
    st.subheader("👁️ Pré-visualização do Arquivo Original")
    st.markdown("Este é o arquivo texturizado que você enviou. Configure as cores ao lado e clique em processar.")
    st.components.v1.html(render_3d_viewer(uploaded_file.getvalue()), height=470)

# CENÁRIO B: Processado com sucesso (Mostra Resultado + Download + Gabarito)
if st.session_state.get('processed'):
    
    col_preview, col_data = st.columns([3, 2])
    
    with col_preview:
        st.subheader(f"✨ Resultado ({len(st.session_state['palette_data'])} Cores)")
        st.markdown("Preview interativo de como o modelo ficou após a separação de cores.")
        
        # --- AQUI ESTÁ A MÁGICA: Usa o 'preview_data' colorido ---
        st.components.v1.html(render_3d_viewer(st.session_state['preview_data']), height=470)
        
        st.success("Conversão Concluída! O arquivo 3MF está pronto.")
        st.download_button(
            label="⬇️ BAIXAR ARQUIVO .3MF (PARA ORCA SLICER)", 
            data=st.session_state['3mf_data'], 
            file_name=f"modelo_{len(st.session_state['palette_data'])}cores.3mf", 
            mime="model/3mf"
        )
        if st.button("🔄 Reiniciar / Carregar Outro"):
             for key in st.session_state.keys():
                 del st.session_state[key]
             st.rerun()

    with col_data:
        st.subheader("📋 Gabarito de Cores RGB")
        st.markdown("Copie estes valores para os filamentos no Orca Slicer:")
        
        container = st.container()
        with container:
            # Altura máxima com scroll para muitas cores
            st.markdown('<div style="height: 550px; overflow-y: auto; padding-right: 10px;">', unsafe_allow_html=True)
            for p in st.session_state['palette_data']:
                card_html = f"""
                <div class="palette-card">
                    <div style="display:flex; align-items:center; gap:15px;">
                        <div class="color-preview" style="background-color: {p['hex']};"></div>
                        <span style="font-weight:600; color:#eee; font-size: 1.1em;">{p['name']}</span>
                    </div>
                    <code style="background:#111; padding:8px 12px; border-radius:6px; color:#00ff7f; font-family:monospace; font-weight:bold;">{p['rgb']}</code>
                </div>
                """
                st.markdown(card_html, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
