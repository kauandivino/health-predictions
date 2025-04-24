import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from PIL import Image
import numpy as np
import os
import os, streamlit as st

# Configurações gerais da página
st.set_page_config(page_title="IA em Biotecnologia", layout="wide")
st.write("Arquivos em /models:", os.listdir("models"))
st.title("IA aplicada à Biotecnologia")
st.markdown("Bem-vindo ao demo de IA em Biotecnologia! Use as abas acima para navegar.")

# Rótulos em português para cada modelo
MODEL_LABELS = {
    "Blood Cell": ["Eosinófilo", "Linfócito", "Monócito", "Neutrófilo"],
    "Brain Tumor": ["Tumor de Glioma", "Sem Tumor", "Tumor de Meningioma", "Tumor Hipofisário"],
    "Lung & Colon Cancer": [
        "Tecido Pulmonar Benigno",
        "Adenocarcinoma de Pulmão",
        "Carcinoma Espinocelular de Pulmão",
        "Adenocarcinoma de Cólon",
        "Tecido Benigno do Cólon"
    ],
    "Pneumonia": ["Normal", "Pneumonia"]
}

# Configuração de cada modelo: (caminho, tamanho de entrada, usar preprocess_input?)
MODEL_CONFIG = {
    "Blood Cell": ("models/blood_cell.h5", (244, 244), True),
    "Brain Tumor": ("models/brain_tumor.h5", (150, 150), False),
    "Lung & Colon Cancer": ("models/lung_colon_cancer.h5", (224, 224), False),
    "Pneumonia": ("models/pneumonia.h5", (244, 244), True)
}

@st.cache_resource
def load_models():
    models = {}
    for name, (path, size, use_pi) in MODEL_CONFIG.items():
        if not os.path.exists(path):
            st.error(f"Modelo não encontrado: {path}")
            continue
        models[name] = load_model(path)
    return models

models = load_models()

# Cria as cinco abas principais
tabs = st.tabs([
    "Sobre IA na Biotecnologia",
    "Classificação de Células Sanguíneas",
    "Detecção de Tumores Cerebrais",
    "Histopatologia de Pulmão & Cólon",
    "Detecção de Pneumonia"
])

# Aba 1: Sobre IA na Biotecnologia
with tabs[0]:
    col1, col2 = st.columns([2, 1])
    with col1:
        st.header("O que é Inteligência Artificial?")
        st.write(
            "A inteligência artificial (IA) refere-se a sistemas "
            "computacionais capazes de realizar tarefas que normalmente "
            "requerem inteligência humana, como reconhecimento de padrões, "
            "classificação de imagens e tomada de decisão."
        )
        st.subheader("Redes Neurais Convolucionais")
        st.write(
            "As CNNs são arquiteturas profundas especialmente eficazes "
            "para processamento de imagens. Elas utilizam camadas de convolução "
            "para extrair características e mapear padrões visuais."
        )
        st.markdown("---")
    with col2:
        st.image("static/images/ai_overview.png",
                 caption="Visão geral de IA",
                 use_column_width=True)
# Aba 2: Classificação de Células Sanguíneas
with tabs[1]:
    st.header("Classificação de Células Sanguíneas")
    st.write(
        "Neste demo, classificamos células sanguíneas em quatro tipos "
        "com base em imagens de microscópio. Dataset: 12.500 imagens, "
        "aprox. 3.000 por classe."
    )
    st.image(
        "static/images/blood_cell_sample.jpg",
        width=300,
        caption="Exemplos de Eosinófilos, Linfócitos, Monócitos e Neutrófilos"
    )
    st.subheader("Teste sua imagem de célula")
    uploaded = st.file_uploader(
        "Envie uma imagem de célula (JPEG/PNG):",
        type=["jpg", "jpeg", "png"],
        key="blood_cell"
    )
    if uploaded:
        img = Image.open(uploaded).convert("RGB")
        st.image(img, caption="Imagem enviada", use_column_width=True)
        # Preprocessamento
        target_size = MODEL_CONFIG["Blood Cell"][1]
        arr_img = img.resize(target_size)
        arr = np.array(arr_img)
        if MODEL_CONFIG["Blood Cell"][2]:
            arr = preprocess_input(arr)
        else:
            arr = arr / 255.0
        # Inferência
        scores = models["Blood Cell"].predict(arr[np.newaxis, ...])[0]
        idx = np.argmax(scores)
        label = MODEL_LABELS["Blood Cell"][idx]
        # Exibição de resultado
        colA, colB = st.columns([3, 2])
        with colA:
            st.subheader(f"Predição: {label}")
        with colB:
            probas = {
                MODEL_LABELS["Blood Cell"][i]: float(scores[i])
                for i in range(len(scores))
            }
            st.bar_chart(probas)

# Aba 3: Detecção de Tumores Cerebrais
with tabs[2]:
    st.header("Detecção de Tumores Cerebrais")
    st.write(
        "Este demo classifica imagens de ressonância magnética em quatro categorias: "
        "Tumor de Glioma, Sem Tumor, Tumor de Meningioma e Tumor Hipofisário."
    )
    st.image(
        "static/images/brain_mri_sample.jpg",
        width=300,
        caption="Exemplo de imagem de ressonância magnética"
    )
    st.subheader("Teste sua imagem de RM")
    uploaded2 = st.file_uploader(
        "Envie uma imagem de RM (JPEG/PNG):",
        type=["jpg", "jpeg", "png"],
        key="brain_tumor"
    )
    if uploaded2:
        img2 = Image.open(uploaded2).convert("RGB")
        st.image(img2, caption="Imagem enviada (RM)", use_column_width=True)
        # Preprocessamento
        target_size2 = MODEL_CONFIG["Brain Tumor"][1]
        arr_img2 = img2.resize(target_size2)
        arr2 = np.array(arr_img2)
        if MODEL_CONFIG["Brain Tumor"][2]:
            arr2 = preprocess_input(arr2)
        else:
            arr2 = arr2 / 255.0
        # Inferência
        scores2 = models["Brain Tumor"].predict(arr2[np.newaxis, ...])[0]
        idx2 = np.argmax(scores2)
        label2 = MODEL_LABELS["Brain Tumor"][idx2]
        # Exibição de resultado
        colC, colD = st.columns([3, 2])
        with colC:
            st.subheader(f"Predição: {label2}")
        with colD:
            probas2 = {
                MODEL_LABELS["Brain Tumor"][i]: float(scores2[i])
                for i in range(len(scores2))
            }
            st.bar_chart(probas2)

# Aba 4: Histopatologia de Pulmão & Cólon
with tabs[3]:
    st.header("Histopatologia de Pulmão & Cólon")
    st.write(
        "Classificamos lâminas histopatológicas em cinco categorias: "
        "Tecido Pulmonar Benigno, Adenocarcinoma de Pulmão, "
        "Carcinoma Espinocelular de Pulmão, Adenocarcinoma de Cólon "
        "e Tecido Benigno do Cólon."
    )
    # Mini-galeria de exemplos
    st.image(
        [
            "static/images/lung_benigno.jpg",
            "static/images/adenoca_pulmao.jpg",
            "static/images/carcino_pulmao.jpg",
            "static/images/adenoca_colon.jpg",
            "static/images/colon_benigno.jpg"
        ],
        width=150,
        caption=[
            "Pulmão Benigno",
            "Adenocarcinoma de Pulmão",
            "Espinocelular de Pulmão",
            "Adenocarcinoma de Cólon",
            "Cólon Benigno"
        ]
    )
    st.subheader("Teste sua lâmina histológica")
    uploaded3 = st.file_uploader(
        "Envie uma imagem de lâmina (JPEG/PNG):",
        type=["jpg", "jpeg", "png"],
        key="lung_colon"
    )
    if uploaded3:
        img3 = Image.open(uploaded3).convert("RGB")
        st.image(img3, caption="Imagem enviada (Histologia)", use_column_width=True)
        # Preprocessamento
        target_size3 = MODEL_CONFIG["Lung & Colon Cancer"][1]
        arr_img3 = img3.resize(target_size3)
        arr3 = np.array(arr_img3)
        if MODEL_CONFIG["Lung & Colon Cancer"][2]:
            arr3 = preprocess_input(arr3)
        else:
            arr3 = arr3 / 255.0
        # Inferência
        scores3 = models["Lung & Colon Cancer"].predict(arr3[np.newaxis, ...])[0]
        idx3 = np.argmax(scores3)
        label3 = MODEL_LABELS["Lung & Colon Cancer"][idx3]
        # Exibição de resultado
        colE, colF = st.columns([3, 2])
        with colE:
            st.subheader(f"Predição: {label3}")
        with colF:
            probas3 = {
                MODEL_LABELS["Lung & Colon Cancer"][i]: float(scores3[i])
                for i in range(len(scores3))
            }
            st.bar_chart(probas3)

# Aba 5: Detecção de Pneumonia
with tabs[4]:
    st.header("Detecção de Pneumonia")
    st.write(
        "Classificação de radiografias de tórax em duas categorias: Normal ou Pneumonia."
    )
    st.image(
        "static/images/pneumonia_xray_sample.png",
        width=300,
        caption="Exemplo de raio-X de tórax"
    )
    st.subheader("Teste sua radiografia")
    uploaded4 = st.file_uploader(
        "Envie um raio-X (JPEG/PNG):",
        type=["jpg", "jpeg", "png"],
        key="pneumonia"
    )
    if uploaded4:
        img4 = Image.open(uploaded4).convert("RGB")
        st.image(img4, caption="Imagem enviada (Raio-X)", use_column_width=True)
        # Preprocessamento
        target_size4 = MODEL_CONFIG["Pneumonia"][1]
        arr_img4 = img4.resize(target_size4)
        arr4 = np.array(arr_img4)
        if MODEL_CONFIG["Pneumonia"][2]:
            arr4 = preprocess_input(arr4)
        else:
            arr4 = arr4 / 255.0
        # Inferência
        scores4 = models["Pneumonia"].predict(arr4[np.newaxis, ...])[0]
        idx4 = np.argmax(scores4)
        label4 = MODEL_LABELS["Pneumonia"][idx4]
        # Exibição de resultado
        colG, colH = st.columns([3, 2])
        with colG:
            st.subheader(f"Predição: {label4}")
        with colH:
            probas4 = {
                MODEL_LABELS["Pneumonia"][i]: float(scores4[i])
                for i in range(len(scores4))
            }
            st.bar_chart(probas4)
