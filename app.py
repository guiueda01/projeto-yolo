import cv2
import numpy as np
from PIL import Image
import streamlit as st
from ultralytics import YOLO

# Configuração da página do Streamlit
st.set_page_config(page_title="Detector de Objetos YOLOv8", layout="wide")

st.title("📷 Detector de Objetos Inteligente")
st.write("Faça o upload de uma imagem para identificar os objetos presentes em tempo real.")


# Inicializa o modelo YOLOv8 Nano (baixa automaticamente no primeiro uso)
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")


try:
    model = load_model()
except Exception as e:
    st.error(f"Erro ao carregar o modelo: {e}")
    st.stop()

# Componente de Upload de Imagem
uploaded_file = st.file_uploader(
    "Escolha uma imagem...", type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    # Converte o arquivo enviado para uma imagem PIL e depois para array NumPy (RGB)
    image = Image.open(uploaded_file)
    img_array = np.array(image)

    # Cria duas colunas para exibição lado a lado
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Imagem Original")
        st.image(image, use_container_width=True)

    with col2:
        st.subheader("Objetos Detectados")

        # Roda a inferência do YOLOv8
        # conf=0.25 define o limite mínimo de confiança para detecção
        results = model(img_array, conf=0.25)

        # O YOLO retorna os resultados com os plots em formato BGR do OpenCV
        res_plotted = results[0].plot()

        # Converte de BGR de volta para RGB para exibição correta no Streamlit
        res_rgb = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)

        st.image(res_rgb, use_container_width=True)

        # Exibe um resumo textual dos objetos encontrados
        st.write("**Resumo das detecções:**")
        boxes = results[0].boxes
        if len(boxes) == 0:
            st.info("Nenhum objeto conhecido foi detectado.")
        else:
            for box in boxes:
                cls_id = int(box.cls[0])
                label = model.names[cls_id]
                conf = float(box.conf[0])
                st.write(f"- ✔️ **{label.title()}** (Confiança: {conf:.2%})")       