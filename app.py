import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO

# 1. Seitenkonfiguration & Custom CSS für Mobile UI
st.set_page_config(
    page_title="Where to Park",
    page_icon="🅿️",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Custom Styling für App-Feeling
st.markdown("""
    <style>
    /* Maximale Breite auf Smartphone-Größe begrenzen */
    .block-container {
        max-width: 450px;
        padding-top: 1.5rem;
        padding-bottom: 3rem;
    }
    /* Primäre Buttons styling */
    .stButton>button {
        width: 100%;
        background-color: #1E88E5;
        color: white;
        border-radius: 10px;
        height: 3em;
        font-weight: bold;
    }
    /* Card Design für Status */
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 12px;
        padding: 15px;
        margin-bottom: 10px;
        border: 1px solid #e9ecef;
    }
    </style>
""", unsafe_allow_html=True)

# Modell laden (gecached, damit es schnell bleibt)
@st.cache_resource
def load_yolo_model():
    # Lädt das vortrainierte YOLOv8 Nano-Modell (leicht & schnell)
    return YOLO("yolov8n.pt")

model = load_yolo_model()

# Session State für Login-Status initialisieren
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "license_plate" not in st.session_state:
    st.session_state.license_plate = ""

# ==========================================
# SEITE 1: LOGIN (NUR KENNZEICHEN)
# ==========================================
if not st.session_state.logged_in:
    st.markdown("<h2 style='text-align: center;'>🅿️ Where to Park</h2>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: gray;'>Finde freie Parkplätze in deiner Nähe</p>", unsafe_allow_html=True)
    
    st.space = st.write("")
    
    with st.form("login_form"):
        license_input = st.text_input("Kfz-Kennzeichen eingeben", placeholder="z.B. HH-AB 1234")
        submit = st.form_submit_button("Anmelden / Einloggen")
        
        if submit:
            if license_input.strip() != "":
                st.session_state.logged_in = True
                st.session_state.license_plate = license_input.upper()
                st.rerun()
            else:
                st.error("Bitte gib dein Kennzeichen ein.")

# ==========================================
# SEITE 2: HAUPT-APP (NACH LOGIN)
# ==========================================
else:
    # Top-Bar & Logout
    col_title, col_user = st.columns([3, 1])
    with col_title:
        st.markdown(f"**🚗 {st.session_state.license_plate}**")
    with col_user:
        if st.button("Abmelden", key="logout"):
            st.session_state.logged_in = False
            st.rerun()

    # Untere/Obere Tab-Navigation (Map, Live-KI-Scan, Einstellungen)
    tab_map, tab_ai, tab_info = st.tabs(["🗺️ Karte", "📷 KI-Erkennung", "ℹ️ Info"])

    # TAB 1: KARTE & STATUS
    with tab_map:
        st.subheader("Freie Parkplätze in der Nähe")
        st.text_input("📍 Standort suchen...", value="Hamburg Mitte")
        
        # Simulierter Parkplatz-Status
        st.markdown("""
        <div class="metric-card">
            <h4>🅿️ Parkhaus Holstentor</h4>
            <p><b>Status:</b> <span style="color:green;">Frei (ca. 18 Plätze)</span><br>
            <b>Entfernung:</b> 0.8 km | <b>Gebühren:</b> 1,50 €/Std.</p>
        </div>
        """, unsafe_allow_html=True)
        st.button("Navigation starten 📍", key="nav1")

        st.markdown("""
        <div class="metric-card">
            <h4>🅿️ Parkplatz Hauptstraße</h4>
            <p><b>Status:</b> <span style="color:red;">Belegt (0 Plätze)</span><br>
            <b>Entfernung:</b> 1.2 km | <b>Gebühren:</b> Kostenlos</p>
        </div>
        """, unsafe_allow_html=True)

    # TAB 2: LIVE-ERKENNUNG MIT YOLOV8
    with tab_ai:
        st.subheader("YOLOv8 Parkplatz-Analyse")
        st.write("Lade ein Bild hoch, um Autos und freie Stellflächen zu erkennen:")
        
        uploaded_file = st.file_uploader("Bild auswählen", type=["jpg", "png", "jpeg"])

        if uploaded_file is not None:
            # Bild laden
            image = Image.open(uploaded_file)
            img_array = np.array(image)

            with st.spinner("YOLOv8 analysiert das Bild..."):
                # YOLOv8 Inferenz ausführen
                results = model(img_array)
                res = results[0]

                # Bounding Boxes rendern
                annotated_frame = res.plot()
                
                # Zählen, wie viele Autos (Class ID 2 in COCO Dataset) erkannt wurden
                car_count = sum(1 for box in res.boxes if int(box.cls[0]) in [2, 3, 5, 7]) # Auto, Lkw, Bus

            # Bild mit Detections anzeigen
            st.image(annotated_frame, caption="YOLOv8 Live-Analyse", use_column_width=True)
            
            # Auswertung anzeigen
            st.success(f"**Ergebnis:** {car_count} Fahrzeuge auf dem Bild erkannt!")

    # TAB 3: PROFIL & MODELL-INFO
    with tab_info:
        st.subheader("System-Informationen")
        st.write(f"**Angemeldetes Kennzeichen:** {st.session_state.license_plate}")
        st.write("**KI-Modell:** YOLOv8 Nano (Ultralytics)")
        st.write("**Framework:** Streamlit / Python")
