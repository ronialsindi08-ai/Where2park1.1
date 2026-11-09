import streamlit as st
import random
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO

# 1. Seitenkonfiguration & Custom CSS für Mobile UI ohne Abschneiden oben & mit starkem Kontrast
st.set_page_config(
    page_title="Where to Park",
    page_icon="🅿️",
    layout="centered",
    initial_sidebar_state="collapsed"
)

st.markdown("""
    <style>
    /* Oberen Abstand vergrößern, damit nichts abgeschnitten wird */
    .block-container {
        max-width: 450px;
        padding-top: 3rem;
        padding-bottom: 3rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #0f4c81;
        color: white;
        border-radius: 10px;
        height: 3em;
        font-weight: bold;
    }
    /* Kontrastreiche Badges für Status */
    .badge-free {
        background-color: #198754;
        color: white;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: bold;
        font-size: 0.85rem;
        float: right;
    }
    .badge-busy {
        background-color: #dc3545;
        color: white;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: bold;
        font-size: 0.85rem;
        float: right;
    }
    /* Weißer Karten-Hintergrund mit dunklem Text für optimalen Kontrast */
    .spot-card {
        background-color: #ffffff;
        border: 1px solid #ced4da;
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 12px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.08);
        color: #111111;
    }
    .spot-title {
        font-size: 1.1rem;
        font-weight: bold;
        color: #0f4c81;
    }
    .spot-info {
        font-size: 0.9rem;
        color: #444444;
        margin-top: 5px;
    }
    </style>
""", unsafe_allow_html=True)

# Modell laden (gecached)
@st.cache_resource
def load_yolo_model():
    return YOLO("yolov8n.pt")

model = load_yolo_model()

# Session State für Login und feste Parkflächen-Daten initialisieren
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "license_plate" not in st.session_state:
    st.session_state.license_plate = ""

# Feste Basis-Daten für P1 bis P10 (Entfernung und Tarif bleiben immer gleich)
BASE_SPOTS = [
    {"name": "P1", "distance": 0.3, "price": "1.50 €/Std."},
    {"name": "P2", "distance": 0.7, "price": "2.00 €/Std."},
    {"name": "P3", "distance": 1.1, "price": "1.00 €/Std."},
    {"name": "P4", "distance": 1.4, "price": "1.50 €/Std."},
    {"name": "P5", "distance": 1.8, "price": "2.50 €/Std."},
    {"name": "P6", "distance": 2.1, "price": "1.00 €/Std."},
    {"name": "P7", "distance": 2.5, "price": "2.00 €/Std."},
    {"name": "P8", "distance": 0.5, "price": "1.50 €/Std."},
    {"name": "P9", "distance": 0.9, "price": "1.00 €/Std."},
    {"name": "P10", "distance": 1.6, "price": "2.00 €/Std."},
]

def randomize_spot_statuses():
    # Genau 5 mal "Frei" und 5 mal "Belegt"
    statuses = ["Frei"] * 5 + ["Belegt"] * 5
    random.shuffle(statuses)
    
    spot_data = []
    for base, status in zip(BASE_SPOTS, statuses):
        spot_data.append({
            "name": base["name"],
            "status": status,
            "distance": base["distance"],
            "price": base["price"]
        })
    return spot_data

if "spots" not in st.session_state:
    st.session_state.spots = randomize_spot_statuses()

# ==========================================
# SEITE 1: LOGIN (KENNZEICHEN)
# ==========================================
if not st.session_state.logged_in:
    st.markdown("<h1 style='text-align: center; color: #0f4c81;'>🅿️ Where to Park</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #444;'>AI Parking Assist - Bitte Kennzeichen eingeben</p>", unsafe_allow_html=True)
    
    st.write("")
    
    with st.form("login_form"):
        license_input = st.text_input("Kfz-Kennzeichen eingeben", placeholder="z.B. HH-AB 1234")
        submit = st.form_submit_button("Einloggen")
        
        if submit:
            if license_input.strip() != "":
                st.session_state.logged_in = True
                st.session_state.license_plate = license_input.upper()
                st.session_state.spots = randomize_spot_statuses()
                st.rerun()
            else:
                st.error("Bitte gib dein Kennzeichen ein.")

# ==========================================
# SEITE 2: HAUPT-APP (NACH LOGIN)
# ==========================================
else:
    col_title, col_user = st.columns([2.5, 1.5])
    with col_title:
        st.markdown(f"**🚗 Kennzeichen:** `{st.session_state.license_plate}`")
    with col_user:
        if st.button("Abmelden", key="logout"):
            st.session_state.logged_in = False
            st.rerun()

    tab_map, tab_ai, tab_info = st.tabs(["🗺️ Parkflächen", "📷 KI-Erkennung", "ℹ️ Info"])

    # TAB 1: PARKFLÄCHEN (P1 - P10)
    with tab_map:
        st.subheader("Aktuelle Parkflächen (Live-Status)")
        st.write("Entfernung und Preise sind fest, der Status ändert sich dynamisch.")
        
        if st.button("🔄 Parkflächen aktualisieren"):
            st.session_state.spots = randomize_spot_statuses()
            st.rerun()

        st.write("")

        for spot in st.session_state.spots:
            if spot["status"] == "Frei":
                badge = '<span class="badge-free">FREI</span>'
            else:
                badge = '<span class="badge-busy">BELEGT</span>'

            st.markdown(f"""
            <div class="spot-card">
                {badge}
                <div class="spot-title">Parkfläche: {spot['name']}</div>
                <div class="spot-info">Entfernung: {spot['distance']} km &nbsp;|&nbsp; Tarif: {spot['price']}</div>
            </div>
            """, unsafe_allow_html=True)

    # TAB 2: LIVE-ERKENNUNG MIT YOLOV8
    with tab_ai:
        st.subheader("YOLOv8 Parkplatz-Scanner")
        st.write("Lade ein Foto hoch, um Fahrzeuge auf einer Fläche zu prüfen:")
        
        uploaded_file = st.file_uploader("Bild auswählen", type=["jpg", "png", "jpeg"])

        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            img_array = np.array(image)

            with st.spinner("Analysiere Bild mit YOLOv8..."):
                results = model(img_array)
                res = results[0]
                annotated_frame = res.plot()
                car_count = sum(1 for box in res.boxes if int(box.cls[0]) in [2, 3, 5, 7])

            st.image(annotated_frame, caption="Erkannte Fahrzeuge", use_column_width=True)
            st.success(f"**Ergebnis:** {car_count} Fahrzeuge auf dem Bild erkannt.")

    # TAB 3: INFO
    with tab_info:
        st.subheader("System-Details")
        st.write(f"**Aktives Kennzeichen:** {st.session_state.license_plate}")
        st.write("**Modell:** YOLOv8 Nano")
        st.write("**Modus:** Feste Entfernungen & Tarife mit dynamischem 50/50 Status-Mix")
