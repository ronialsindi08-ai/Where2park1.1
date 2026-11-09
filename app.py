import streamlit as st
import random
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO

# 1. Seitenkonfiguration & Custom CSS für Mobile UI mit starkem Kontrast
st.set_page_config(
    page_title="Where to Park",
    page_icon="🅿️",
    layout="centered",
    initial_sidebar_state="collapsed"
)

st.markdown("""
    <style>
    .block-container {
        max-width: 450px;
        padding-top: 1.5rem;
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
    }
    .badge-busy {
        background-color: #dc3545;
        color: white;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: bold;
    }
    .spot-card {
        background-color: #ffffff;
        border: 1px solid #ced4da;
        border-radius: 10px;
        padding: 12px 15px;
        margin-bottom: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    </style>
""", unsafe_allow_html=True)

# Modell laden (gecached)
@st.cache_resource
def load_yolo_model():
    return YOLO("yolov8n.pt")

model = load_yolo_model()

# Session State für Login und Parkflächen initialisieren
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "license_plate" not in st.session_state:
    st.session_state.license_plate = ""

# Funktion, die 10 Parkflächen (P1 bis P10) mit genau 50% frei / 50% belegt generiert
def get_random_parking_spots():
    spots = [f"P{i}" for i in range(1, 11)]
    # 5 mal Frei ("Frei"), 5 mal Belegt ("Belegt")
    statuses = ["Frei"] * 5 + ["Belegt"] * 5
    random.shuffle(statuses)
    
    spot_data = []
    for spot, status in zip(spots, statuses):
        spot_data.append({
            "name": spot,
            "status": status,
            "distance": round(random.uniform(0.2, 2.5), 1),
            "price": f"{random.choice([1.00, 1.50, 2.00]):.2f} €/Std."
        })
    return spot_data

# Wenn man sich frisch einloggt oder aktualisiert, neue Zufallswerte erzeugen
if "spots" not in st.session_state:
    st.session_state.spots = get_random_parking_spots()

# ==========================================
# SEITE 1: LOGIN (NUR KENNZEICHEN)
# ==========================================
if not st.session_state.logged_in:
    st.markdown("<h2 style='text-align: center; color: #0f4c81;'>🅿️ Where to Park</h2>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #555;'>Gib dein Kennzeichen ein, um die Live-Übersicht zu starten</p>", unsafe_allow_html=True)
    
    st.write("")
    
    with st.form("login_form"):
        license_input = st.text_input("Kfz-Kennzeichen eingeben", placeholder="z.B. HH-AB 1234")
        submit = st.form_submit_button("Einloggen")
        
        if submit:
            if license_input.strip() != "":
                st.session_state.logged_in = True
                st.session_state.license_plate = license_input.upper()
                st.session_state.spots = get_random_parking_spots() # Frische Daten beim Login
                st.rerun()
            else:
                st.error("Bitte gib dein Kennzeichen ein.")

# ==========================================
# SEITE 2: HAUPT-APP (NACH LOGIN)
# ==========================================
else:
    col_title, col_user = st.columns([3, 1])
    with col_title:
        st.markdown(f"**🚗 Kennzeichen:** `{st.session_state.license_plate}`")
    with col_user:
        if st.button("Logout", key="logout"):
            st.session_state.logged_in = False
            st.rerun()

    tab_map, tab_ai, tab_info = st.tabs(["🗺️ Parkflächen", "📷 KI-Erkennung", "ℹ️ Info"])

    # TAB 1: PARKFLÄCHEN (P1 - P10) MIT AUSGEWOGENEM ZUFALLS-STATUS
    with tab_map:
        st.subheader("Aktuelle Parkflächen (Live-Status)")
        st.write("Der Status ändert sich dynamisch (jemand fährt ein/aus).")
        
        if st.button("🔄 Parkflächen aktualisieren (Neuer Zufall)"):
            st.session_state.spots = get_random_parking_spots()
            st.rerun()

        st.write("")

        # Alle 10 Parkflächen übersichtlich als Cards anzeigen
        for spot in st.session_state.spots:
            if spot["status"] == "Frei":
                badge = '<span class="badge-free">FREI</span>'
                action_btn = f"Hinfahren ({spot['name']})"
            else:
                badge = '<span class="badge-busy">BELEGT</span>'
                action_btn = f"Besetzt"

            st.markdown(f"""
            <div class="spot-card">
                <b>Parkfläche: {spot['name']}</b> &nbsp;&nbsp; {badge}<br>
                <small style="color: #444;">Entfernung: {spot['distance']} km | Tarif: {spot['price']}</small>
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
        st.write("**Modus:** Ausgewogener Zufalls-Status (50% Frei / 50% Belegt bei 10 Flächen)")
