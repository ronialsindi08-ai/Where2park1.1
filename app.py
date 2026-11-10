import streamlit as st
import random
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import folium
from streamlit_folium import st_folium

# 1. Seitenkonfiguration für Mobile UI
st.set_page_config(
    page_title="Where to Park",
    page_icon="🅿️",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Custom Styling mit moderner Schriftart (Inter) und starkem Kontrast
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    .block-container {
        max-width: 450px;
        padding-top: 2.5rem;
        padding-bottom: 3rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #0d6efd;
        color: white;
        border-radius: 8px;
        height: 2.8em;
        font-weight: 600;
        border: none;
    }
    .badge-free {
        background-color: #198754;
        color: white;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.85rem;
        float: right;
    }
    .badge-busy {
        background-color: #dc3545;
        color: white;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.85rem;
        float: right;
    }
    .spot-card {
        background-color: #ffffff;
        border: 1px solid #dee2e6;
        border-radius: 12px;
        padding: 14px;
        margin-bottom: 12px;
        box-shadow: 0 2px 6px rgba(0,0,0,0.06);
        color: #212529;
    }
    .spot-title {
        font-size: 1.1rem;
        font-weight: 700;
        color: #0b5ed7;
    }
    .spot-info {
        font-size: 0.9rem;
        color: #495057;
        margin-top: 4px;
    }
    </style>
""", unsafe_allow_html=True)

# YOLOv8 Modell laden
@st.cache_resource
def load_yolo_model():
    return YOLO("yolov8n.pt")

model = load_yolo_model()

# Session State Initialisierung
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "license_plate" not in st.session_state:
    st.session_state.license_plate = ""

# Feste Basis-Daten für P1 bis P10 mit GPS-Koordinaten für die Karte
BASE_SPOTS = [
    {"name": "P1", "distance": 0.3, "price": "1.50 €/Std.", "lat": 53.8680, "lon": 10.6860},
    {"name": "P2", "distance": 0.7, "price": "2.00 €/Std.", "lat": 53.8692, "lon": 10.6895},
    {"name": "P3", "distance": 1.1, "price": "1.00 €/Std.", "lat": 53.8655, "lon": 10.6830},
    {"name": "P4", "distance": 1.4, "price": "1.50 €/Std.", "lat": 53.8710, "lon": 10.6815},
    {"name": "P5", "distance": 1.8, "price": "2.50 €/Std.", "lat": 53.8640, "lon": 10.6910},
    {"name": "P6", "distance": 2.1, "price": "1.00 €/Std.", "lat": 53.8725, "lon": 10.6875},
    {"name": "P7", "distance": 2.5, "price": "2.00 €/Std.", "lat": 53.8630, "lon": 10.6790},
    {"name": "P8", "distance": 0.5, "price": "1.50 €/Std.", "lat": 53.8668, "lon": 10.6880},
    {"name": "P9", "distance": 0.9, "price": "1.00 €/Std.", "lat": 53.8702, "lon": 10.6845},
    {"name": "P10", "distance": 1.6, "price": "2.00 €/Std.", "lat": 53.8620, "lon": 10.6850},
]

def randomize_spot_statuses():
    statuses = ["Frei"] * 5 + ["Belegt"] * 5
    random.shuffle(statuses)
    
    spot_data = []
    for base, status in zip(BASE_SPOTS, statuses):
        spot_data.append({
            "name": base["name"],
            "status": status,
            "distance": base["distance"],
            "price": base["price"],
            "lat": base["lat"],
            "lon": base["lon"]
        })
    return spot_data

if "spots" not in st.session_state:
    st.session_state.spots = randomize_spot_statuses()

# ==========================================
# LOGIN SEITE (Mit Bindestrich-Pflicht)
# ==========================================
if not st.session_state.logged_in:
    st.markdown("<h1 style='text-align: center; color: #0d6efd; font-size: 2.2rem;'>🅿️ Where to Park</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #6c757d;'>Mobile KI-Parkplatzsuche</p>", unsafe_allow_html=True)
    
    st.write("")
    
    with st.form("login_form"):
        license_input = st.text_input("Kfz-Kennzeichen eingeben", placeholder="z.B. HH-AB 1234")
        submit = st.form_submit_button("Einloggen")
        
        if submit:
            clean_input = license_input.strip()
            # Validierung: Darf nicht leer sein UND muss einen Bindestrich enthalten
            if clean_input != "" and "-" in clean_input:
                st.session_state.logged_in = True
                st.session_state.license_plate = clean_input.upper()
                st.session_state.spots = randomize_spot_statuses()
                st.rerun()
            elif clean_input == "":
                st.error("Bitte gib dein Kennzeichen ein.")
            else:
                st.error("Ungültiges Kennzeichen! Bitte gib ein gültiges Kennzeichen mit Bindestrich ein (z.B. HH-AB 1234).")

# ==========================================
# MAIN APP (NACH LOGIN)
# ==========================================
else:
    col_title, col_user = st.columns([2.6, 1.4])
    with col_title:
        st.markdown(f"**🚗 Kennzeichen:** `{st.session_state.license_plate}`")
    with col_user:
        if st.button("Logout", key="logout"):
            st.session_state.logged_in = False
            st.rerun()

    tab_map, tab_list, tab_ai, tab_info = st.tabs(["🗺️ Karte", "📋 Übersicht", "📷 KI-Scan", "ℹ️ Info"])

    # TAB 1: INTERAKTIVE KARTE
    with tab_map:
        st.subheader("Live Parkplatz-Karte")
        
        if st.button("🔄 Status aktualisieren (Ein-/Ausparken)"):
            st.session_state.spots = randomize_spot_statuses()
            st.rerun()

        m = folium.Map(location=[53.8675, 10.6855], zoom_start=14, tiles="OpenStreetMap")

        for spot in st.session_state.spots:
            color = "green" if spot["status"] == "Frei" else "red"
            popup_text = f"<b>{spot['name']}</b><br>Status: {spot['status']}<br>Tarif: {spot['price']}"
            
            folium.Marker(
                location=[spot["lat"], spot["lon"]],
                popup=popup_text,
                tooltip=f"{spot['name']}: {spot['status']}",
                icon=folium.Icon(color=color, icon="info-sign")
            ).add_to(m)

        st_folium(m, width=380, height=350)

    # TAB 2: DETAIL-LISTE (P1 - P10)
    with tab_list:
        st.subheader("Alle Parkflächen")
        
        for spot in st.session_state.spots:
            if spot["status"] == "Frei":
                badge = '<span class="badge-free">FREI</span>'
            else:
                badge = '<span class="badge-busy">BELEGT</span>'

            st.markdown(f"""
            <div class="spot-card">
                {badge}
                <div class="spot-title">Parkfläche {spot['name']}</div>
                <div class="spot-info">Entfernung: {spot['distance']} km &nbsp;|&nbsp; Tarif: {spot['price']}</div>
            </div>
            """, unsafe_allow_html=True)

    # TAB 3: LIVE KI-SCAN (6-Zonen Raster)
    with tab_ai:
        st.subheader("6-Zonen Parkplatz-Scanner")
        st.write("Analysiert das Luftbild aufgeteilt in 6 Einzelparkplätze (Asphalt vs. Fahrzeugfarbe):")
        uploaded_file = st.file_uploader("Luftbild hochladen", type=["jpg", "png", "jpeg"])

        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert("RGB")
            img_array = np.array(image)
            h, w, _ = img_array.shape

            with st.spinner("Teile Bild in 6 Parkzonen auf..."):
                rows, cols = 3, 2
                cell_h = h // rows
                cell_w = w // cols
                
                grid_results = []
                annotated_img = img_array.copy()

                spot_index = 1
                for r in range(rows):
                    for c in range(cols):
                        y1, y2 = r * cell_h, (r + 1) * cell_h
                        x1, x2 = c * cell_w, (c + 1) * cell_w
                        
                        cell = img_array[y1:y2, x1:x2]
                        
                        hsv_cell = cv2.cvtColor(cell, cv2.COLOR_RGB2HSV)
                        saturation = hsv_cell[:, :, 1]
                        avg_sat = np.mean(saturation)
                        
                        is_occupied = avg_sat > 30
                        status_text = "Belegt" if is_occupied else "Frei"
                        
                        grid_results.append({
                            "zone": f"Parkplatz Zone {spot_index}",
                            "status": status_text
                        })
                        
                        color_box = (0, 200, 0) if not is_occupied else (200, 0, 0)
                        cv2.rectangle(annotated_img, (x1, y1), (x2, y2), color_box, 4)
                        
                        spot_index += 1

            st.image(annotated_img, caption="6-Zonen Analyse (Grün = Frei, Rot = Belegt)", use_column_width=True)
            
            st.write("---")
            st.markdown("### 📊 Einzelauswertung der 6 Zonen:")
            for item in grid_results:
                if item["status"] == "Frei":
                    st.success(f"✅ **{item['zone']}**: Parkplatz verfügbar")
                else:
                    st.error(f"🚫 **{item['zone']}**: Kein Parkplatz verfügbar (Belegt)")

    # TAB 4: INFO
    with tab_info:
        st.subheader("System-Details")
        st.write(f"**Angemeldetes Fahrzeug:** {st.session_state.license_plate}")
        st.write("**KI-Modell:** 6-Zonen Raster-Farb-Analyse")
        st.write("**Design:** Modernes UI mit Inter-Schriftart")
