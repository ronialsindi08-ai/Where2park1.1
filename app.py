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

# Custom Styling für echtes App-Gefühl & dauerhaft fixierte untere Leiste
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        background-color: #f4f6f9;
        color: #212529;
    }
    .block-container {
        max-width: 450px !important;
        padding-top: 1.5rem !important;
        padding-bottom: 7rem !important; /* Platz für die fixierte Leiste */
        margin: 0 auto;
    }
    .stButton>button {
        width: 100%;
        background-color: #0b5ed7;
        color: white;
        border-radius: 12px;
        height: 3.2em;
        font-weight: 600;
        border: none;
        box-shadow: 0 4px 10px rgba(11, 94, 215, 0.2);
    }
    .stButton>button:hover {
        background-color: #0a58ca;
    }
    .header-box {
        display: flex;
        justify-content: space-between;
        align-items: center;
        background-color: #ffffff;
        padding: 12px 16px;
        border-radius: 14px;
        border: 1px solid #dee2e6;
        margin-bottom: 1rem;
        box-shadow: 0 2px 6px rgba(0,0,0,0.04);
    }
    .spot-card {
        background-color: #ffffff;
        border: 1px solid #dee2e6;
        border-radius: 14px;
        padding: 16px;
        margin-bottom: 12px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
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
    /* Die Navigationsleiste ist nun IMMER fix am unteren Bildschirmrand verankert */
    .fixed-bottom-nav {
        position: fixed;
        bottom: 0;
        left: 50%;
        transform: translateX(-50%);
        width: 100%;
        max-width: 450px;
        background-color: #ffffff;
        border-top: 1px solid #dee2e6;
        padding: 10px 0;
        z-index: 99999;
        box-shadow: 0 -4px 15px rgba(0,0,0,0.08);
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
if "active_tab" not in st.session_state:
    st.session_state.active_tab = "Karte"

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
# LOGIN SEITE
# ==========================================
if not st.session_state.logged_in:
    # Logo eingebunden
    st.markdown("""
        <div style="display: flex; justify-content: center; margin-bottom: 0.5rem; margin-top: 1rem;">
            <svg width="84" height="84" viewBox="0 0 100 100" fill="none" xmlns="http://www.w3.org/2000/svg">
                <rect width="100" height="100" rx="24" fill="#0b5ed7"/>
                <path d="M30 70V30H48C56 30 62 35 62 43C62 51 56 56 48 56H38V70H30ZM38 48H47C51 48 54 45 54 43C54 41 51 38 47 38H38V48Z" fill="white"/>
                <path d="M55 52L68 70H78L62 48L75 30H65L55 43V30H48V70H55V52Z" fill="white" fill-opacity="0.85"/>
            </svg>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("<h1 style='text-align: center; color: #0b5ed7; font-size: 2.2rem; margin-bottom: 0;'>Where to Park</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #6c757d; font-size: 0.9rem; letter-spacing: 1px; margin-bottom: 2rem;'>AI PARKING ASSIST</p>", unsafe_allow_html=True)
    
    with st.form("login_form"):
        license_input = st.text_input("Kfz-Kennzeichen eingeben", placeholder="z.B. HH-AB 1234")
        submit = st.form_submit_button("GET STARTED")
        
        if submit:
            clean_input = license_input.strip()
            if clean_input != "" and "-" in clean_input:
                st.session_state.logged_in = True
                st.session_state.license_plate = clean_input.upper()
                st.session_state.spots = randomize_spot_statuses()
                st.rerun()
            elif clean_input == "":
                st.error("Bitte gib dein Kennzeichen ein.")
            else:
                st.error("Bitte gib ein gültiges Kennzeichen mit Bindestrich ein (z.B. HH-AB 1234).")

# ==========================================
# MAIN APP (NACH LOGIN)
# ==========================================
else:
    st.markdown(f"""
        <div class="header-box">
            <div><span style="font-size: 0.8rem; color: #6c757d; display: block;">FAHRZEUG</span><strong style="color: #0b5ed7; font-size: 1rem;">{st.session_state.license_plate}</strong></div>
    """, unsafe_allow_html=True)
    
    if st.button("Logout", key="logout"):
        st.session_state.logged_in = False
        st.rerun()
        
    st.markdown("</div>", unsafe_allow_html=True)

    tab = st.session_state.active_tab

    # TAB 1: KARTE
    if tab == "Karte":
        st.subheader("Live Parkplatz-Karte")
        
        if st.button("🔄 Status aktualisieren"):
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

        st_folium(m, width=410, height=520)

    # TAB 2: ÜBERSICHT (LISTE)
    elif tab == "Übersicht":
        st.subheader("Alle Parkflächen")
        
        for spot in st.session_state.spots:
            badge_class = "badge-free" if spot["status"] == "Frei" else "badge-busy"
            badge_text = "FREI" if spot["status"] == "Frei" else "BELEGT"

            st.markdown(f"""
            <div class="spot-card">
                <span class="{badge_class}">{badge_text}</span>
                <div class="spot-title">Parkfläche {spot['name']}</div>
                <div class="spot-info">Entfernung: {spot['distance']} km &nbsp;|&nbsp; Tarif: {spot['price']}</div>
            </div>
            """, unsafe_allow_html=True)

    # TAB 3: KI-SCAN
    elif tab == "KI-Scan":
        st.subheader("Smart Parkplatz-Scanner")
        st.write("Wähle das Raster-Format deines Fotos aus:")
        
        layout_choice = st.selectbox(
            "Raster-Format wählen:",
            [
                "1x1 (1 Parkplatz)",
                "2x1 (2 Parkplätze)",
                "2x2 (4 Parkplätze)",
                "3x2 (6 Parkplätze)",
                "4x2 (8 Parkplätze)",
                "3x3 (9 Parkplätze)",
                "4x3 (12 Parkplätze)"
            ]
        )
        
        uploaded_file = st.file_uploader("Luftbild hochladen", type=["jpg", "png", "jpeg"])

        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert("RGB")
            img_array = np.array(image)
            h, w, _ = img_array.shape

            if "1x1" in layout_choice:
                rows, cols = 1, 1
            elif "2x1" in layout_choice:
                rows, cols = 1, 2
            elif "2x2" in layout_choice:
                rows, cols = 2, 2
            elif "3x2" in layout_choice:
                rows, cols = 3, 2
            elif "4x2" in layout_choice:
                rows, cols = 4, 2
            elif "3x3" in layout_choice:
                rows, cols = 3, 3
            elif "4x3" in layout_choice:
                rows, cols = 4, 3
            else:
                rows, cols = 3, 2

            with st.spinner(f"Analysiere Raster ({rows}x{cols})..."):
                cell_h = h // rows
                cell_w = w // cols
                
                grid_results = []
                annotated_img = img_array.copy()

                spot_index = 1
                for r in range(rows):
                    for c in range(cols):
                        y1, y2 = r * cell_h, (min((r + 1) * cell_h, h))
                        x1, x2 = c * cell_w, (min((c + 1) * cell_w, w))
                        
                        cell = img_array[y1:y2, x1:x2]
                        if cell.size > 0:
                            ch, cw, _ = cell.shape
                            
                            my = int(ch * 0.20)
                            mx = int(cw * 0.20)
                            core_cell = cell[my:ch-my, mx:cw-mx]
                            if core_cell.size == 0:
                                core_cell = cell

                            r_ch, g_ch, b_ch = cv2.split(core_cell)
                            diff_rg = cv2.absdiff(r_ch, g_ch)
                            diff_gb = cv2.absdiff(g_ch, b_ch)
                            diff_rb = cv2.absdiff(r_ch, b_ch)
                            pixel_color_diff = diff_rg.astype(float) + diff_gb.astype(float) + diff_rb.astype(float)
                            
                            hsv_core = cv2.cvtColor(core_cell, cv2.COLOR_RGB2HSV)
                            pixel_sat = hsv_core[:, :, 1].astype(float)
                            
                            gray_core = cv2.cvtColor(core_cell, cv2.COLOR_RGB2GRAY)
                            pixel_gray = gray_core.astype(float)
                            is_white_or_dark = (pixel_gray > 165) | (pixel_gray < 65)
                            
                            car_pixels = (pixel_color_diff > 35) | (pixel_sat > 18) | is_white_or_dark
                            car_pixel_ratio = np.sum(car_pixels) / core_cell.size
                            
                            is_occupied = car_pixel_ratio > 0.22
                            status_text = "frei" if not is_occupied else "belegt"
                            
                            grid_results.append({
                                "zone": f"P{spot_index}",
                                "status": status_text
                            })
                            
                            box_color = (0, 255, 0) if not is_occupied else (255, 0, 0)
                            cv2.rectangle(annotated_img, (x1, y1), (x2, y2), box_color, 4)
                            cv2.putText(annotated_img, f"P{spot_index}", (x1 + 10, y1 + 35), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
                            
                            spot_index += 1

            st.image(annotated_img, caption="Ergebnis (Grün = Frei, Rot = Belegt)", use_column_width=True)
            
            st.write("---")
            st.markdown("### 📊 Einzelauswertung:")
            for item in grid_results:
                if item["status"] == "frei":
                    st.success(f"✅ **{item['zone']} ist frei**")
                else:
                    st.error(f"🚫 **{item['zone']} ist belegt**")

    # TAB 4: INFO
    elif tab == "Info":
        st.subheader("System-Details")
        st.write(f"**Angemeldetes Fahrzeug:** {st.session_state.license_plate}")
        st.write("**KI-Modell:** Präzisions-Raster mit Kernbereichs-Analyse")
        st.write("**Design:** Fixierte untere Navigation (Viewport-gesichert)")

    # Dauerhaft fixierte untere Navigationsleiste
    st.markdown("""
        <div class="fixed-bottom-nav">
            <div style="display: flex; justify-content: space-around; max-width: 450px; margin: 0 auto;">
    """, unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        if st.button("🗺️ Karte", key="nav_karte"):
            st.session_state.active_tab = "Karte"
            st.rerun()
    with c2:
        if st.button("📋 Liste", key="nav_liste"):
            st.session_state.active_tab = "Übersicht"
            st.rerun()
    with c3:
        if st.button("📷 Scan", key="nav_scan"):
            st.session_state.active_tab = "KI-Scan"
            st.rerun()
    with c4:
        if st.button("ℹ️ Info", key="nav_info"):
            st.session_state.active_tab = "Info"
            st.rerun()

    st.markdown("</div></div>", unsafe_allow_html=True)
