import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import io
from scipy.signal import find_peaks
from scipy.interpolate import interp1d
import google.generativeai as genai
import importlib.metadata

# 1. ตั้งค่าหน้าเว็บ
st.set_page_config(page_title="ISOLDRACE: Circuit Tools Mode", layout="wide", page_icon="🏎️")
st.title("🏁 ISOLDRACE: Pro Analysis (Circuit Tools Style)")

# --- Sidebar ---
with st.sidebar:
    st.header("⚙️ Configuration")
    
    st.markdown("### 🔑 AI Brain (Google Gemini)")
    api_key = st.text_input("Enter Gemini API Key:", type="password")
    
    st.divider()
    st.header("📍 Track Settings")
    if 'sf_lat' not in st.session_state: st.session_state.sf_lat = 14.957958
    if 'sf_lon' not in st.session_state: st.session_state.sf_lon = 103.085923
    
    sf_lat_input = st.number_input("Start/Finish Latitude:", value=st.session_state.sf_lat, format="%.6f")
    sf_lon_input = st.number_input("Start/Finish Longitude:", value=st.session_state.sf_lon, format="%.6f")
    sf_radius_input = st.slider("Detection Radius (m):", 10, 100, 50)
    min_corner_speed = st.slider("Min Corner Speed (km/h):", 30, 100, 40)
    
    TRACK_CONFIG = {
        'sf_lat': sf_lat_input,
        'sf_lon': sf_lon_input,
        'sf_radius_m': sf_radius_input
    }

# --- Functions ---
def smart_coord_convert(val):
    if pd.isna(val) or val == 0: return val
    if abs(val) <= 180: return val
    degrees = int(val / 100)
    minutes = abs(val) % 100
    if minutes >= 60 or abs(degrees) > 180: return val / 60.0
    decimal = degrees + (minutes / 60)
    if val < 0: decimal = -decimal
    return decimal

def dist_from_sf(lat1, lon1, lat2, lon2):
    R = 6371000
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlambda/2)**2
    return 2 * R * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

def parse_file(uploaded_file):
    filename = uploaded_file.name.lower()
    content = uploaded_file.getvalue().decode('latin-1') 
    
    if filename.endswith('.vbo'):
        lines = content.splitlines()
        data_start = 0
        cols = []
        is_col = False
        for i, line in enumerate(lines):
            line = line.strip()
            if '[column names]' in line: is_col = True; continue
            if is_col and line.startswith('['): is_col = False
            if is_col and line: cols.extend(line.split())
            if '[data]' in line: data_start = i + 1; break
        if not cols or data_start == 0: return None, "Invalid VBO"
        try:
            df = pd.read_csv(io.StringIO("\n".join(lines[data_start:])), sep=r'\s+', names=cols, engine='python')
            return df, None
        except Exception as e: return None, str(e)
    else:
        try:
            df = pd.read_csv(uploaded_file)
            return df, None
        except Exception as e: return None, str(e)

def process_laps(df, filename):
    df.columns = df.columns.str.lower()
    cols = df.columns.tolist()
    
    def get_col(candidates):
        for c in candidates:
            for col in cols: 
                if c in col: return col
        return None

    speed_c = get_col(['vel', 'speed', 'kmh'])
    lat_c = get_col(['lat'])
    lon_c = get_col(['lon', 'long'])
    
    if not speed_c or not lat_c: return []

    work_df = df.copy()
    work_df['speed'] = work_df[speed_c]
    work_df['lat'] = work_df[lat_c].apply(smart_coord_convert)
    work_df['lon'] = work_df[lon_c].apply(smart_coord_convert)
    
    if work_df['lon'].mean() < 0 and TRACK_CONFIG['sf_lon'] > 0:
            work_df['lon'] = work_df['lon'].abs()

    work_df['dist_to_sf'] = dist_from_sf(work_df['lat'], work_df['lon'], TRACK_CONFIG['sf_lat'], TRACK_CONFIG['sf_lon'])
    
    is_near = (work_df['dist_to_sf'] < TRACK_CONFIG['sf_radius_m']) & (work_df['speed'] > 10)
    sf_idx = work_df[is_near].index.tolist()
    
    final_sf = []
    if sf_idx:
        final_sf = [sf_idx[0]]
        for idx in sf_idx[1:]:
            if idx - final_sf[-1] > 100: final_sf.append(idx)
            
    processed_laps = []
    if len(final_sf) > 1:
        for i in range(len(final_sf)-1):
            s, e = final_sf[i], final_sf[i+1]
            lap_data = work_df.iloc[s:e].copy()
            lap_sec = len(lap_data) * 0.1 
            
            if 30 < lap_sec < 600:
                lap_data['lap_dist'] = (lap_data['speed'] / 3.6 * 0.1).cumsum()
                lap_data['time_elapsed'] = np.arange(len(lap_data)) * 0.1
                
                speed_inv = -lap_data['speed'].values
                peaks, _ = find_peaks(speed_inv, distance=50, prominence=5)
                corners = []
                for p in peaks:
                    if lap_data['speed'].iloc[p] > min_corner_speed:
                        corners.append({'dist': lap_data['lap_dist'].iloc[p], 'min_speed': lap_data['speed'].iloc[p]})
                
                m, s_time = divmod(lap_sec, 60)
                ms = int((lap_sec - int(lap_sec))*1000)
                lap_time_str = f"{int(m)}:{int(s_time):02d}.{ms:03d}"
                
                processed_laps.append({
                    'filename': filename,
                    'lap_no': len(processed_laps) + 1,
                    'data': lap_data,
                    'time_str': lap_time_str,
                    'seconds': lap_sec,
                    'corners': corners
                })
    return processed_laps

def ask_ai_coach(corner_df, best_lap_time, api_key_val):
    if not api_key_val: return "⚠️ Please enter API Key"
    genai.configure(api_key=api_key_val)
    candidates = ['gemini-2.0-flash', 'gemini-1.5-flash', 'gemini-pro']
    prompt = f"""
    Act as a professional Race Engineer. Analyze corner speeds (km/h).
    Global Best Lap: {best_lap_time}
    Data:
    {corner_df.to_string()}
    1. Compare the laps. Who is faster where?
    2. Suggest improvements in Thai.
    """
    for m in candidates:
        try:
            model = genai.GenerativeModel(m)
            with st.spinner(f"AI ({m}) thinking..."):
                res = model.generate_content(prompt)
                return f"✅ **Model:** `{m}`\n\n{res.text}"
        except: continue
    return "❌ AI Error: All models failed."

# --- Helper: Resample ---
def resample_lap_data(lap, max_dist):
    common_dist = np.arange(0, max_dist, 1) 
    f_speed = interp1d(lap['data']['lap_dist'], lap['data']['speed'], bounds_error=False, fill_value="extrapolate")
    f_time = interp1d(lap['data']['lap_dist'], lap['data']['time_elapsed'], bounds_error=False, fill_value="extrapolate")
    return common_dist, f_speed(common_dist), f_time(common_dist)

# --- Main App ---
uploaded_files = st.file_uploader("📂 Upload VBO/CSV files", type=['csv', 'vbo'], accept_multiple_files=True)

if uploaded_files:
    all_laps = []
    with st.spinner("Processing files..."):
        for f in uploaded_files:
            df, err = parse_file(f)
            if df is not None:
                laps = process_laps(df, f.name)
                all_laps.extend(laps)

    if all_laps:
        all_laps_df = pd.DataFrame([{
            'File': l['filename'],
            'Lap': l['lap_no'],
            'Time': l['time_str'],
            'Seconds': l['seconds']
        } for l in all_laps])
        
        best_lap_idx = all_laps_df['Seconds'].idxmin()
        global_best_lap = all_laps[best_lap_idx]
        
        st.success(f"🏆 Reference Lap: {global_best_lap['time_str']} ({global_best_lap['filename']} L{global_best_lap['lap_no']})")

        # --- TAB LAYOUT ---
        tab1, tab2, tab3 = st.tabs(["📊 Circuit Tools View", "📉 Corner Matrix", "💬 AI Coach"])
        
        with tab1:
            st.markdown("### 🏎️ Speed & Delta Analysis")
            
            lap_options = [f"{l['filename']} - L{l['lap_no']} ({l['time_str']})" for l in all_laps]
            sorted_idx = all_laps_df.sort_values('Seconds').index.tolist()
            default_sel = [lap_options[i] for i in sorted_idx[:2]] if len(sorted_idx) > 1 else [lap_options[sorted_idx[0]]]
            
            selected_opts = st.multiselect("Select Laps to Compare:", lap_options, default=default_sel)
            
            if selected_opts:
                selected_laps_data = [all_laps[lap_options.index(opt)] for opt in selected_opts]
                max_dist = max([l['data']['lap_dist'].max() for l in selected_laps_data])
                ref_dist, ref_speed, ref_time = resample_lap_data(global_best_lap, max_dist)
                
                # --- TRICK: ใช้ secondary_y=True เพื่อแอบซ่อนกราฟ Delta ไว้ในกราฟบน ---
                fig = make_subplots(
                    rows=2, cols=1, 
                    shared_xaxes=True, 
                    vertical_spacing=0.03,
                    row_heights=[0.7, 0.3], 
                    subplot_titles=("Speed Trace (km/h)", "Time Delta (s)"),
                    specs=[[{"secondary_y": True}], [{"secondary_y": False}]] # เปิดใช้แกน Y รองสำหรับ Row 1
                )
                
                colors = px.colors.qualitative.Plotly
                
                for i, opt in enumerate(selected_opts):
                    idx = lap_options.index(opt)
                    lap = all_laps[idx]
                    color = colors[i % len(colors)]
                    is_ref = (lap == global_best_lap)
                    line_width = 3 if is_ref else 1.5
                    
                    c_dist, c_speed, c_time = resample_lap_data(lap, max_dist)
                    delta_vals = c_time - ref_time
                    
                    # 1. กราฟ Speed (แกนหลัก กราฟบน)
                    fig.add_trace(go.Scatter(
                        x=c_dist, y=c_speed, 
                        mode='lines', name=f"{opt}", 
                        line=dict(color=color, width=line_width),
                        legendgroup=opt,
                        hovertemplate='Speed: <b>%{y:.1f} km/h</b>'
                    ), row=1, col=1, secondary_y=False)
                    
                    # 2. กราฟ Delta (กราฟล่าง - เพื่อการแสดงผล)
                    fig.add_trace(go.Scatter(
                        x=c_dist, y=delta_vals, 
                        mode='lines', 
                        line=dict(color=color, width=1.5, dash='solid' if not is_ref else 'dot'),
                        fill='tozeroy' if not is_ref else None,
                        legendgroup=opt,
                        showlegend=False,
                        hoverinfo='skip' # ไม่ต้องโชว์ Hover ข้างล่าง เพราะเราจะไปดูข้างบน
                    ), row=2, col=1)

                    # 3. [The Secret Trick] กราฟ Delta ล่องหน (แอบซ่อนในกราฟบน เพื่อเอาค่า Hover)
                    fig.add_trace(go.Scatter(
                        x=c_dist, y=delta_vals,
                        mode='lines',
                        line=dict(color='rgba(0,0,0,0)'), # สีใส (มองไม่เห็น)
                        name=f"Δ {opt}", # ชื่อนี้จะขึ้นในกล่อง Hover
                        legendgroup=opt,
                        showlegend=False,
                        hovertemplate='Delta: <b>%{y:+.2f} s</b><extra></extra>' # บังคับโชว์ค่า
                    ), row=1, col=1, secondary_y=True) # ยัดใส่ Row 1 แกนรอง

                # Config
                fig.update_layout(
                    height=700, 
                    hovermode="x unified", # รวมทุกอย่างในกล่องเดียว
                    margin=dict(l=20, r=20, t=40, b=20),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )
                
                # ซ่อนแกน Y รองของกราฟบน (เพราะเราใช้แค่เก็บค่า ไม่ได้ใช้แสดงเส้น)
                fig.update_yaxes(showticklabels=False, showgrid=False, secondary_y=True, row=1, col=1)
                
                # Spike Lines
                common_spike = dict(showspikes=True, spikemode='across', spikesnap='cursor', showline=True, spikedash='solid', spikecolor="black", spikethickness=1)
                fig.update_xaxes(**common_spike, row=1, col=1)
                fig.update_xaxes(**common_spike, row=2, col=1)
                fig.update_yaxes(showspikes=False)

                st.plotly_chart(fig, use_container_width=True)

        with tab2:
            st.subheader("📉 Corner Speed Matrix")
            c_data = []
            ref_corners = global_best_lap['corners']
            for i, c in enumerate(ref_corners):
                row = {'Corner': f"T{i+1}"}
                for l in all_laps:
                    match = [x for x in l['corners'] if abs(x['dist'] - c['dist']) < 50]
                    row[f"{l['filename'][:6]}.. L{l['lap_no']}"] = match[0]['min_speed'] if match else None
                c_data.append(row)
            
            if c_data:
                c_df = pd.DataFrame(c_data).set_index('Corner')
                st.dataframe(c_df.style.highlight_max(axis=1, color='#cce5ff').format("{:.1f}"))
                st.session_state['corner_data_for_ai'] = c_df
            else:
                st.warning("No corner data detected.")

        with tab3:
            st.subheader("🤖 AI Race Engineer Feedback")
            if st.button("🧠 Analyze with AI"):
                if 'corner_data_for_ai' in st.session_state:
                    feedback = ask_ai_coach(st.session_state['corner_data_for_ai'], global_best_lap['time_str'], api_key)
                    st.markdown(feedback)
                else:
                    st.warning("No data to analyze.")