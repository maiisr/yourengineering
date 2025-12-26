import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import io
from scipy.signal import find_peaks
import google.generativeai as genai
import importlib.metadata
import time

# 1. ตั้งค่าหน้าเว็บ
st.set_page_config(page_title="ISOLDRACE: AI Coach Pro", layout="wide", page_icon="🏎️")
st.title("🏁 ISOLDRACE: AI Racing Coach System")

# --- Sidebar ---
with st.sidebar:
    st.header("⚙️ Configuration")
    
    st.markdown("### 🔑 AI Brain (Google Gemini)")
    api_key = st.text_input("Enter Gemini API Key:", type="password")
    
    try:
        ver = importlib.metadata.version("google-generativeai")
        st.caption(f"GenAI Lib Version: {ver}")
    except:
        st.caption("GenAI Lib: Unknown")

    st.divider()
    st.header("📍 Track Settings")
    if 'sf_lat' not in st.session_state: st.session_state.sf_lat = 14.957958
    if 'sf_lon' not in st.session_state: st.session_state.sf_lon = 103.085923
    
    sf_lat_input = st.number_input("Start/Finish Latitude:", value=st.session_state.sf_lat, format="%.6f", key='input_lat')
    sf_lon_input = st.number_input("Start/Finish Longitude:", value=st.session_state.sf_lon, format="%.6f", key='input_lon')
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

def parse_vbo_file(uploaded_file):
    content = uploaded_file.getvalue().decode('latin-1') 
    lines = content.splitlines()
    data_start_line = 0
    column_names = []
    is_column_section = False
    
    for i, line in enumerate(lines):
        line = line.strip()
        if '[column names]' in line:
            is_column_section = True
            continue
        if is_column_section and line.startswith('['):
            is_column_section = False
        if is_column_section and line:
            parts = line.split() 
            column_names.extend(parts)
        if '[data]' in line:
            data_start_line = i + 1
            break
            
    if not column_names or data_start_line == 0:
        return None, "รูปแบบไฟล์ VBO ไม่ถูกต้อง"

    try:
        data_str = "\n".join(lines[data_start_line:])
        df = pd.read_csv(io.StringIO(data_str), sep=r'\s+', names=column_names, engine='python')
        return df, None
    except Exception as e:
        return None, str(e)

# --- 🧠 AI Coach Function (Robust Multi-Model Try) ---
def ask_ai_coach(corner_df, best_lap_time, api_key_val):
    if not api_key_val:
        return "⚠️ กรุณาใส่ API Key ใน Sidebar ก่อนครับ"
    
    genai.configure(api_key=api_key_val)
    
    # ลำดับการลองใช้โมเดล (ถ้าตัวแรกไม่ได้ จะลองตัวถัดไป)
    # เราตัดตัว exp ออกเพราะมักจะมีปัญหากับ Free Tier
    model_candidates = [
        'gemini-2.0-flash',       # ตัวใหม่ล่าสุด (เสถียร)
        'gemini-1.5-flash',       # ตัวยอดนิยม
        'gemini-1.5-pro',         # ตัวฉลาด
        'gemini-pro',             # ตัวเก๋า
        'gemini-2.0-flash-exp'    # ตัวทดลอง (ไว้ท้ายสุด)
    ]
    
    prompt = f"""
    Act as a professional Race Engineer for a Honda Brio race car at Buriram International Circuit.
    Analyze the following cornering speed data (in km/h).
    
    Data (Rows=Corners, Cols=Lap Numbers):
    {corner_df.to_string()}
    
    Best Lap Time: {best_lap_time}
    
    Task:
    1. Identify Top 3 inconsistent corners.
    2. Give advice in Thai language.
    3. Tone: Professional, encouraging.
    """

    last_error = ""
    
    # Loop ลองทีละโมเดล
    for model_name in model_candidates:
        try:
            # ลองสร้าง Model
            model = genai.GenerativeModel(model_name)
            
            # ลองส่งคำถาม (ใส่ timeout กันค้าง)
            with st.spinner(f'🤖 กำลังลองเชื่อมต่อสมอง: {model_name}...'):
                response = model.generate_content(prompt)
                
                # ถ้าตอบกลับมาสำเร็จ ให้ส่งผลลัพธ์เลย
                return f"✅ **AI Model Used:** `{model_name}`\n\n" + response.text
                
        except Exception as e:
            # ถ้า Error ให้เก็บไว้ แล้วลองตัวถัดไป
            error_msg = str(e)
            if "429" in error_msg:
                print(f"Model {model_name} rate limited.")
            elif "404" in error_msg:
                print(f"Model {model_name} not found.")
            last_error = error_msg
            continue # ไปลองตัวถัดไปในลิสต์
            
    # ถ้าลองครบทุกตัวแล้วยังไม่ได้
    return f"❌ ขออภัยครับ ลองเปลี่ยนโมเดลครบทุกตัวแล้วยังไม่สำเร็จ\nError ล่าสุด: {last_error}\n\n💡 แนะนำ: รอสัก 1 นาทีแล้วกดใหม่ (อาจติด Rate Limit ชั่วคราว)"

# --- Main Logic ---
uploaded_file = st.file_uploader("อัปโหลดไฟล์ Datalogger (CSV หรือ VBO)", type=['csv', 'vbo'])

if uploaded_file is not None:
    filename = uploaded_file.name.lower()
    df = None
    
    if filename.endswith('.vbo'):
        df, error_msg = parse_vbo_file(uploaded_file)
    else:
        try:
            df = pd.read_csv(uploaded_file)
            error_msg = None
        except Exception as e:
            error_msg = str(e)

    if df is not None:
        df.columns = df.columns.str.lower()
        col_cols = df.columns.tolist()
        def get_idx(keywords):
            for k in keywords:
                for c in col_cols:
                    if k in c: return col_cols.index(c)
            return 0

        c1, c2, c3 = st.columns(3)
        with c1: speed_col = st.selectbox("Speed:", col_cols, index=get_idx(['vel', 'speed', 'kmh']))
        with c2: lat_col = st.selectbox("Latitude:", col_cols, index=get_idx(['lat']))
        with c3: lon_col = st.selectbox("Longitude:", col_cols, index=get_idx(['lon', 'long']))

        if st.button("🚀 เริ่มวิเคราะห์ (Analyze Data)", type="primary"):
            with st.spinner("กำลังประมวลผล..."):
                work_df = df.copy()
                work_df['speed'] = work_df[speed_col]
                work_df['lat'] = work_df[lat_col].apply(smart_coord_convert)
                work_df['lon'] = work_df[lon_col].apply(smart_coord_convert)
                
                if work_df['lon'].mean() < 0 and TRACK_CONFIG['sf_lon'] > 0:
                     work_df['lon'] = work_df['lon'].abs()

                work_df['dist_to_sf'] = dist_from_sf(work_df['lat'], work_df['lon'], TRACK_CONFIG['sf_lat'], TRACK_CONFIG['sf_lon'])
                
                is_near_sf = (work_df['dist_to_sf'] < TRACK_CONFIG['sf_radius_m']) & (work_df['speed'] > 10)
                sf_indices = work_df[is_near_sf].index.tolist()
                
                final_sf = []
                if sf_indices:
                    final_sf = [sf_indices[0]]
                    for idx in sf_indices[1:]:
                        if idx - final_sf[-1] > 100: final_sf.append(idx)
                
                laps = []
                lap_summaries = []
                
                if len(final_sf) > 1:
                    for i in range(len(final_sf)-1):
                        s_idx = final_sf[i]
                        e_idx = final_sf[i+1]
                        lap_data = work_df.iloc[s_idx:e_idx].copy()
                        lap_sec = len(lap_data) * 0.1
                        
                        if lap_sec > 30 and lap_sec < 600:
                            lap_data['lap_dist'] = (lap_data['speed'] / 3.6 * 0.1).cumsum()
                            lap_data['lap_no'] = len(laps) + 1
                            
                            speed_inv = -lap_data['speed'].values
                            peaks, _ = find_peaks(speed_inv, distance=50, prominence=5)
                            
                            corners = []
                            for p in peaks:
                                spd = lap_data['speed'].iloc[p]
                                dst = lap_data['lap_dist'].iloc[p]
                                if spd > min_corner_speed:
                                    corners.append({'dist': dst, 'min_speed': spd})
                            
                            laps.append({'data': lap_data, 'corners': corners})
                            
                            m, s = divmod(lap_sec, 60)
                            ms = int((lap_sec - int(lap_sec))*1000)
                            lap_summaries.append({
                                'Lap': len(laps),
                                'Time': f"{int(m)}:{int(s):02d}.{ms:03d}",
                                'Seconds': lap_sec,
                                'Top Speed': lap_data['speed'].max(),
                                'Corner Count': len(corners)
                            })

                if laps:
                    lap_df = pd.DataFrame(lap_summaries)
                    best_idx = lap_df['Seconds'].idxmin()
                    best_lap_no = lap_df.loc[best_idx, 'Lap']
                    best_lap_time = lap_df.loc[best_idx, 'Time']
                    best_lap_data = [l for l in laps if l['data']['lap_no'].iloc[0] == best_lap_no][0]

                    c_df = pd.DataFrame() 
                    if len(laps) > 0:
                        ref_corners = best_lap_data['corners']
                        corner_data = []
                        for i, c in enumerate(ref_corners):
                            corner_row = {'Corner': f"T{i+1}"}
                            for l in laps:
                                l_no = l['data']['lap_no'].iloc[0]
                                match = [x for x in l['corners'] if abs(x['dist'] - c['dist']) < 50]
                                if match:
                                    corner_row[f"Lap {l_no}"] = match[0]['min_speed']
                                else:
                                    corner_row[f"Lap {l_no}"] = None
                            corner_data.append(corner_row)
                        c_df = pd.DataFrame(corner_data).set_index('Corner')

                    st.session_state['analysis_done'] = True
                    st.session_state['laps'] = laps
                    st.session_state['lap_df'] = lap_df
                    st.session_state['c_df'] = c_df
                    st.session_state['best_lap_time'] = best_lap_time
                    st.session_state['best_lap_no'] = best_lap_no
                else:
                    st.session_state['analysis_done'] = False
                    st.warning("⚠️ ไม่พบข้อมูลรอบวิ่งที่สมบูรณ์")

        # Display Section
        if st.session_state.get('analysis_done'):
            laps = st.session_state['laps']
            lap_df = st.session_state['lap_df']
            c_df = st.session_state['c_df']
            best_lap_time = st.session_state['best_lap_time']
            best_lap_no = st.session_state['best_lap_no']

            st.success(f"📍 Analysis Complete! Found {len(laps)} laps. Best Lap: {best_lap_time}")

            tab1, tab2, tab3, tab4 = st.tabs(["⏱️ Laps", "📉 Corner Matrix", "🏎️ Ghost Car", "💬 AI Coach Feedback"])

            with tab1:
                st.dataframe(lap_df.style.highlight_min(subset=['Seconds'], color='#d4edda'))

            with tab2:
                st.dataframe(c_df.style.highlight_max(axis=1, color='#cce5ff').format("{:.1f}"))

            with tab3:
                sel = st.multiselect("Select Laps:", lap_df['Lap'].tolist(), default=[best_lap_no])
                if sel:
                    fig = go.Figure()
                    for Ln in sel:
                        ld = [l for l in laps if l['data']['lap_no'].iloc[0] == Ln][0]
                        is_best = (Ln == best_lap_no)
                        fig.add_trace(go.Scatter(x=ld['data']['lap_dist'], y=ld['data']['speed'], mode='lines', name=f"Lap {Ln}", line=dict(width=3 if is_best else 1)))
                    st.plotly_chart(fig, use_container_width=True)

            with tab4:
                st.subheader("🤖 AI Race Engineer Feedback")
                if st.button("🧠 ขอคำแนะนำจาก AI Coach", type="primary"):
                    if api_key:
                        feedback = ask_ai_coach(c_df, best_lap_time, api_key)
                        st.markdown(feedback)
                    else:
                        st.warning("⚠️ กรุณาใส่ API Key")
    else:
        if error_msg: st.error(f"Error: {error_msg}")