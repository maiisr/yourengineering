import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import io
from scipy.signal import find_peaks, savgol_filter
from scipy.interpolate import interp1d
import google.generativeai as genai
import importlib.metadata

# 1. ตั้งค่าหน้าเว็บ
st.set_page_config(page_title="ISOLDRACE: Circuit Tools Edition", layout="wide", page_icon="🏎️")
st.title("🏁 ISOLDRACE: Pro Analysis (Circuit Tools Edition)")

# --- 🏁 ข้อมูลสนามช้าง (BURIRAM OFFICIAL) 🏁 ---
BURIRAM_SF = {'lat': 14.957958, 'lon': 103.085923} 
SECTOR_SPLITS = [1500, 3000] 

# ระยะทางของแต่ละโค้ง (Gate) สำหรับจับความเร็วต่ำสุด
BURIRAM_GATES = {
    "T1 (โค้งขวาแรก)": (200, 600),    "T2 (ทางตรงยาว)": (800, 1100),   "T3 (ยูเทิร์นขวา)": (1150, 1550),
    "T4 (ซ้ายความเร็วสูง)": (1600, 1850), "T5 (ซ้ายเข้าใน)": (1900, 2150),  "T6 (ขวาสั้น)": (2200, 2400),
    "T7 (ขวาหักศอก)": (2450, 2700),  "T8 (ซ้ายหักศอก)": (2750, 3050),  "T9 (ซ้ายเนิน)": (3100, 3350),
    "T10 (ขวาเร็ว)": (3400, 3650),  "T11 (ขวาแคบ)": (3700, 4000),  "T12 (ยูเทิร์นสุดท้าย)": (4100, 4500)
}

# --- Sidebar ---
with st.sidebar:
    st.header("⚙️ การตั้งค่า (Configuration)")
    api_key = st.text_input("🔑 ใส่ Gemini API Key (ถ้ามี):", type="password")
    
    st.divider()
    st.header("📍 ปรับจูนตำแหน่ง (Alignment)")
    
    # Slider ปรับ Offset ระยะทาง (สำคัญมากสำหรับจูนกราฟให้ตรง Circuit Tools)
    dist_offset = st.slider("ขยับกราฟซ้าย-ขวา (เมตร):", -50.0, 50.0, 0.0, step=0.5)
    
    # ตัวกรองรอบ
    min_lap_time = st.number_input("เวลาต่อรอบขั้นต่ำ (วินาที):", value=90, help="กรองรอบ Out Lap ทิ้ง")
    
    TRACK_CONFIG = {
        'sf_lat': BURIRAM_SF['lat'],
        'sf_lon': BURIRAM_SF['lon'],
        'sf_radius_m': 60 # เพิ่มรัศมีเผื่อ GPS เพี้ยน
    }

# --- Core Functions ---
def smart_coord_convert(val):
    if pd.isna(val) or val == 0: return val
    if abs(val) <= 180: return val
    degrees = int(val / 100)
    minutes = abs(val) % 100
    if minutes >= 60 or abs(degrees) > 180: return val / 60.0
    decimal = degrees + (minutes / 60)
    if val < 0: decimal = -decimal
    return decimal

def dist_from_point(lat1, lon1, lat2, lon2):
    R = 6371000
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlambda/2)**2
    return 2 * R * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

def parse_file(uploaded_file):
    """อ่านไฟล์ VBO แบบทนทาน (Robust Parser)"""
    filename = uploaded_file.name.lower()
    content = uploaded_file.getvalue().decode('latin-1') 
    
    if filename.endswith('.vbo'):
        lines = content.splitlines()
        data_start = 0
        cols = []
        is_col = False # Flag เพื่อบอกว่ากำลังอ่านชื่อคอลัมน์
        
        for i, line in enumerate(lines):
            line = line.strip()
            if '[column names]' in line:
                is_col = True
                continue
            
            if is_col:
                if line.startswith('['): # จบโซนชื่อคอลัมน์
                    is_col = False
                else:
                    # อ่านชื่อคอลัมน์ (รองรับทั้งแนวนอนและแนวตั้ง)
                    cols.extend(line.split())
            
            if '[data]' in line:
                data_start = i + 1
                break
        
        if not cols or data_start == 0: return None, "Invalid VBO Structure"
        
        try:
            # ใช้ on_bad_lines='skip' เพื่อข้ามบรรทัดที่เสีย
            df = pd.read_csv(io.StringIO("\n".join(lines[data_start:])), sep=r'\s+', names=cols, engine='python', on_bad_lines='skip')
            return df, None
        except Exception as e: return None, str(e)
    else:
        try:
            df = pd.read_csv(uploaded_file)
            return df, None
        except Exception as e: return None, str(e)

def process_laps(df, filename):
    try:
        df.columns = df.columns.str.lower()
        cols = df.columns.tolist()
        
        # Mapping Columns
        speed_c = next((c for c in cols if c in ['vel', 'speed', 'kmh', 'velocity']), None)
        lat_c = next((c for c in cols if c in ['lat', 'latitude']), None)
        lon_c = next((c for c in cols if c in ['lon', 'long', 'longitude']), None)
        
        if not speed_c or not lat_c: return []

        work_df = df.copy()
        work_df['speed'] = work_df[speed_c]
        work_df['lat'] = work_df[lat_c].apply(smart_coord_convert)
        work_df['lon'] = work_df[lon_c].apply(smart_coord_convert)
        if work_df['lon'].mean() < 0 and TRACK_CONFIG['sf_lon'] > 0: work_df['lon'] = work_df['lon'].abs()

        # ป้องกันไฟล์สั้นเกินไปจน Savgol Error
        if len(work_df) < 50: return []

        # --- Physics Calculation (Smoothed for Circuit Tools look) ---
        v_ms = work_df['speed'] / 3.6
        v_ms_smooth = savgol_filter(v_ms, 15, 2) # ลด Noise ก่อนคำนวณ
        
        # Longitudinal G
        work_df['long_g'] = savgol_filter(np.gradient(v_ms_smooth, 0.1) / 9.81, 25, 3)
        
        # Lateral G (Heading based)
        lat_rad = np.radians(work_df['lat']); lon_rad = np.radians(work_df['lon'])
        dlat = np.gradient(lat_rad); dlon = np.gradient(lon_rad)
        heading = np.arctan2(dlon * np.cos(lat_rad), dlat)
        d_heading = np.gradient(np.unwrap(heading)) / 0.1 
        work_df['lat_g'] = savgol_filter((v_ms_smooth * d_heading / 9.81) * -1, 25, 3)

        # --- Lap Detection ---
        work_df['dist_to_sf'] = dist_from_point(work_df['lat'], work_df['lon'], TRACK_CONFIG['sf_lat'], TRACK_CONFIG['sf_lon'])
        inv_dist = -work_df['dist_to_sf'].values
        peaks, _ = find_peaks(inv_dist, height=-TRACK_CONFIG['sf_radius_m'], distance=200)
        final_sf = list(peaks)
        
        processed_laps = []
        lap_counter = 1
        
        if len(final_sf) > 1:
            for i in range(len(final_sf)-1):
                s, e = final_sf[i], final_sf[i+1]
                lap_data = work_df.iloc[s:e].copy()
                lap_sec = len(lap_data) * 0.1
                
                # Filter Out Laps
                if lap_sec > min_lap_time:
                    # 1. Reset Distance to 0.0 (หัวใจสำคัญของ Circuit Tools Alignment)
                    lap_data['lap_dist'] = (lap_data['speed'] / 3.6 * 0.1).cumsum()
                    lap_data['lap_dist'] = lap_data['lap_dist'] - lap_data['lap_dist'].iloc[0]
                    lap_data['time_elapsed'] = np.arange(len(lap_data)) * 0.1
                    
                    # 2. Sectors
                    try:
                        f_time = interp1d(lap_data['lap_dist'], lap_data['time_elapsed'], bounds_error=False, fill_value="extrapolate")
                        sectors = {
                            'S1': f_time(SECTOR_SPLITS[0]),
                            'S2': f_time(SECTOR_SPLITS[1]) - f_time(SECTOR_SPLITS[0]),
                            'S3': lap_data['time_elapsed'].max() - f_time(SECTOR_SPLITS[1])
                        }
                    except:
                        sectors = {'S1': 0, 'S2': 0, 'S3': 0}

                    # 3. Corners (Fixed Gates)
                    corners_found = {}
                    for c_name, (start_m, end_m) in BURIRAM_GATES.items():
                        segment = lap_data[(lap_data['lap_dist'] >= start_m) & (lap_data['lap_dist'] <= end_m)]
                        if not segment.empty:
                            min_idx = segment['speed'].idxmin()
                            corners_found[c_name] = {
                                'min_speed': segment['speed'].loc[min_idx]
                            }
                        else: corners_found[c_name] = None

                    # Check Out Lap again (Avg speed)
                    if lap_data['speed'].mean() > 60:
                        m, s_time = divmod(lap_sec, 60)
                        ms = int((lap_sec - int(lap_sec))*1000)
                        lap_time_str = f"{int(m)}:{int(s_time):02d}.{ms:03d}"
                        
                        processed_laps.append({
                            'filename': filename,
                            'lap_no': lap_counter,
                            'data': lap_data,
                            'time_str': lap_time_str,
                            'seconds': lap_sec,
                            'sectors': sectors,
                            'corners': corners_found
                        })
                        lap_counter += 1
        return processed_laps
    except Exception as e:
        return []

def resample_lap_data(lap, max_dist, offset=0):
    common_dist = np.arange(0, max_dist, 1) 
    x_dist = lap['data']['lap_dist'] + offset
    
    f_speed = interp1d(x_dist, lap['data']['speed'], bounds_error=False, fill_value="extrapolate")
    f_time = interp1d(x_dist, lap['data']['time_elapsed'], bounds_error=False, fill_value="extrapolate")
    f_long = interp1d(x_dist, lap['data']['long_g'], bounds_error=False, fill_value="extrapolate")
    f_lat = interp1d(x_dist, lap['data']['lat_g'], bounds_error=False, fill_value="extrapolate")
    
    return common_dist, f_speed(common_dist), f_time(common_dist), f_long(common_dist), f_lat(common_dist)

def ask_ai_coach(prompt_text, api_key_val):
    if not api_key_val: return "⚠️ กรุณาใส่ API Key ในเมนูด้านซ้ายก่อนครับ"
    genai.configure(api_key=api_key_val)
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        with st.spinner(f"🤖 AI Coach กำลังคิดวิเคราะห์..."):
            res = model.generate_content(prompt_text)
            return f"✅ **คำแนะนำจาก AI:**\n\n{res.text}"
    except Exception as e: return f"❌ เกิดข้อผิดพลาด: {e}"

def analyze_corner_performance(c_df):
    analysis_results = []
    for corner_name, speeds in c_df.iterrows():
        valid_speeds = pd.to_numeric(speeds, errors='coerce').dropna()
        if len(valid_speeds) > 0:
            max_speed = valid_speeds.max()
            avg_speed = valid_speeds.mean()
            std_dev = valid_speeds.std() if len(valid_speeds) > 1 else 0
            potential_loss = max_speed - avg_speed
            analysis_results.append({
                'Corner': corner_name,
                'Potential Loss': potential_loss,
                'Consistency': std_dev
            })
    return pd.DataFrame(analysis_results)

# --- Main App ---
uploaded_files = st.file_uploader("📂 อัปโหลดไฟล์ VBO/CSV ที่นี่", type=['csv', 'vbo'], accept_multiple_files=True)

if uploaded_files:
    all_laps = []
    with st.spinner("⏳ กำลังอ่านไฟล์และคำนวณข้อมูล..."):
        for f in uploaded_files:
            df, err = parse_file(f)
            if df is not None:
                laps = process_laps(df, f.name)
                all_laps.extend(laps)
            else:
                st.error(f"❌ อ่านไฟล์ {f.name} ไม่ได้: {err}")

    if all_laps:
        all_laps_df = pd.DataFrame([{
            'File': l['filename'], 'Lap': l['lap_no'], 'Time': l['time_str'], 'Seconds': l['seconds'],
            'S1': l['sectors']['S1'], 'S2': l['sectors']['S2'], 'S3': l['sectors']['S3']
        } for l in all_laps])
        
        best_lap_idx = all_laps_df['Seconds'].idxmin()
        global_best_lap = all_laps[best_lap_idx]
        
        ideal_s1 = all_laps_df['S1'].min()
        ideal_s2 = all_laps_df['S2'].min()
        ideal_s3 = all_laps_df['S3'].min()
        ideal_total = ideal_s1 + ideal_s2 + ideal_s3
        gain = global_best_lap['seconds'] - ideal_total
        
        m_i, s_i = divmod(ideal_total, 60)
        ms_i = int((ideal_total - int(ideal_total))*1000)
        ideal_str = f"{int(m_i)}:{int(s_i):02d}.{ms_i:03d}"

        lap_options = [f"{l['filename']} - L{l['lap_no']} ({l['time_str']})" for l in all_laps]
        
        # --- SESSION SUMMARY ---
        st.markdown("### 🏁 สรุปผลการขับ (Session Summary)")
        col1, col2, col3 = st.columns(3)
        with col1: 
            st.metric("🏆 เวลาที่ดีที่สุด (Best Lap)", global_best_lap['time_str'], f"Lap {global_best_lap['lap_no']}")
        with col2: 
            st.metric("✨ เวลาในอุดมคติ (Ideal Lap)", ideal_str, delta=f"-{gain:.3f} s", delta_color="inverse")
            st.caption("ถ้ารวม Sector ที่ดีที่สุดเข้าด้วยกัน")
        with col3: 
            st.metric("🚀 เวลาที่ลดได้อีก (Potential Gain)", f"{gain:.3f} s", "โอกาสทำเวลาให้ดีขึ้น")

        # --- TABS ---
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 กราฟความเร็ว (Speed & Delta)", 
            "🍳 วงกลมแรงเกาะ (G-Force)", 
            "📉 เจาะลึกโค้ง (Corner Matrix)", 
            "🏆 ช่วงสนาม (Sectors)", 
            "💬 โค้ช AI (AI Coach)"
        ])
        
        # TAB 1: Speed & Delta
        with tab1:
            st.markdown("### 🏎️ เปรียบเทียบความเร็วและเวลา (Circuit Tools View)")
            st.info("💡 **วิธีดู:** \n1. **กราฟบน (Speed):** ใครอยู่สูงกว่า = เร็วกว่า \n2. **กราฟกลาง (Delta):** ถ้ากราฟชี้ขึ้น = เราช้ากว่า, ชี้ลง = เราเร็วกว่า \n3. **กราฟล่าง (G-Force):** แรงเบรกต้องลงลึกและชัน")
            
            sorted_idx = all_laps_df.sort_values('Seconds').index.tolist()
            default_sel = [lap_options[i] for i in sorted_idx[:2]] if len(sorted_idx) > 1 else [lap_options[sorted_idx[0]]]
            
            selected_opts = st.multiselect("เลือกรอบที่จะเปรียบเทียบ:", lap_options, default=default_sel, key='tab1_sel')
            
            if selected_opts:
                selected_laps_data = [all_laps[lap_options.index(opt)] for opt in selected_opts]
                max_dist = max([l['data']['lap_dist'].max() for l in selected_laps_data])
                ref_dist, ref_speed, ref_time, ref_long, ref_lat = resample_lap_data(global_best_lap, max_dist, 0)
                
                fig = make_subplots(
                    rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.5, 0.25, 0.25], 
                    subplot_titles=("ความเร็ว (Speed km/h)", "ส่วนต่างเวลา (Delta Time s)", "แรงเบรก/เร่ง (Longitudinal G)")
                )
                
                colors = px.colors.qualitative.Plotly
                for i, opt in enumerate(selected_opts):
                    idx = lap_options.index(opt)
                    lap = all_laps[idx]
                    color = colors[i % len(colors)]
                    
                    # Apply Manual Offset from Sidebar
                    offset_val = st.session_state.get('dist_offset', 0.0) if lap != global_best_lap else 0.0
                    c_dist, c_speed, c_time, c_long, c_lat = resample_lap_data(lap, max_dist, offset_val)
                    
                    fig.add_trace(go.Scatter(x=c_dist, y=c_speed, mode='lines', name=opt, line=dict(color=color, width=2), legendgroup=opt), row=1, col=1)
                    fig.add_trace(go.Scatter(x=c_dist, y=c_time-ref_time, mode='lines', line=dict(color=color, width=1.5), fill='tozeroy', showlegend=False, legendgroup=opt), row=2, col=1)
                    fig.add_trace(go.Scatter(x=c_dist, y=c_long, mode='lines', line=dict(color=color, width=1.5), showlegend=False, legendgroup=opt), row=3, col=1)

                fig.update_layout(height=800, hovermode="x unified", margin=dict(l=10, r=10, t=30, b=10))
                # Spike Lines (Cursor)
                common_spike = dict(showspikes=True, spikemode='across', spikesnap='cursor', showline=True, spikedash='solid', spikecolor="black", spikethickness=1)
                fig.update_xaxes(**common_spike, row=1, col=1); fig.update_xaxes(**common_spike, row=2, col=1); fig.update_xaxes(**common_spike, row=3, col=1)
                st.plotly_chart(fig, use_container_width=True)

        # TAB 2: Friction Circle
        with tab2:
            st.markdown("#### 🍳 วงกลมแรงเกาะ (Friction Circle)")
            st.info("💡 **ทริค:** ดูกราฟจุดกระจายตัว (Scatter Plot) เพื่อเช็คการใช้ยาง\n- **วงกลมป่องๆ:** ✅ ใช้ยางคุ้มค่า (เบรกพร้อมเลี้ยว / Trail Braking)\n- **สี่เหลี่ยมข้าวหลามตัด:** ❌ เบรกเสร็จแล้วค่อยเลี้ยว (เสียเวลา)")
            
            g_opts = st.multiselect("เลือกรอบเพื่อดู G-G Diagram:", lap_options, default=default_sel, key='tab2_sel')
            if g_opts:
                col1, col2 = st.columns([2, 1])
                with col1:
                    fig_gg = go.Figure()
                    for r in [0.5, 1.0, 1.5]: fig_gg.add_shape(type="circle", xref="x", yref="y", x0=-r, y0=-r, x1=r, y1=r, line_color="LightGrey")
                    
                    for i, opt in enumerate(g_opts):
                        idx = lap_options.index(opt); lap = all_laps[idx]
                        fig_gg.add_trace(go.Scatter(x=lap['data']['lat_g'], y=lap['data']['long_g'], mode='markers', name=opt, marker=dict(size=4, opacity=0.5)))
                    
                    fig_gg.update_layout(
                        width=600, height=600, 
                        xaxis=dict(range=[-2,2], title="แรงเหวี่ยงข้าง (Lateral G) [ซ้าย/ขวา]"), 
                        yaxis=dict(range=[-2,2], title="แรงเบรก/เร่ง (Longitudinal G) [เร่ง/เบรก]"),
                        template="plotly_white"
                    )
                    st.plotly_chart(fig_gg, use_container_width=True)

        # TAB 3: Corner Matrix
        with tab3:
            st.markdown("### 📉 วิเคราะห์ความเร็วในโค้ง (Corner Speed Matrix)")
            st.info("ตารางนี้แสดง **ความเร็วต่ำสุด (Min Speed)** ในแต่ละโค้ง ช่วยบอกว่าคุณเข้าโค้งไหนช้าหรือเร็วเมื่อเทียบกับรอบอื่นๆ")
            
            c_data = []
            for c_name in BURIRAM_GATES.keys():
                row = {'Corner': c_name}
                for l in all_laps:
                    val = l['corners'].get(c_name)
                    row[f"{l['filename'][:6]}.. L{l['lap_no']}"] = val['min_speed'] if val else None
                c_data.append(row)
            
            if c_data:
                c_df = pd.DataFrame(c_data).set_index('Corner')
                
                # Auto Analysis
                st.markdown("#### ⚡ บทสรุปจุดแข็ง/จุดอ่อน")
                analysis_df = analyze_corner_performance(c_df)
                if not analysis_df.empty:
                    worst = analysis_df.sort_values('Potential Loss', ascending=False).head(3)
                    col_a, col_b = st.columns(2)
                    with col_a: 
                        st.error("🚨 **จุดที่ควรแก้ด่วน (Critical Corners):**")
                        for _, r in worst.iterrows(): st.markdown(f"**{r['Corner']}**: ความเร็วหายไป {r['Potential Loss']:.1f} km/h")
                    with col_b:
                        st.success("✅ **จุดที่ทำได้ดี (Consistency):**")
                        best = analysis_df.sort_values('Consistency').head(3)
                        for _, r in best.iterrows(): st.markdown(f"**{r['Corner']}**: ขับได้นิ่งมาก (ผันผวน ±{r['Consistency']:.1f})")
                
                st.dataframe(c_df.style.highlight_max(axis=1, color='#cce5ff').format("{:.1f}"), use_container_width=True)
                st.session_state['corner_data_for_ai'] = c_df

        # TAB 4: Sectors
        with tab4:
            st.markdown("#### 📊 เวลาแต่ละช่วงสนาม (Sector Performance)")
            st.info("แบ่งสนามเป็น 3 ช่วง: **S1** (ทางตรง), **S2** (ช่วงเลี้ยวเยอะ), **S3** (โค้งสุดท้าย) เพื่อดูว่าเราเสียเวลาช่วงไหน")
            
            def highlight_sectors(row):
                styles = [''] * len(row)
                if abs(row['S1'] - ideal_s1) < 0.001: styles[4] = 'color: purple; font-weight: bold; background-color: #f3e5f5' 
                if abs(row['S2'] - ideal_s2) < 0.001: styles[5] = 'color: purple; font-weight: bold; background-color: #f3e5f5'
                if abs(row['S3'] - ideal_s3) < 0.001: styles[6] = 'color: purple; font-weight: bold; background-color: #f3e5f5'
                return styles
                
            disp_df = all_laps_df[['File', 'Lap', 'Time', 'Seconds', 'S1', 'S2', 'S3']].copy()
            st.dataframe(disp_df.style.apply(highlight_sectors, axis=1).format({'Seconds': '{:.3f}', 'S1': '{:.3f}', 'S2': '{:.3f}', 'S3': '{:.3f}'}), use_container_width=True)

        # TAB 5: AI Coach
        with tab5:
            st.subheader("🤖 ปรึกษาโค้ช AI (AI Race Engineer)")
            st.caption("กดปุ่มด้านล่างเพื่อให้ AI วิเคราะห์ข้อมูลทั้งหมดและแนะนำเทคนิคการขับ (ต้องใส่ API Key ก่อน)")
            
            if st.button("🧠 วิเคราะห์การขับเดี๋ยวนี้"):
                if 'corner_data_for_ai' in st.session_state:
                    prompt = f"""
                    Role: Professional Race Engineer. Language: Thai (Speak like a supportive coach).
                    Analyze this driver's session data at Chang International Circuit.
                    - Best Lap: {global_best_lap['time_str']}
                    - Potential Gain: {gain:.3f}s
                    
                    Corner Minimum Speeds (km/h):
                    {st.session_state['corner_data_for_ai'].to_string()}
                    
                    Task:
                    1. Identify the top 3 corners where the driver is inconsistent or losing speed compared to their best.
                    2. Give specific advice on braking or racing line for those corners.
                    3. Summarize the overall driving style based on the data.
                    """
                    feedback = ask_ai_coach(prompt, api_key)
                    st.markdown(feedback)
                else: st.warning("⚠️ กรุณารอให้ระบบประมวลผลข้อมูลเสร็จสิ้นก่อนครับ")
