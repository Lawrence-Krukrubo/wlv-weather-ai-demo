"""
⛅ Shawbury Weather AI Predictor (Streamlit App)
Developed for University of Wolverhampton — Computer Science Demo for Sandwell College
Dataset: Met Office 10-Year Record for Shawbury (2010–2019)
"""

import io
import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import streamlit as st

# ---------------------------------------------------------
# 1. Page Configuration & Styling
# ---------------------------------------------------------
st.set_page_config(
    page_title="Shawbury Weather AI Predictor",
    page_icon="⛅",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom header styling — University Gold Banner
st.markdown("""
    <style>
    .gold-header-banner {
        background: linear-gradient(135deg, #FFFBEB 0%, #FEF08A 25%, #FACC15 65%, #EAB308 100%);
        border: 2px solid #CA8A04;
        border-radius: 14px;
        padding: 22px 28px;
        margin-bottom: 25px;
        box-shadow: 0 8px 20px rgba(202, 138, 4, 0.22);
    }
    .gold-main-title {
        font-size: 2.25rem;
        font-weight: 800;
        color: #0F172A;
        margin: 0 0 6px 0;
        letter-spacing: -0.5px;
        line-height: 1.2;
    }
    .gold-sub-title {
        font-size: 1.15rem;
        font-weight: 600;
        color: #1E293B;
        margin: 0;
        line-height: 1.35;
    }
    .metric-card {
        background-color: #F3F4F6;
        border-radius: 10px;
        padding: 15px;
        border-left: 5px solid #2563EB;
    }
    </style>
    <div class="gold-header-banner">
        <div class="gold-main-title">⛅ Shawbury Weather AI Predictor</div>
        <div class="gold-sub-title">University of Wolverhampton — Computer Science & AI Interactive Demo for Sandwell College</div>
    </div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# 2. Dataset Loading (Self-Contained Fallback + CSV/URL Support)
# ---------------------------------------------------------
EMBEDDED_SHAWBURY_CSV = """year,month,tmax,tmin,rain,sun,af
2010,1,4.0,-1.9,57.4,61.7,20
2010,2,5.9,-0.9,25.8,53.0,18
2010,3,10.4,1.1,35.2,122.5,12
2010,4,14.0,3.0,19.6,178.8,6
2010,5,15.4,5.6,37.0,183.5,1
2010,6,20.8,9.7,38.6,227.9,0
2010,7,20.7,12.8,74.6,86.1,0
2010,8,19.1,10.6,48.2,135.7,0
2010,9,18.0,9.6,59.0,131.8,0
2010,10,14.0,5.5,67.8,113.8,4
2010,11,8.0,0.8,34.6,75.4,13
2010,12,2.0,-5.8,28.9,41.2,23
2011,1,6.4,0.2,49.2,50.2,16
2011,2,9.5,3.5,47.0,48.8,5
2011,3,10.8,1.1,14.6,131.5,12
2011,4,16.9,5.6,4.8,190.5,1
2011,5,16.8,7.5,45.0,167.3,0
2011,6,18.7,8.1,57.2,184.9,0
2011,7,20.2,10.3,47.8,166.8,0
2011,8,20.4,11.0,27.4,116.6,0
2011,9,20.0,10.6,18.6,131.2,0
2011,10,16.6,8.4,31.8,99.8,0
2011,11,12.2,5.9,34.0,46.4,2
2011,12,9.3,2.8,77.8,50.2,9
2012,1,8.8,1.6,43.6,74.1,9
2012,2,7.6,0.9,19.2,50.8,14
2012,3,13.0,3.2,16.6,153.1,4
2012,4,11.1,2.8,112.4,110.1,6
2012,5,16.5,6.9,50.8,189.2,1
2012,6,17.4,10.0,82.6,83.8,0
2012,7,19.3,11.4,128.2,140.8,0
2012,8,20.3,11.7,86.2,128.3,0
2012,9,17.0,7.8,116.2,140.1,0
2012,10,12.7,5.1,69.6,95.0,3
2012,11,9.8,1.9,86.8,63.5,10
2012,12,7.7,0.7,133.0,51.2,10
2013,1,6.0,0.6,73.1,30.5,18
2013,2,6.1,-0.2,51.6,66.9,15
2013,3,5.7,-1.0,64.1,93.4,19
2013,4,11.8,2.1,10.8,139.8,9
2013,5,14.7,5.5,84.0,163.7,0
2013,6,18.3,8.4,28.4,163.0,0
2013,7,24.1,12.1,59.2,257.4,0
2013,8,20.8,11.5,66.2,125.5,0
2013,9,17.9,9.3,33.4,106.4,0
2013,10,15.6,9.0,96.6,75.3,0
2013,11,9.5,2.4,66.0,67.0,8
2013,12,9.8,2.4,54.8,55.3,8
2014,1,8.5,1.9,105.8,51.5,8
2014,2,9.0,3.2,76.6,80.0,2
2014,3,12.0,2.4,41.4,140.0,7
2014,4,14.5,5.5,37.0,140.9,1
2014,5,16.4,8.0,81.4,144.2,0
2014,6,19.8,10.0,51.2,208.7,0
2014,7,22.9,11.7,32.8,248.0,0
2014,8,19.0,10.2,89.2,158.9,0
2014,9,19.4,9.1,17.0,123.9,0
2014,10,15.6,7.9,69.0,71.7,0
2014,11,11.1,4.8,76.8,60.0,6
2014,12,8.3,1.4,72.8,72.1,12
2015,1,7.7,0.8,60.3,73.2,12
2015,2,7.2,0.7,25.4,55.9,12
2015,3,10.2,2.0,49.2,110.2,8
2015,4,14.2,3.2,12.8,222.8,3
2015,5,14.6,6.6,66.4,152.5,1
2015,6,19.2,8.7,40.4,236.8,0
2015,7,20.0,10.9,58.4,156.1,0
2015,8,20.1,11.0,77.0,128.0,0
2015,9,16.9,7.3,52.2,154.9,0
2015,10,14.7,6.3,35.6,84.8,0
2015,11,12.5,6.0,87.2,30.8,3
2015,12,12.6,6.9,84.6,28.1,0
2016,1,8.7,2.1,102.6,40.5,6
2016,2,8.4,0.7,51.6,87.4,11
2016,3,9.6,1.2,65.8,107.1,10
2016,4,11.8,2.6,74.8,165.0,8
2016,5,17.4,6.6,47.2,212.8,1
2016,6,19.6,11.1,117.8,126.2,0
2016,7,20.6,11.9,39.2,154.5,0
2016,8,21.4,12.3,44.0,175.2,0
2016,9,19.6,11.9,50.8,129.2,0
2016,10,14.3,6.5,21.8,106.1,0
2016,11,9.0,1.6,78.4,75.7,9
2016,12,9.3,1.7,31.6,46.9,12
2017,1,7.2,0.9,61.4,38.2,16
2017,2,8.9,3.0,38.8,45.0,6
2017,3,12.5,4.0,52.2,121.1,5
2017,4,13.4,4.1,22.6,159.5,2
2017,5,18.1,7.9,25.2,198.6,1
2017,6,20.2,11.8,46.4,151.0,0
2017,7,21.7,12.1,68.2,160.4,0
2017,8,19.6,11.3,63.0,129.8,0
2017,9,17.2,9.2,114.0,96.0,0
2017,10,15.5,8.8,42.2,67.7,1
2017,11,10.3,3.1,51.2,69.4,6
2017,12,7.9,1.0,83.8,55.2,11
2018,1,8.1,1.8,67.4,55.6,9
2018,2,6.2,-0.5,28.2,84.2,17
2018,3,8.1,0.8,85.2,59.9,14
2018,4,13.3,5.4,71.0,106.9,1
2018,5,18.7,6.9,48.4,226.9,0
2018,6,21.5,10.5,32.4,246.5,0
2018,7,25.2,12.2,21.2,254.5,0
2018,8,21.3,12.2,44.4,142.1,0
2018,9,17.6,8.8,88.0,125.4,0
2018,10,14.5,6.1,52.6,106.5,3
2018,11,10.9,4.5,35.8,50.2,4
2018,12,9.6,3.7,71.4,43.2,5
2019,1,6.9,0.7,35.2,36.3,15
2019,2,11.3,1.2,32.8,104.2,11
2019,3,11.6,3.6,53.4,109.6,4
2019,4,13.9,3.9,38.4,134.0,3
2019,5,16.0,6.1,26.8,171.8,1
2019,6,18.2,9.9,115.4,111.7,0
2019,7,21.8,12.5,39.2,160.6,0
2019,8,21.5,12.9,76.4,161.2,0
2019,9,18.5,9.5,120.6,136.3,0
2019,10,13.3,5.8,116.6,76.1,2
2019,11,8.7,3.5,78.8,43.7,5
2019,12,8.4,2.2,69.0,56.0,8"""

@st.cache_data
def load_weather_data(custom_url=""):
    """Loads weather data from URL, local file, or embedded Met Office record."""
    if custom_url and custom_url.strip():
        try:
            return pd.read_csv(custom_url.strip())
        except Exception:
            st.warning("Could not read custom URL; defaulting to Shawbury Met dataset.")
    
    # Check if a local CSV file exists in current directory
    for candidate in ["weather_demo_shawbury_2010_2019.csv", "shawbury_weather.csv", "weather.csv"]:
        if os.path.exists(candidate):
            return pd.read_csv(candidate)
            
    # Fallback to embedded full record
    return pd.read_csv(io.StringIO(EMBEDDED_SHAWBURY_CSV))

# Sidebar controls for data and options
st.sidebar.header("⚙️ Data & Settings")
custom_url_input = st.sidebar.text_input("Optional CSV URL (GitHub Raw):", "")
df = load_weather_data(custom_url_input)

st.sidebar.success(f"Loaded {len(df)} monthly Met records (2010–2019).")
st.sidebar.markdown("---")
st.sidebar.markdown("### 🎓 Degree Pathways")
st.sidebar.info("""
- **Year 1 (4MM013 / 4CS001):** Python, Math & Algorithm logic
- **Year 2 (5CS037 / 5CS045):** Artificial Intelligence & Full Stack Apps
- **Year 3 (6CS012 / 6CS059):** Machine Learning & Intelligent Systems
""")

# ---------------------------------------------------------
# 3. Model Training (scikit-learn Linear Regression)
# ---------------------------------------------------------
# Model 1: tmin -> tmax
X_temp = df[['tmin']]
y_temp = df['tmax']
model_temp = LinearRegression()
model_temp.fit(X_temp, y_temp)

slope = model_temp.coef_[0]
intercept = model_temp.intercept_

# Model 2: [tmin, tmax] -> af (Air Frost days)
X_frost = df[['tmin', 'tmax']]
y_frost = df['af']
model_frost = LinearRegression()
model_frost.fit(X_frost, y_frost)

# ---------------------------------------------------------
# 4. Interactive Prediction Section
# ---------------------------------------------------------
col_left, col_right = st.columns([1, 1], gap="large")

with col_left:
    st.subheader("🎯 Step 1: Predict Maximum Temperature (`tmax`)")
    st.write("Use the slider below to set a **minimum morning temperature (`tmin`)**. The AI uses its learned formula to predict the **afternoon high**.")
    
    user_tmin = st.slider(
        "Morning Low Temperature (tmin °C):",
        min_value=-6.0,
        max_value=16.0,
        value=3.5,
        step=0.5,
        help="Slide to pick a hypothetical morning minimum temperature."
    )
    
    pred_tmax = float(model_temp.predict([[user_tmin]])[0])
    
    st.metric(
        label="Predicted Afternoon Maximum (tmax)",
        value=f"{pred_tmax:.1f} °C",
        delta=f"{(pred_tmax - user_tmin):.1f} °C daily rise"
    )
    
    # Campus condition badge
    if pred_tmax >= 20.0:
        st.success("☀️ **Campus Forecast:** Warm and sunny! Ideal for outdoor campus events.")
    elif pred_tmax >= 13.0:
        st.info("⛅ **Campus Forecast:** Mild and pleasant. Standard sweater weather.")
    elif pred_tmax >= 7.0:
        st.warning("🧥 **Campus Forecast:** Cool. Grab a jacket before heading to lectures.")
    else:
        st.error("🌧️ **Campus Forecast:** Chilly day! Coat and scarf recommended.")

with col_right:
    st.subheader("❄️ Step 2: Bonus Frost Days Predictor (`af`)")
    st.write("The secondary AI model uses **both minimum and maximum temperatures** to estimate the number of **Air Frost days** in that month.")
    
    user_custom_tmax = st.slider(
        "Confirm or override afternoon high (tmax °C):",
        min_value=0.0,
        max_value=26.0,
        value=float(np.clip(round(pred_tmax, 1), 0.0, 26.0)),
        step=0.5
    )
    
    pred_af = max(0, int(round(float(model_frost.predict([[user_tmin, user_custom_tmax]])[0]))))
    
    st.metric(
        label="Predicted Days of Air Frost (af)",
        value=f"{pred_af} Days"
    )
    
    if pred_af >= 15:
        st.error("🧊 **Heavy Frost Risk:** High number of sub-zero mornings expected.")
    elif pred_af >= 5:
        st.warning("❄️ **Moderate Frost:** Occasional frosty mornings; check travel schedules.")
    else:
        st.success("🌱 **Mild Conditions:** Frost is rare or absent under these temperatures.")

# ---------------------------------------------------------
# 5. Visualizing "How the AI Learned" (Interactive Chart)
# ---------------------------------------------------------
st.divider()
st.subheader("📈 How the Machine Learned the Relationship")
st.write(
    f"In pure math, the AI discovered the best-fit line: "
    f"**tmax = ({slope:.2f} × tmin) + {intercept:.2f}** "
    f"using 120 historical months."
)

# Generate scatter and trendline points for visualization
tmin_range = np.linspace(df['tmin'].min() - 1, df['tmin'].max() + 1, 60)
tmax_line = model_temp.predict(tmin_range.reshape(-1, 1))

chart_df = pd.DataFrame({
    'tmin_Historical': df['tmin'],
    'tmax_Historical': df['tmax']
})

line_df = pd.DataFrame({
    'tmin_Line': tmin_range,
    'tmax_AI_Learned': tmax_line
})

# Display chart using Streamlit native line/scatter
st.line_chart(
    data=line_df.set_index('tmin_Line'),
    color="#2563EB",
    use_container_width=True
)

st.caption("The blue line shows the AI's learned prediction function across all temperatures.")

# ---------------------------------------------------------
# 6. Raw Data Explorer
# ---------------------------------------------------------
with st.expander("📊 Explore the Full Shawbury Met Office Dataset (120 Records)"):
    st.write("This table shows the 10-year monthly meteorological dataset used to train the models:")
    st.dataframe(df, use_container_width=True)
    
    col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
    col_stat1.metric("Average Minimum Temp", f"{df['tmin'].mean():.1f} °C")
    col_stat2.metric("Average Maximum Temp", f"{df['tmax'].mean():.1f} °C")
    col_stat3.metric("Highest Recorded Temp", f"{df['tmax'].max():.1f} °C")
    col_stat4.metric("Lowest Recorded Temp", f"{df['tmin'].min():.1f} °C")

# ---------------------------------------------------------
# 7. Session Footer
# ---------------------------------------------------------
st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; color: #6B7280; font-size: 0.9rem;">
        University of Wolverhampton — School of Engineering, Computing and Mathematical Sciences<br>
        BSc (Hons) Computer Science | BSc (Hons) Cybersecurity | BSc (Hons) Computing & IT
    </div>
    """,
    unsafe_allow_html=True
)
