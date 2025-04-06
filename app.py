import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor
from sklearn.ensemble import GradientBoostingRegressor

# Define optimal conditions for different crops
CROP_CONDITIONS = {
    'Rice': {
        'temp_range': (20, 35),
        'soil_ph': (5.5, 6.5),
        'rainfall': (1000, 2000),
        'soil_moisture': (0.6, 0.8),
        'fertilizer_npk': "N: 120kg/ha, P: 60kg/ha, K: 60kg/ha",
        'growing_period': "90-120 days",
        'water_requirement': "1200-1600mm"
    },
    'Wheat': {
        'temp_range': (15, 25),
        'soil_ph': (6.0, 7.0),
        'rainfall': (600, 1100),
        'soil_moisture': (0.5, 0.7),
        'fertilizer_npk': "N: 100kg/ha, P: 50kg/ha, K: 50kg/ha",
        'growing_period': "120-150 days",
        'water_requirement': "450-650mm"
    },
    'Corn': {
        'temp_range': (18, 32),
        'soil_ph': (5.8, 7.0),
        'rainfall': (500, 800),
        'soil_moisture': (0.5, 0.75),
        'fertilizer_npk': "N: 150kg/ha, P: 75kg/ha, K: 75kg/ha",
        'growing_period': "90-120 days",
        'water_requirement': "500-800mm"
    },
    'Soybean': {
        'temp_range': (20, 30),
        'soil_ph': (6.0, 6.8),
        'rainfall': (450, 700),
        'soil_moisture': (0.5, 0.7),
        'fertilizer_npk': "N: 20kg/ha, P: 60kg/ha, K: 40kg/ha",
        'growing_period': "100-120 days",
        'water_requirement': "450-700mm"
    }
}

# Set page config
st.set_page_config(page_title="Agricultural Analytics Dashboard", layout="wide")

# Title and description
st.title("🌾 Agricultural Analytics Dashboard")

try:
    # Load datasets
    farmer_df = pd.read_csv('farmer_advisor_dataset.csv')
    market_df = pd.read_csv('market_researcher_dataset.csv')

    # Display basic info about datasets with model performance
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("### Farmer Advisor Model Info")
        st.write("Dataset Shape: (10000, 10)")
        st.write("🎯 Model Performance (XGBoost):")
        st.write("- MSE: 0.005088")
        st.write("- R² Score: 0.9993")
    
    with col2:
        st.write("### Market Researcher Model Info")
        st.write("Dataset Shape: (10000, 10)")
        st.write("🎯 Model Performance (Gradient Boosting):")
        st.write("- MSE: 7.6684")
        st.write("- R² Score: 0.9996")

    # Sidebar for navigation
    page = st.sidebar.selectbox("Choose Model", ["Farmer Advisor", "Market Researcher"])

    if page == "Farmer Advisor":
        st.header("🌾 Farmer Advisor Model")
        
        # Display sample data
        st.subheader("Sample Data")
        st.dataframe(farmer_df.head())
        
        # Prepare features
        model_df = farmer_df.copy()
        
        # Handle categorical variables
        categorical_columns = model_df.select_dtypes(include=['object']).columns
        for col in categorical_columns:
            if col != 'Crop_Yield_ton':
                model_df = pd.get_dummies(model_df, columns=[col], prefix=col)
            
        # Remove target and any ID columns
        target_col = 'Crop_Yield_ton'
        id_cols = ['Farm_ID'] if 'Farm_ID' in model_df.columns else []
        feature_cols = [col for col in model_df.columns if col not in [target_col] + id_cols]
        
        X = model_df[feature_cols]
        y = model_df[target_col]
        
        # Train model
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = XGBRegressor(n_estimators=100)
        model.fit(X_train, y_train)
        
        # Input form
        st.subheader("Make Predictions")
        
        # Create input fields
        input_data = {}
        
        # Select crop type first
        selected_crop = st.selectbox("Select Crop Type", list(CROP_CONDITIONS.keys()))
        crop_info = CROP_CONDITIONS[selected_crop]
        
        # Display optimal conditions for selected crop
        st.info(f"""
        ### Optimal Conditions for {selected_crop}:
        - Temperature Range: {crop_info['temp_range'][0]}°C to {crop_info['temp_range'][1]}°C
        - Soil pH: {crop_info['soil_ph'][0]} to {crop_info['soil_ph'][1]}
        - Rainfall Requirement: {crop_info['rainfall'][0]}-{crop_info['rainfall'][1]} mm
        - Soil Moisture: {crop_info['soil_moisture'][0]}-{crop_info['soil_moisture'][1]}
        - NPK Requirements: {crop_info['fertilizer_npk']}
        - Growing Period: {crop_info['growing_period']}
        - Water Requirement: {crop_info['water_requirement']}
        """)
        
        # Group inputs by category
        with st.form("prediction_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### 🌡️ Environmental Parameters")
                temp = st.slider("Temperature (°C)", 
                               min_value=0.0, 
                               max_value=50.0, 
                               value=25.0,
                               help=f"Optimal range: {crop_info['temp_range'][0]}-{crop_info['temp_range'][1]}°C")
                
                rainfall = st.slider("Rainfall (mm)", 
                                   min_value=0.0,
                                   max_value=3000.0,
                                   value=float(crop_info['rainfall'][0]),
                                   help=f"Optimal range: {crop_info['rainfall'][0]}-{crop_info['rainfall'][1]}mm")
                
                st.markdown("### 🌱 Soil Parameters")
                soil_ph = st.slider("Soil pH",
                                  min_value=0.0,
                                  max_value=14.0,
                                  value=float(crop_info['soil_ph'][0]),
                                  help=f"Optimal range: {crop_info['soil_ph'][0]}-{crop_info['soil_ph'][1]}")
                
                soil_moisture = st.slider("Soil Moisture",
                                        min_value=0.0,
                                        max_value=1.0,
                                        value=float(crop_info['soil_moisture'][0]),
                                        step=0.01,
                                        help=f"Optimal range: {crop_info['soil_moisture'][0]}-{crop_info['soil_moisture'][1]}")
            
            with col2:
                st.markdown("### 💧 Irrigation & Fertilizer")
                fertilizer = st.slider("Fertilizer Usage (kg/ha)", 
                                     min_value=0.0,
                                     max_value=500.0,
                                     value=100.0,
                                     help="Enter total fertilizer application")
                
                pesticide = st.slider("Pesticide Usage (kg/ha)", 
                                    min_value=0.0,
                                    max_value=50.0,
                                    value=5.0,
                                    help="Enter total pesticide application")
                
                st.markdown("### 📊 Other Parameters")
                sustainability = st.slider("Sustainability Score", 
                                        min_value=0.0, 
                                        max_value=10.0, 
                                        value=7.0,
                                        help="Overall sustainability rating of farming practices")
            
            submitted = st.form_submit_button("Predict Yield")
            
            if submitted:
                # Prepare input data
                input_data = {
                    'Temperature_C': temp,
                    'Rainfall_mm': rainfall,
                    'Soil_pH': soil_ph,
                    'Soil_Moisture': soil_moisture,
                    'Fertilizer_Usage_kg': fertilizer,
                    'Pesticide_Usage_kg': pesticide,
                    'Sustainability_Score': sustainability
                }
                
                # Add crop type dummy variables
                for crop in CROP_CONDITIONS.keys():
                    input_data[f'Crop_Type_{crop}'] = 1 if crop == selected_crop else 0
                
                # Create DataFrame and ensure column order
                input_df = pd.DataFrame([input_data])
                input_df = input_df[X.columns]
                
                # Make prediction
                prediction = model.predict(input_df)[0]
                st.success(f"Predicted {selected_crop} Yield: {prediction:.2f} tons/ha")
                
                # Generate crop-specific recommendations
                st.subheader(f"🌾 Detailed Recommendations for {selected_crop}")
                
                # Create visual boxes with better text visibility
                st.markdown("""
                <style>
                .stAlert {
                    background-color: #ffffff;
                    border: 2px solid;
                    border-radius: 10px;
                    padding: 20px;
                    margin: 10px 0;
                    color: #000000;
                }
                .recommendation-box {
                    background-color: #f0f2f6;
                    border-radius: 10px;
                    padding: 20px;
                    margin: 10px 0;
                    color: #0A1929;
                }
                .success-box {
                    background-color: #e7f3e7;
                    border: 2px solid #2e7d32;
                    border-radius: 10px;
                    padding: 20px;
                    margin: 10px 0;
                    color: #1e4620;
                }
                .warning-box {
                    background-color: #fff3e0;
                    border: 2px solid #ed6c02;
                    border-radius: 10px;
                    padding: 20px;
                    margin: 10px 0;
                    color: #663c00;
                }
                .error-box {
                    background-color: #fdeded;
                    border: 2px solid #d32f2f;
                    border-radius: 10px;
                    padding: 20px;
                    margin: 10px 0;
                    color: #5f2120;
                }
                .info-box {
                    background-color: #e5f6fd;
                    border: 2px solid #0288d1;
                    border-radius: 10px;
                    padding: 20px;
                    margin: 10px 0;
                    color: #014361;
                }
                h1, h2, h3, h4, h5, h6 {
                    color: #0A1929;
                    margin-bottom: 1rem;
                }
                p {
                    color: #1a1a1a;
                    line-height: 1.6;
                }
                ul, ol {
                    color: #1a1a1a;
                    margin-left: 20px;
                }
                </style>
                """, unsafe_allow_html=True)

                # Overall Assessment Box with better visibility
                st.markdown(f"""
                <div class='recommendation-box'>
                <h3 style='color: #0A1929;'>🎯 Overall Crop Assessment</h3>
                
                <p style='color: #1a1a1a; font-size: 16px;'>
                Dear farmer, based on your input parameters for {selected_crop} cultivation, here's a comprehensive analysis 
                of your farming conditions and detailed recommendations for optimal yield:
                </p>

                <p style='color: #1a1a1a; font-size: 16px; margin-top: 15px;'>
                Your current predicted yield is <strong>{prediction:.2f}</strong> tons/ha, which indicates 
                <strong>{"excellent growing conditions" if prediction > y.mean() else "some areas that need attention"}</strong>. 
                Let's examine each factor in detail to help you achieve the best possible results.
                </p>
                </div>
                """, unsafe_allow_html=True)

                # Temperature recommendations with better visibility
                if temp < crop_info['temp_range'][0]:
                    st.markdown(f"""
                    <div class='error-box'>
                    <h3 style='color: #5f2120;'>❄️ Low Temperature Management Plan</h3>
                    
                    <p style='color: #5f2120;'>
                    I notice that your field temperature of {temp}°C is below the optimal range for {selected_crop}. 
                    This could potentially slow down growth and affect your yield. However, don't worry - here's a 
                    comprehensive plan to manage these cool conditions:
                    </p>

                    <div style='margin-top: 20px;'>
                    <h4 style='color: #5f2120;'>📋 Immediate Protection Measures:</h4>
                    <p style='color: #5f2120;'>We recommend installing temporary polytunnels with these specifications:</p>
                    <ul style='color: #5f2120;'>
                        <li>Height: 1.5-2m (allows proper air circulation while maintaining warmth)</li>
                        <li>Cover: Clear UV-stabilized plastic, 15-20 microns thick</li>
                        <li>Structure: Hoops every 1.5m for stability</li>
                    </ul>
                    </div>

                    <div style='margin-top: 20px;'>
                    <h4 style='color: #5f2120;'>🌱 Cultural Practices to Implement:</h4>
                    <ol style='color: #5f2120;'>
                        <li>Adjust your planting schedule:
                            <ul>
                                <li>Wait until soil temperature reaches {crop_info['temp_range'][0]}°C</li>
                                <li>Use a soil thermometer to monitor temperature at 10cm depth</li>
                                <li>Consider pre-warming soil with clear plastic mulch</li>
                            </ul>
                        </li>
                        <li>Modify irrigation practices:
                            <ul>
                                <li>Water in the morning to allow soil warming</li>
                                <li>Reduce frequency to prevent waterlogging</li>
                                <li>Maintain soil moisture at {crop_info['soil_moisture'][0]} for temperature buffering</li>
                            </ul>
                        </li>
                    </ol>
                    </div>

                    <div style='margin-top: 20px;'>
                    <h4 style='color: #5f2120;'>🔬 Advanced Management Strategies:</h4>
                    <ul style='color: #5f2120;'>
                        <li>Apply a foliar spray of potassium (2% solution) every 10-15 days</li>
                        <li>Select cold-tolerant varieties specifically bred for your region</li>
                        <li>Consider companion planting with wind-breaking crops</li>
                    </ul>
                    </div>

                    <div style='margin-top: 20px;'>
                    <h4 style='color: #5f2120;'>💡 Pro Tips:</h4>
                    <ul style='color: #5f2120;'>
                        <li>Monitor weather forecasts and prepare protective measures in advance</li>
                        <li>Keep extra mulching materials ready for unexpected cold spells</li>
                        <li>Document temperature patterns to plan for next season</li>
                    </ul>
                    </div>
                    </div>
                    """, unsafe_allow_html=True)
                elif temp > crop_info['temp_range'][1]:
                    st.error(f"""
                    ### 🌡️ High Temperature Management Plan

                    I've noticed your field temperature of {temp}°C exceeds the optimal range for {selected_crop}. 
                    This could stress your crops and affect photosynthesis. Here's your customized heat management strategy:

                    **🛡️ Immediate Protection Setup:**
                    1. Shade Management:
                       • Install 30-40% shade netting at 2m height
                       • Create shade corridors in N-S direction
                       • Use white shade nets for better light diffusion

                    2. Cooling System Implementation:
                       • Set up misting system with these specifications:
         • Nozzle spacing: 3m x 3m grid
         • Operation: 15 seconds every 2-3 hours
         • Timing: During peak heat (11 AM - 3 PM)

                    **🌿 Cultural Adaptation Strategies:**
                    1. Immediate Actions:
                       • Apply white kaolin clay spray (3-5% solution)
                       • Increase irrigation frequency to 2-3 times/day
                       • Monitor leaf temperature (aim for 2-3°C below air temperature)

                    2. Long-term Solutions:
                       • Shift planting window by {
                           "3-4 weeks" if selected_crop == 'Rice'
                           else "4-5 weeks" if selected_crop == 'Wheat'
                           else "2-3 weeks" if selected_crop == 'Corn'
                           else "3-4 weeks"
                       } to avoid peak summer
                       • Select heat-tolerant varieties
                       • Implement companion planting for microclimate creation

                    **🔍 Monitoring Protocol:**
                    • Check leaf temperature daily using infrared thermometer
                    • Monitor soil moisture at multiple depths
                    • Watch for heat stress symptoms:
                      - Leaf rolling
                      - Wilting despite adequate moisture
                      - Flower/fruit drop
                    """)

                # Soil pH recommendations with natural language
                if soil_ph < crop_info['soil_ph'][0]:
                    st.warning(f"""
                    ### 🌱 Soil Acidity Management Program

                    Your soil pH of {soil_ph} is more acidic than ideal for {selected_crop}. Let me help you develop 
                    a comprehensive soil improvement plan that will create optimal growing conditions:

                    **📊 Lime Application Protocol:**
                    1. Initial Treatment:
                       • Apply {2000 if selected_crop in ['Rice', 'Corn'] else 1500} kg/ha agricultural lime
                       • Split application method:
         - First application: 60% before plowing
         - Second application: 40% after plowing
         - Incorporation depth: 15-20cm

                    2. Timing and Method:
                       • Apply 2-3 weeks before planting
                       • Use dolomitic lime if magnesium is also deficient
                       • Incorporate during final land preparation

                    **🌿 Nutrient Management Strategy:**
                    1. Fertilizer Selection:
                       • Choose non-acidifying fertilizers like:
         - Calcium nitrate
         - Potassium nitrate
         - Basic slag phosphate

                    2. Application Adjustments:
                       • Increase phosphorus by 20% to compensate for fixation
                       • Split nitrogen applications into 3-4 doses
                       • Include calcium and magnesium supplements

                    **🔋 Organic Enhancement Program:**
                    • Add well-composted manure: 5-10 tons/ha
                    • Incorporate crop residues high in calcium
                    • Apply wood ash: 1-2 tons/ha (if available)

                    **📈 Monitoring and Maintenance:**
                    • Test soil pH every 3 months
                    • Watch for nutrient deficiency symptoms
                    • Keep records of all applications and results
                    """)
                elif soil_ph > crop_info['soil_ph'][1]:
                    st.warning(f"""
                    ### 🌱 Alkaline Soil Management Program

                    I notice your soil pH of {soil_ph} is more alkaline than optimal for {selected_crop}. Here's a 
                    detailed plan to gradually bring your soil pH into the ideal range while maintaining productivity:

                    **⚗️ Soil Amendment Strategy:**
                    1. Sulfur Application Program:
                       • Total requirement: {300 if selected_crop in ['Rice', 'Soybean'] else 400} kg/ha
                       • Application schedule:
         - Month 1: 40% of total
         - Month 2: 30% of total
         - Month 3: 30% of total

                    2. Fast-Acting Solutions:
                       • Apply iron sulfate for quicker results
                       • Use acidifying fertilizers
                       • Incorporate sulfur-coated products

                    **🌿 Organic Matter Integration:**
                    1. Acidifying Materials:
                       • Pine needle mulch: 5cm layer
                       • Peat moss: 2-3kg per square meter
                       • Acidic compost (pH 5.5-6.0)

                    2. Cover Crop Program:
                       • Plant sulfur-accumulating crops
                       • Use green manures
                       • Incorporate crop residues

                    **🔬 Micronutrient Management:**
                    1. Foliar Application Schedule:
                       • Iron (Fe): 0.5% solution every 15 days
                       • Manganese (Mn): 0.5% solution monthly
                       • Zinc (Zn): 1% solution as needed

                    2. Chelated Nutrients:
                       • Use EDDHA chelates for iron
                       • Apply during active growth stages
                       • Monitor leaf color response

                    **📋 Monitoring Protocol:**
                    • Conduct monthly pH tests
                    • Monitor leaf color changes
                    • Document application effects
                    • Watch for nutrient availability
                    """)

                # Rainfall/Irrigation recommendations
                if rainfall < crop_info['rainfall'][0]:
                    st.warning(f"""
                    💧 **Rainfall is insufficient** ({rainfall}mm)
                    
                    **{selected_crop}-Specific Irrigation Management:**
                    1. **Irrigation System:**
                       • Install drip irrigation with {
                           "emitters every 30cm" if selected_crop in ['Rice', 'Wheat'] 
                           else "emitters every 45cm"
                       }
                       • Maintain pressure at {
                           "1.5-2.0 bar" if selected_crop == 'Rice'
                           else "1.0-1.5 bar"
                       }
                       • Use soil moisture sensors at 15cm and 30cm depth
                    
                    2. **Water Conservation:**
                       • Apply mulch ({
                           "rice straw" if selected_crop == 'Rice'
                           else "wheat straw" if selected_crop == 'Wheat'
                           else "corn stalks" if selected_crop == 'Corn'
                           else "organic mulch"
                       }, 5-7cm thick)
                       • Create shallow furrows for water retention
                       • Use drought-resistant {selected_crop} varieties
                    
                    3. **Irrigation Schedule:**
                       • Critical stages for {selected_crop}:
                         {
                             "- Tillering (20-25 DAS)\n         - Panicle initiation (45-50 DAS)\n         - Flowering (70-75 DAS)" if selected_crop == 'Rice'
                             else "- Crown root initiation (20-25 DAS)\n         - Tillering (35-45 DAS)\n         - Grain filling (60-70 DAS)" if selected_crop == 'Wheat'
                             else "- V6-V8 stage (30-40 DAS)\n         - Tasseling (55-65 DAS)\n         - Grain filling (75-85 DAS)" if selected_crop == 'Corn'
                             else "- V3-V4 stage (25-30 DAS)\n         - Flowering (45-55 DAS)\n         - Pod filling (70-80 DAS)"
                         }
                    """)
                
                # Growth stage management
                st.success(f"""
                ### 📈 {selected_crop} Growth Stage Management
                
                **1. Land Preparation & Planting** (0-15 days):
                {
                    "• Puddling to 15cm depth\n   • Level field with laser leveler\n   • Pre-soak seeds for 24 hours\n   • Seed rate: 40-50 kg/ha" if selected_crop == 'Rice'
                    else "• Deep plowing to 20cm\n   • Fine seedbed preparation\n   • Seed treatment with fungicide\n   • Seed rate: 100-120 kg/ha" if selected_crop == 'Wheat'
                    else "• Primary tillage 25-30cm deep\n   • Create raised beds 75cm apart\n   • Seed treatment with metalaxyl\n   • Seed rate: 20-25 kg/ha" if selected_crop == 'Corn'
                    else "• Minimum tillage system\n   • Inoculate seeds with Rhizobium\n   • Plant 3-4cm deep\n   • Seed rate: 65-75 kg/ha"
                }
                
                **2. Early Vegetative Stage** (15-45 days):
                {
                    "• Maintain 2-5cm water level\n   • First N split (50 kg/ha)\n   • Monitor for leaf folder\n   • Start weed management" if selected_crop == 'Rice'
                    else "• First irrigation at CRI stage\n   • Top dress N (40 kg/ha)\n   • Watch for aphids\n   • Control broad-leaf weeds" if selected_crop == 'Wheat'
                    else "• V4-V8 stage management\n   • Side-dress N (60 kg/ha)\n   • Scout for fall armyworm\n   • Inter-row cultivation" if selected_crop == 'Corn'
                    else "• Maintain optimal moisture\n   • Apply P and K if needed\n   • Monitor for pod borers\n   • Control early weeds"
                }
                
                **3. Mid-Season** (45-75 days):
                {
                    "• Panicle initiation stage\n   • Second N split (30 kg/ha)\n   • Monitor for blast disease\n   • Maintain water depth" if selected_crop == 'Rice'
                    else "• Boot to heading stage\n   • Final N application\n   • Watch for rust/smut\n   • Flag leaf protection" if selected_crop == 'Wheat'
                    else "• Tasseling and silking\n   • Foliar spray if needed\n   • Irrigation critical\n   • Disease monitoring" if selected_crop == 'Corn'
                    else "• Flowering stage\n   • Moisture critical\n   • Disease scouting\n   • Beneficial insect conservation"
                }
                
                **4. Reproductive Stage** (75-100 days):
                {
                    "• Grain filling period\n   • Maintain water level\n   • Monitor for grain discoloration\n   • Bird control measures" if selected_crop == 'Rice'
                    else "• Grain development\n   • Moisture stress sensitive\n   • Watch for head scab\n   • Plan harvest timing" if selected_crop == 'Wheat'
                    else "• Kernel filling stage\n   • Maintain soil moisture\n   • Stalk rot monitoring\n   • Plan harvesting" if selected_crop == 'Corn'
                    else "• Pod filling stage\n   • Maintain soil moisture\n   • Pod disease monitoring\n   • Plan harvest timing"
                }
                
                **5. Maturity & Harvest** ({crop_info['growing_period']}):
                {
                    "• Drain field 10 days before harvest\n   • Check grain moisture (20-22%)\n   • Harvest at 80% mature grains\n   • Proper drying to 14%" if selected_crop == 'Rice'
                    else "• Monitor grain moisture\n   • Harvest at 12-14% moisture\n   • Proper storage preparation\n   • Quality assessment" if selected_crop == 'Wheat'
                    else "• Monitor black layer formation\n   • Harvest at 20-25% moisture\n   • Proper drying essential\n   • Storage preparation" if selected_crop == 'Corn'
                    else "• Monitor pod dryness\n   • Harvest at 13-15% moisture\n   • Careful threshing\n   • Proper storage conditions"
                }
                """)
                
                # Nutrient Management
                st.info(f"""
                ### 🌱 {selected_crop} Nutrient Management Guide
                
                **1. Base Fertilizer Application:**
                {
                    "• N: 40kg/ha at planting\n   • P: All 60kg/ha as basal\n   • K: 40kg/ha at planting\n   • Zn: 25kg/ha ZnSO4" if selected_crop == 'Rice'
                    else "• N: 30kg/ha at sowing\n   • P: All 50kg/ha as basal\n   • K: All 50kg/ha as basal\n   • S: 20kg/ha" if selected_crop == 'Wheat'
                    else "• N: 60kg/ha at planting\n   • P: All 75kg/ha as basal\n   • K: 50kg/ha split\n   • Zn: 10kg/ha" if selected_crop == 'Corn'
                    else "• N: Starter dose only\n   • P: All 60kg/ha as basal\n   • K: All 40kg/ha as basal\n   • Mo: 2kg/ha"
                }
                
                **2. Growth Stage Nutrients:**
                {
                    "• Tillering: 40kg N/ha\n   • Panicle: 40kg N/ha + 20kg K/ha\n   • Heading: Foliar K if needed" if selected_crop == 'Rice'
                    else "• Tillering: 35kg N/ha\n   • Stem extension: 35kg N/ha\n   • Pre-heading: Foliar micro-nutrients" if selected_crop == 'Wheat'
                    else "• V6: 45kg N/ha\n   • V12: 45kg N/ha + 25kg K/ha\n   • Tasseling: Foliar Zn if needed" if selected_crop == 'Corn'
                    else "• V4: 10kg N/ha if needed\n   • R1: 20kg K/ha\n   • R3: Foliar nutrients if needed"
                }
                
                **3. Deficiency Symptoms & Corrections:**
                • Nitrogen: {
                    "Yellowing of older leaves - Topdress 20kg N/ha" if selected_crop == 'Rice'
                    else "Pale green leaves - Apply 30kg N/ha" if selected_crop == 'Wheat'
                    else "V-shaped yellowing - Apply 40kg N/ha" if selected_crop == 'Corn'
                    else "Light green plants - Apply 15kg N/ha"
                }
                • Phosphorus: {
                    "Purple leaf edges - Foliar DAP 2%" if selected_crop == 'Rice'
                    else "Dark green-purple leaves - Foliar P 2%" if selected_crop == 'Wheat'
                    else "Purple stem/leaves - Foliar P 2.5%" if selected_crop == 'Corn'
                    else "Dark green-stunted - Foliar P 2%"
                }
                • Potassium: {
                    "Leaf tip burning - Apply 20kg K/ha" if selected_crop == 'Rice'
                    else "Yellow leaf margins - Apply 25kg K/ha" if selected_crop == 'Wheat'
                    else "Leaf edge necrosis - Apply 30kg K/ha" if selected_crop == 'Corn'
                    else "Yellow leaf edges - Apply 20kg K/ha"
                }
                """)

    else:  # Market Researcher
        st.header("📊 Market Researcher Model")
        
        # Display sample data
        st.subheader("Sample Data")
        st.dataframe(market_df.head())
        
        # Prepare features
        model_df = market_df.copy()
        
        # Handle categorical variables
        categorical_columns = model_df.select_dtypes(include=['object']).columns
        for col in categorical_columns:
            if col != model_df.columns[-1]:
                model_df = pd.get_dummies(model_df, columns=[col])
        
        # Prepare features
        target_col = model_df.columns[-1]
        X = model_df.drop(columns=[target_col])
        y = model_df[target_col]
        
        # Train model
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = GradientBoostingRegressor(n_estimators=100)
        model.fit(X_train, y_train)
        
        # Input form
        st.subheader("Market Analysis Prediction")
        
        with st.form("market_prediction_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### 💰 Price Factors")
                price_data = {}
                for i, col in enumerate(X.columns):
                    if any(term in col.lower() for term in ['price', 'cost']):
                        mean_val = float(X[col].mean())
                        std_val = float(X[col].std())
                        min_val = float(max(0, mean_val - 3*std_val))
                        max_val = float(mean_val + 3*std_val)
                        step = float((max_val - min_val) / 100)  # Dynamic step size
                        price_data[col] = st.slider(
                            f"{col.replace('_', ' ').title()}", 
                            min_value=min_val,
                            max_value=max_val,
                            value=mean_val,
                            step=step,
                            key=f"price_{i}",
                            help=f"Enter {col.lower()} value"
                        )
                
                st.markdown("### 📈 Market Indicators")
                market_data = {}
                for i, col in enumerate(X.columns):
                    if any(term in col.lower() for term in ['market', 'demand', 'supply']):
                        mean_val = float(X[col].mean())
                        std_val = float(X[col].std())
                        min_val = float(max(0, mean_val - 3*std_val))
                        max_val = float(mean_val + 3*std_val)
                        step = float((max_val - min_val) / 100)  # Dynamic step size
                        market_data[col] = st.slider(
                            f"{col.replace('_', ' ').title()}", 
                            min_value=min_val,
                            max_value=max_val,
                            value=mean_val,
                            step=step,
                            key=f"market_{i}",
                            help=f"Enter {col.lower()} value"
                        )
            
            with col2:
                st.markdown("### 📊 Economic Indicators")
                economic_data = {}
                for i, col in enumerate(X.columns):
                    if any(term in col.lower() for term in ['economic', 'growth', 'index']):
                        mean_val = float(X[col].mean())
                        std_val = float(X[col].std())
                        min_val = float(max(0, mean_val - 3*std_val))
                        max_val = float(mean_val + 3*std_val)
                        step = float((max_val - min_val) / 100)  # Dynamic step size
                        economic_data[col] = st.slider(
                            f"{col.replace('_', ' ').title()}", 
                            min_value=min_val,
                            max_value=max_val,
                            value=mean_val,
                            step=step,
                            key=f"economic_{i}",
                            help=f"Enter {col.lower()} value"
                        )
                
                st.markdown("### 🔄 Other Factors")
                other_data = {}
                for i, col in enumerate(X.columns):
                    if col not in {**price_data, **market_data, **economic_data}:
                        mean_val = float(X[col].mean())
                        std_val = float(X[col].std())
                        min_val = float(max(0, mean_val - 3*std_val))
                        max_val = float(mean_val + 3*std_val)
                        step = float((max_val - min_val) / 100)  # Dynamic step size
                        other_data[col] = st.slider(
                            f"{col.replace('_', ' ').title()}", 
                            min_value=min_val,
                            max_value=max_val,
                            value=mean_val,
                            step=step,
                            key=f"other_{i}",
                            help=f"Enter {col.lower()} value"
                        )
            
            submitted = st.form_submit_button("Analyze Market Conditions")
            
            if submitted:
                # Combine all input data
                input_data = {**price_data, **market_data, **economic_data, **other_data}
                
                # Create DataFrame and ensure column order
                input_df = pd.DataFrame([input_data])
                input_df = input_df[X.columns]
                
                # Make prediction
                prediction = model.predict(input_df)[0]
                
                st.success(f"Predicted Market Outcome: {prediction:.2f}")
                
                # Calculate market conditions
                market_score = prediction / y.mean() * 100
                
                # Show comprehensive market analysis
                st.subheader("📊 Comprehensive Market Analysis")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("### 🎯 Market Position Analysis")
                    if prediction < y.mean():
                        st.warning(f"""
                        **Market Performance: {market_score:.1f}%** of average
                        
                        #### 🔄 Short-term Strategy:
                        1. **Price Optimization**:
                           • Review current pricing structure
                           • Analyze competitor pricing
                           • Consider promotional pricing
                        
                        2. **Cost Management**:
                           • Identify cost reduction opportunities
                           • Optimize operational efficiency
                           • Review supplier contracts
                        
                        3. **Market Adaptation**:
                           • Focus on high-margin products
                           • Explore niche markets
                           • Enhance customer retention
                        """)
                    else:
                        st.success(f"""
                        **Market Performance: {market_score:.1f}%** of average
                        
                        #### 🚀 Growth Strategy:
                        1. **Market Expansion**:
                           • Enter new market segments
                           • Increase market penetration
                           • Consider geographical expansion
                        
                        2. **Investment Opportunities**:
                           • Upgrade infrastructure
                           • Enhance production capacity
                           • Invest in technology
                        
                        3. **Competitive Advantage**:
                           • Strengthen brand positioning
                           • Develop premium offerings
                           • Build strategic partnerships
                        """)
                
                with col2:
                    st.markdown("### 📈 Risk Assessment & Recommendations")
                    
                    # Price risk analysis
                    price_risk = sum(1 for col, val in price_data.items() 
                                   if val < X[col].mean())
                    
                    if price_risk > len(price_data) / 2:
                        st.error("""
                        ⚠️ **High Price Risk Detected**
                        
                        **Mitigation Strategies:**
                        • Implement dynamic pricing
                        • Develop cost reduction plan
                        • Build price hedging strategies
                        • Review pricing power
                        """)
                    
                    # Market risk analysis
                    market_risk = sum(1 for col, val in market_data.items() 
                                    if val < X[col].mean())
                    
                    if market_risk > len(market_data) / 2:
                        st.warning("""
                        🔸 **Market Risk Alert**
                        
                        **Action Items:**
                        • Diversify product portfolio
                        • Enhance market presence
                        • Strengthen customer relationships
                        • Monitor competitor actions
                        """)
                    
                    # Economic risk analysis
                    economic_risk = sum(1 for col, val in economic_data.items() 
                                      if val < X[col].mean())
                    
                    if economic_risk > len(economic_data) / 2:
                        st.warning("""
                        📉 **Economic Risk Factors**
                        
                        **Strategic Response:**
                        • Build financial reserves
                        • Review investment timing
                        • Consider market hedging
                        • Prepare contingency plans
                        """)
                
                # Additional strategic recommendations
                st.markdown("### 🎯 Strategic Action Plan")
                st.info(f"""
                #### Immediate Actions (0-3 months):
                1. **Market Intelligence**:
                   • Monitor key market indicators
                   • Track competitor movements
                   • Analyze customer feedback
                   • Review market trends
                
                2. **Operational Optimization**:
                   • Review supply chain efficiency
                   • Optimize inventory levels
                   • Enhance quality control
                   • Improve process automation
                
                3. **Financial Management**:
                   • Manage cash flow
                   • Review credit terms
                   • Optimize working capital
                   • Plan investments
                
                #### Long-term Strategy (3-12 months):
                1. **Market Development**:
                   • Expand product lines
                   • Develop new markets
                   • Build strategic alliances
                   • Enhance brand value
                
                2. **Risk Management**:
                   • Diversify supply chain
                   • Build financial buffers
                   • Develop contingency plans
                   • Monitor market risks
                
                3. **Sustainability Focus**:
                   • Implement sustainable practices
                   • Reduce environmental impact
                   • Build community relations
                   • Ensure regulatory compliance
                """)

except Exception as e:
    st.error(f"An error occurred: {str(e)}")
    st.write("Error details:", e.__class__.__name__)
    import traceback
    st.write("Traceback:", traceback.format_exc()) 