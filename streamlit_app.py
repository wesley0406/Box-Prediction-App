import streamlit as st
import pandas as pd
import sys
import os
import datetime
import plotly.graph_objects as go

# --- Setup ---
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)
os.chdir(current_dir)

try:
    from Predict import Prediction_Package
except ImportError as e:
    st.error(f"Failed to import Prediction_Package: {e}")
    st.stop()

# --- Helper Functions ---
def create_fill_percent_bar(screw_vol, box_codes, box_volumes):
    """
    Creates a horizontal bar chart showing the percentage of extra space (or fill).
    adapted from VISUALIZE_QUOTE_V6.1.py
    """
    # Calculate fill percentage (or rather, "Extra Space after Filled" as per original label)
    # Original logic: fill_pcts = [(1 - screw_vol / vol) * 100 if vol else 0 for vol in box_volumes]
    # If screw_vol > vol, this will be negative (overfilled)
    # If screw_vol < vol, this will be positive (extra space)
    fill_pcts = []
    hover_texts = []
    
    for vol in box_volumes:
        if vol > 0:
            pct = (1 - screw_vol / vol) * 100
            fill_pcts.append(pct)
            hover_texts.append(f"Box Vol: {vol:,} mm³<br>Screw Vol: {screw_vol:,.2f} mm³")
        else:
            fill_pcts.append(0)
            hover_texts.append("Invalid Volume")

    fig = go.Figure()

    # Colors for S-series (Yellow/Orange) and N-series (Grey/Dark)
    # Assuming order: S_small, S_large, N_small, N_large based on Predict.py output structure
    colors = ['#EDC948', '#DC793E', '#B2BEB5', '#5A5A5A']
    
    # Adjust colors if fewer bars
    current_colors = colors[:len(box_codes)] if len(box_codes) <= len(colors) else colors * (len(box_codes) // len(colors) + 1)

    fig.add_trace(go.Bar(
        x=fill_pcts,
        y=box_codes,
        orientation='h',
        marker_color=current_colors[:len(box_codes)],
        text=[f"{p:.1f}%" for p in fill_pcts],
        textposition="auto",
        hovertext=hover_texts,
        name="Fill %"
    ))

    fig.update_layout(
        title="Extra Space % (Positive = Fits, Negative = Overfilled)",
        xaxis=dict(title="Extra Space %", ticksuffix="%"),
        yaxis=dict(autorange="reversed"), # Top strings at top
        height=250,
        margin=dict(t=30, l=10, r=10, b=10),
        showlegend=False
    )

    return fig

# --- Streamlit App ---

st.set_page_config(page_title="Box Prediction Tool", layout="wide")

st.title("Box Prediction Tool")
st.markdown("""
Predict box packing ratio and volume for screws based on dimensions and type.
""")

# Input Form
with st.form("prediction_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        size_input = st.text_input("Size (Diameter x Length)", value="M4.5x20", help="Format: Diameter x Length (e.g., #8x1 or M4.5x20)")
        quantity = st.number_input("Quantity", min_value=1, value=1000)

    with col2:
        screw_type = st.selectbox("Screw Type",  
            ("SCB", "SDW", "SWD","SMS", "SDS", "STP", "STT", "STC"),
            placeholder = "Choose the screw type")
        head_type = st.selectbox("Head Type", ("FLT", "PAN", "BUG", "HEX", "OV9", "SOC", "TRU","TRM","MOD", "CHE"),
            placeholder = "Choose the head type")
        

    with st.expander("Compare with Custom Box (Optional)"):
        custom_box_name = st.text_input("Box Name", value="My Custom Box")
        c1, c2, c3 = st.columns(3)
        with c1:
            custom_L = st.number_input("Length (mm)", min_value=0.0, step=1.0)
        with c2:
            custom_W = st.number_input("Width (mm)", min_value=0.0, step=1.0)
        with c3:
            custom_H = st.number_input("Height (mm)", min_value=0.0, step=1.0)

    submitted = st.form_submit_button("Predict")

if submitted:
    # Parse Size Input
    diameter = None
    length = None
    
    try:
        if 'x' in size_input:
            parts = size_input.split('x')
            diameter = parts[0].strip()
            length = parts[1].strip()
        else:
            st.error("Invalid Size format. Please use 'Diameter x Length' (e.g., #8x1).")
            st.stop()
            
        # Create Data Structure for Prediction
        input_data = pd.DataFrame([{
            "Diameter": diameter,
            "Length": length,
            "Quantity": int(quantity),
            "pdc_4": "-",
            "pdc_5": "-",
            "Screw_Type": screw_type,
            "Head_Type": head_type,
            "Box type": None
        }])

        st.write("### Input Data")
        st.dataframe(input_data)

        # Run Prediction
        with st.spinner("Calculating prediction..."):
            try:
                bot = Prediction_Package()
                bot.WEB_SOURCE = input_data
                bot.predict_product()
                bot.Match_Box()
                
                result = bot.Result
                
                st.write("### Prediction Results")
                
                if not result.empty:
                    # Metrics
                    c1, c2 = st.columns(2)
                    c1.metric(label="Predicted Packing Ratio", value=f"{result.iloc[0]['predicted_packing_ratio']:.4f}")
                    c2.metric(label="Predicted Decision Volume", value=f"{result.iloc[0]['predicted_decision_volume']:.2f}")
                    
                    st.write("#### Detailed Result")
                    with st.spinner("Raw data"):
                        st.dataframe(result)

                    row = result.iloc[0]
                    matched_box_str = row.get("Matched_Box")
                    screw_vol = row.get("predicted_decision_volume", 0)

                    # Visualization Logic
                    if matched_box_str and isinstance(matched_box_str, str):
                        box_codes_raw = matched_box_str.split("/")
                        # Filter valid codes (ignore None/empty)
                        valid_codes = []
                        valid_volumes = []
                        
                        # Need to access bot.ALL_Box for volumes. 
                        # bot.Match_Box() calculates Volume(mm³) in S_Box, N_Box, ALL_Box
                        
                        for code in box_codes_raw:
                            if code and code != "None":
                                # Find volume in ALL_Box
                                match = bot.ALL_Box[bot.ALL_Box["Quote_Code"] == code]
                                if not match.empty:
                                    vol = match["Volume(mm³)"].values[0]
                                    valid_codes.append(code)
                                    valid_volumes.append(vol)
                        
                        # --- Custom Box Visualization Logic ---
                        if custom_L > 0 and custom_W > 0 and custom_H > 0:
                            custom_vol = custom_L * custom_W * custom_H
                            valid_codes.append(custom_box_name)
                            valid_volumes.append(custom_vol)
                        
                        if valid_codes:
                            st.write("#### Box Fill Visualization")
                            fig = create_fill_percent_bar(screw_vol, valid_codes, valid_volumes)
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.warning("No valid box codes found for visualization.")


                else:
                    st.warning("No results returned.")

            except Exception as e:
                st.error(f"An error occurred during prediction: {e}")
                import traceback
                traceback.print_exc()

    except Exception as parse_error:
        st.error(f"Error parsing input: {parse_error}")
