# Box Prediction Tool

A Streamlit-based machine-learning system that recommends optimal packaging configurations—including box size selection and packing efficiency—for screws using dimensional, quantity, and type inputs.

## Features
- **Smart Prediction**: Calculates the predicted packing ratio and decision volume using a trained neural network.
- **Box Matching**: Automatically compares predictions against a dataset of standard boxes to find the best fit.
- **Visualization**: Interactive Plotly charts to visualize the "Extra Space %" for predicted boxes.
- **Custom Comparison**: Input your own box dimensions to see how they compare to the AI-recommended options.
- **Standalone & Portable**: Self-contained package with all necessary model artifacts.

## File Structure
```text
Box_Prediction_App/
├── models/               # Trained model, config, and preprocessors
├── streamlit_app.py      # Main Streamlit UI application
├── Predict.py            # Core prediction logic
├── NN_Structure_EMBED.py # Neural network architecture
├── Box_Choice.xlsx       # Box dimension database
├── HEAD_TYPE.xlsx        # Reference data for head types
├── requirements.txt      # Python dependencies
└── setup.bat             # (Windows) Automated setup and run script
```

## Getting Started (Windows)

1. **Prerequisites**: Ensure [Python 3.9+](https://www.python.org/downloads/) is installed.
2. **Install & Run**: Double-click `setup.bat`. This will:
   - Create a local virtual environment (`venv`).
   - Install required libraries (`streamlit`, `tensorflow`, etc.).
   - Launch the application in your browser.

## Manual Installation

If you prefer to set it up manually:

1. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the app:
   ```bash
   streamlit run streamlit_app.py
   ```

## Model Information
The application uses a deep learning model (trained on 2025-09-04) featuring:
- Embedding layers for categorical features (Screw Type, Head Type).
- Dense layers for numerical feature extraction.
- Huber loss for robust regression against outliers.
