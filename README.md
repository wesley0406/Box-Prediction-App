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
## How to use
You can enter the screw spec for the head type and the screw type shortname please clikc the Reference to see what you are after.
The system will tell you how big the box should be to contain all the screw.


<img src="https://raw.githubusercontent.com/wesley0406/Box-Prediction-App/main/screenshot/mainpage.JPG" width="1060" height="350" />

# Result 
The Bar-Chart below will show you hoe close the screw fit in the box, normally we suggest 0~-10% will be suitable for thr daily use.
Once the number fall below -10% it will be overpacked.
<img src="https://raw.githubusercontent.com/wesley0406/Box-Prediction-App/main/screenshot/result.JPG" width="1060" height="450" />

# Result with personal desgin box

<img src="https://raw.githubusercontent.com/wesley0406/Box-Prediction-App/main/screenshot/customized_input.JPG" width="1060" height="450" />




## Model Information
The application utilizes a deep learning regression model designed to predict box-related outputs based on both categorical and numerical features

- Embedding layers for categorical features (Screw Type, Head Type)
- Numerical inputs are processed through fully connected (dense) layers to capture nonlinear relationships
- Uses Huber Loss, providing robustness against outliers while maintaining sensitivity for small errors
- Applies Bayesian Optimization to automatically tune model parameters (e.g., learning rate, epoch), improving performance efficiently
