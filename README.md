    # Handwriting Personality Predictor v2 (Improved)

    This version includes enhanced feature extraction (slant, margins, spacing) and trains separate RandomForest models for each Big Five trait.

Quickstart (Windows PowerShell):

cd "E:\\Ai project\\handwriting_personality_project_v2"
python -m venv venv
.\\venv\\Scripts\\Activate
pip install -r requirements.txt
python src/dataset_prep.py   # creates data/features.csv from images + labels
python src/train_model.py    # trains models and saves them into models/
python src/app_backend.py    # run backend
streamlit run src/streamlit_app.py

Note: dataset_prep uses OpenCV/scikit-image; ensure requirements install. The repo includes one synthetic sample file and sample labels.csv for demo.
