# src/app_backend.py - loads multiple trait models & predicts
from fastapi import FastAPI, File, UploadFile
import joblib, os, cv2, numpy as np, pandas as pd
from io import BytesIO
from PIL import Image
import sys
sys.path.append(os.path.dirname(__file__))
from dataset_prep import extract_basic_features

app = FastAPI(title='Handwriting Personality API')

MODEL_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', 'models'))

TRAITS = ['extrovert','neurotic','open','agree','conscientious']
models = {}
for t in TRAITS:
    path = os.path.join(MODEL_DIR, f'rf_{t}.pkl')
    if os.path.exists(path):
        models[t] = joblib.load(path)
    else:
        models[t] = None

@app.post('/predict_all')
async def predict_all(file: UploadFile = File(...)):
    contents = await file.read()
    img = Image.open(BytesIO(contents)).convert('L')
    img = np.array(img)
    img = cv2.resize(img, (512,256))
    feats = extract_basic_features(img)
    results = {}
    for trait, bundle in models.items():
        if bundle is None:
            results[trait] = {'error':'model missing'}
            continue
        feat_list = bundle['features']
        X = pd.DataFrame([feats])[feat_list]
        prob = float(bundle['model'].predict_proba(X)[:,1][0])
        pred = int(prob > 0.5)
        results[trait] = {'probability': prob, 'prediction': pred}
    return results

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8000)
