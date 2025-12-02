# src/dataset_prep.py - Enhanced feature extraction
import os, cv2, numpy as np, pandas as pd
from skimage.feature import hog
from skimage.filters import threshold_otsu
from scipy import ndimage as ndi

def load_image(path, gray=True, target_size=(512,256)):
    im = cv2.imread(path)
    if im is None:
        raise FileNotFoundError(path)
    if gray:
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    im = cv2.resize(im, target_size)
    return im

def estimate_slant_angle(img):
    edges = cv2.Canny(img, 50, 150)
    lines = cv2.HoughLines(edges, 1, np.pi/180, 200)
    if lines is None:
        return 0.0
    angles = []
    for rho,theta in lines[:,0]:
        angle = (theta - np.pi/2) * (180/np.pi)
        angles.append(angle)
    return float(np.median(angles)) if angles else 0.0

def avg_interword_spacing(bw):
    h = bw.shape[0]
    gaps = []
    for row in range(10, h-10, 10):
        line = bw[row, :]
        zeros = []
        count = 0
        for v in line:
            if v==0:
                count += 1
            else:
                if count>0:
                    zeros.append(count)
                count = 0
        if zeros:
            gaps.append(np.median(zeros))
    return float(np.median(gaps)) if gaps else 0.0

def extract_basic_features(img):
    features = {}
    th = threshold_otsu(img)
    bw = (img < th).astype(np.uint8)
    features['stroke_density'] = float(bw.mean())
    dist = cv2.distanceTransform(1 - bw, cv2.DIST_L2, 5)
    features['median_stroke_thickness'] = float(np.median(dist[bw==1]) if bw.sum()>0 else 0.0)
    vsum = bw.sum(axis=0)
    hsum = bw.sum(axis=1)
    features['v_proj_std'] = float(np.std(vsum))
    features['h_proj_std'] = float(np.std(hsum))
    features['slant_angle'] = estimate_slant_angle(img)
    h,w = bw.shape
    cols = bw.sum(axis=0)
    rows = bw.sum(axis=1)
    left_margin = int(np.argmax(cols>0)) if cols.max()>0 else 0
    right_margin = int(w - np.argmax(cols[::-1]>0)) if cols.max()>0 else w
    top_margin = int(np.argmax(rows>0)) if rows.max()>0 else 0
    bottom_margin = int(h - np.argmax(rows[::-1]>0)) if rows.max()>0 else h
    features['left_margin_frac'] = float(left_margin / w)
    features['right_margin_frac'] = float((w - right_margin) / w)
    features['top_margin_frac'] = float(top_margin / h)
    features['bottom_margin_frac'] = float((h - bottom_margin) / h)
    features['avg_interword_spacing'] = avg_interword_spacing(bw)
    hog_feat = hog(img, pixels_per_cell=(32,32), cells_per_block=(2,2), feature_vector=True)
    hog_feat = hog_feat[:64]
    for i, v in enumerate(hog_feat):
        features[f'hog_{i}'] = float(v)
    return features

def build_feature_dataframe(images_dir, labels_csv, target_size=(512,256)):
    labels = pd.read_csv(labels_csv)
    rows = []
    for _, row in labels.iterrows():
        fname = row['filename']
        path = os.path.join(images_dir, fname)
        try:
            img = load_image(path, target_size=target_size)
        except FileNotFoundError:
            print('missing', path); continue
        feats = extract_basic_features(img)
        feats['filename'] = fname
        for col in labels.columns:
            if col != 'filename':
                feats[col] = row[col]
        rows.append(feats)
    df = pd.DataFrame(rows)
    return df.fillna(0.0)

if __name__ == '__main__':
    df = build_feature_dataframe('data/samples', 'data/labels.csv')
    df.to_csv('data/features.csv', index=False)
    print('Saved data/features.csv')
