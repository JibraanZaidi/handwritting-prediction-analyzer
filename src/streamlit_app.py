# src/streamlit_app.py - Frontend for multi-trait predictions
import streamlit as st, requests
from PIL import Image
API_URL = 'http://localhost:8000/predict_all'
st.set_page_config(page_title='Handwriting Personality Predictor v2')
st.title('✍️ Handwriting Personality Predictor')
st.markdown('**Disclaimer:** Demo only. Not clinically validated.')
uploaded = st.file_uploader('Upload a handwriting image (jpg/png)', type=['png','jpg','jpeg'])
if uploaded:
    img = Image.open(uploaded).convert('RGB')
    st.image(img, caption="Uploaded Image", use_container_width=True)

    if st.button('Analyze handwriting'):
        with st.spinner('Analyzing...'):
            files = {'file': (uploaded.name, uploaded.getvalue(), 'image/jpeg')}
            try:
                res = requests.post(API_URL, files=files, timeout=30)
                res.raise_for_status()
                out = res.json()
                st.success('Done — results:')
                for trait, info in out.items():
                    if 'error' in info:
                        st.write(f"**{trait}**: ERROR - {info['error']}")
                    else:
                        st.write(f"**{trait}** — probability: {info['probability']:.3f}, prediction: {info['prediction']}")
            except Exception as e:
                st.error('API error: '+str(e))
