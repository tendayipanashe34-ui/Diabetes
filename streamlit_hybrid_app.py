import streamlit as st
from predictor import FEATURES_12, FEATURES_11, DEFAULTS, load_resources, make_prediction


def main():
    st.title('Diabetes Risk Prediction (Hybrid)')
    st.write('Enter patient features (12 selected features) and predict diabetes risk using the hybrid model.')

    model, scaler = load_resources()
    if model is None or scaler is None:
        st.error('Model or scaler not found. Ensure `diabetes_hybrid_model.pkl` (or `diabetes_ann_model.pkl`) and `scaler.pkl` are present in the working directory.')
        st.stop()

    st.sidebar.header('Patient Inputs')
    inputs = {}
    for feature in FEATURES_12:
        if feature == 'gender':
            g = st.sidebar.radio('gender', options=['Male', 'Female'], index=0)
            inputs['gender'] = 1.0 if g == 'Male' else 0.0
        else:
            inputs[feature] = st.sidebar.number_input(
                label=feature,
                value=float(DEFAULTS.get(feature, 0.0)),
                format='%.2f'
            )

    if st.button('Predict'):
        values12 = [inputs[f] for f in FEATURES_12]
        try:
            label, prob = make_prediction(model, scaler, values12)
        except Exception:
            values11 = [inputs[f] for f in FEATURES_11]
            label, prob = make_prediction(model, scaler, values11)

        st.subheader('Prediction')
        st.write(f'**Class:** {label}')
        st.write(f'**Probability:** {prob:.4f}')
        st.markdown('---')
        st.subheader('Input values')
        st.write(inputs)


if __name__ == '__main__':
    main()
