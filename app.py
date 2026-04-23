import pickle
import pandas as pd
import numpy as np
import streamlit as st

def predict_habitable(pl_rade,pl_bmasse,pl_orbper,pl_eqt,st_teff,pl_insol):
    try:
        # load the scaler
        with open(scaler_path,'rb') as file1:
            scaler = pickle.load(file1)

        # load the model
        with open(model_path,'rb')as file2:
            model = pickle.load(file2)

        # prepare input data
        dct = {
            'planet_radius':[planet_radius],
            'planet_mass':[planet_mass],
            'planet_orbitalperiod':[planet_orbitalperiod],
            'planet_equilibrium':[planet_equilibrium],
            'planet_host star temperature':[planet_host star temperature],
            'planet_stellar insolation':[planet_stellar insolation]
        }
        x_new = pd.DataFrame(dct)

        # Transform input data
        xnew_pre = scaler.transform(x_new)

        # make predictions
        pred =  model.predict(xnew_pre)
        probs = model.predict_proba(xnew_pre)
        max_prob = np.max(probs)

        return pred,max_prob
    except Exception as e:
        # log and display errors
        st.error(f"Error during predication: {str(e)}")
        return None, None
    
# StreamLit UI
st.title("Exoplanet Habitability Analyzer")

# input fields for the features
planet_radius = st.number_input("planet_radius")
planet_mass = st.number_input("planet_mass")
planet_orbitalperiod = st.number_input("planet_orbitalperiod")
planet_equilibrium	= st.number_input("planet_equilibrium")
planet_host star temperature = st.number_input("planet_host star temperature")
planet_stellar insolation = st.number_input("planet_stellar insolation")

# prediction button
if st.button("Predict"):
    # file paths
    scaler_path = 'notebook/scaler.pkl'
    model_path = 'notebook/model.pkl'

    #call the predictions functions
    pred, max_prob = predict_habitable(pl_rade,pl_bmasse,pl_orbper,pl_eqt,st_teff,pl_insol)

    # Display results
    if pred is not None and max_prob is not None:
        st.subheader(f'Predicated habitable:{pred[0]}')
        st.subheader(f'Prediction Probabiltiy:{max_prob:.4f}')
        st.progress(max_prob)
    else:
        st.error("Predication Failed. Check the input values are model files. ")
