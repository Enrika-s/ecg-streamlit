import streamlit as st
import pandas as pd
import pickle

@st.cache_resource
def load_model():
    with open('ecg_model/trained_model.pkl', 'rb') as f:
        model, scaler = pickle.load(f)
    return model, scaler

def preprocess_input(data, scaler):
    if data.shape[1] != 32:
        st.error("Input data must have 32 columns corresponding to the features used in training.")
        return None
    return scaler.transform(data)

def main():
    st.set_page_config(page_title="ECG Classification App", page_icon="❤️", layout="centered")

    st.markdown("""
    <style>
    .banner {
        width: 100%;
        height: auto;
        margin-bottom: 20px;
    }
    .custom-title {
        font-size: 3em;
        text-align: center;
    }
    .emergency-text {
        text-align: center;
        font-size: 1em;
        margin-top: 50px;
    }
    .emergency-text .highlight {
        color: red;
        font-weight: bold;
    }
    .disclaimer-container {
        background-color: #ffcccc;
        padding: 20px;
        border-radius: 10px;
        color: black;
        text-align: center;
        margin-bottom: 20px;
    }
    .disclaimer-button-container {
        display: flex;
        justify-content: center;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div>
        <img src="https://www.aedsuperstore.com/resources/wp-content/uploads/sites/4/2016/10/Normal-Sinus-Rhythm.gif" class="banner">
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<h1 class="custom-title">ECG Classification App</h1>', unsafe_allow_html=True)
    st.markdown("""
    Welcome to the ECG Classification App. This tool allows you to upload an ECG file in CSV format and get a classification prediction indicating whether the ECG is **Normal** or **Abnormal (Arrhythmia)**.
    """)

    if "show_disclaimer" not in st.session_state:
        st.session_state.show_disclaimer = True

    if st.session_state.show_disclaimer:
        st.markdown("""
        <div class="disclaimer-container">
            <p><strong>Disclaimer</strong>: This app is for educational purposes only and should not be used for medical diagnosis or treatment. Always consult with a healthcare professional for any medical concerns.</p>
        </div>
        """, unsafe_allow_html=True)
        col_center = st.columns([2, 1, 2])[1]
        with col_center:
            if st.button("I Understand", key="acknowledge"):
                st.session_state.show_disclaimer = False
                st.experimental_rerun()
    else:
        uploaded_file = st.file_uploader("Choose a CSV file containing ECG data", type="csv")

        if uploaded_file is not None:
            user_data = pd.read_csv(uploaded_file)

            with st.expander("Show Uploaded Data"):
                st.write(user_data)

            if st.button('Analyze my results'):
                with st.spinner('Analyzing data... ❤️'):
                    model, scaler = load_model()
                    processed_data = preprocess_input(user_data, scaler)
                    if processed_data is not None:
                        prediction = model.predict(processed_data)
                        prediction_proba = model.predict_proba(processed_data)

                        prediction_label = 'Normal' if prediction[0] == 0 else 'Abnormal (Arrhythmia)'
                        confidence_score = prediction_proba[0][prediction[0]] * 100

                        st.write("## Prediction Result")
                        st.markdown(f"<h3 style='text-align: center; color: {'green' if prediction[0] == 0 else 'red'};'>{prediction_label} ({confidence_score:.2f}% confidence)</h3>", unsafe_allow_html=True)

                        if prediction[0] == 0:
                            st.image("https://i.postimg.cc/nhMj7hpL/Normal.png", caption="ECG - Normal Sinus Rhythm", use_column_width=True)
                            st.markdown(f"""
                            Your ECG appears **normal**.
                            This means that your **heart rhythm appears to be consistent**, suggesting that **your heart is healthy and functioning as it should**!


                            Keep your heart healthy by:

                            - Staying active and getting 150 minutes of **exercise** per week
                            - Eating heart-healthy foods
                            - Managing stress


                            Please note: While our model has a **confidence level of {confidence_score:.2f}%** in your predicted results (read more about what that number means **here**), it does have limitations:

                            - This is only one point in time - take multiple recordings to get the bigger picture of your heart function.
                            - This model uses a single or 2-lead ECG input, which allows for more flexible and convenient measurement, but means it is inherently incomplete compared to the clinical standard 12-lead ECG.
                            - If you are experiencing symptoms of a heart event or have reason to believe you are at risk of having a heart problem - see a medical professional.

                            While this model can be useful for capturing ECG results measured outside of a clinical setting and predicting outcomes with 99% accuracy, it cannot replace a clinical assessment by a medical professional.
                            """)
                        else:
                            st.image("https://i.postimg.cc/br3rDK5b/Arrhythmia.png", caption="ECG - Abnormal (Arrhythmia)", use_column_width=True)
                            st.markdown(f"""
                            Your ECG appears **abnormal**.

                            This means that your **heart rhythm appears inconsistent in this measurement**, which **may be a sign of heart disease or arrhythmia**.

                            <!-- **You should consult with a medical doctor for further assessment**.

                            **Record the following information to help your clinician**:

                            - **What symptoms prompted you to record your ECG**?
                            - **What were you doing when these symptoms appeared, or do you believe you know what caused them**?
                            - **Has this happened before**?
                            - **Have you received a prior diagnosis of heart disease or had a heart event in the past**?
                            - **Do you have a family member who has had a heart event or died of heart disease**?
                            - **Save a copy of this ECG recording and prediction and bring it with you**! -->

                            **Your family doctor can refer you for a full ECG assessment, or you can check your local walk-in clinic or hospital emergency room**.

                            **Taking action now can have a meaningful impact on your ability to proactively manage and treat potential heart-related complications - and improve your health in both the short and long term**!

                            **Please note**: While our model has a **confidence level of {confidence_score:.2f}%** in your predicted results (read more about what that number means here), it does have limitations:

                            - This is only one point in time - take multiple recordings to get the bigger picture of your heart function.
                            - This model uses a single or 2-lead ECG input, which allows for more flexible and convenient measurement, but means it is inherently incomplete compared to the clinical standard 12-lead ECG.
                            - If you are experiencing symptoms of a heart event or have reason to believe you are at risk of having a heart problem - see a medical professional.

                            While this model can be useful for capturing ECG results measured outside of a clinical setting, and predict outcomes with 99% accuracy, it cannot replace a clinical assessment by a medical professional.
                            """)
                    else:
                        st.error("An error occurred during prediction: Invalid data format")

    st.markdown("""
    <div class="emergency-text">
        If you believe you are having a heart attack or a medical emergency, call your local <a href="https://en.wikipedia.org/wiki/List_of_emergency_telephone_numbers" class="highlight">emergency services</a>.
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
