import streamlit as st
import pickle
import warnings
import pandas as pd
import plotly.graph_objects as go
import wikipedia

warnings.filterwarnings("ignore")


# Load the dataset
@st.cache_data
def load_data():
    return pd.read_csv('assets/WebMD_DV.csv')

# Load the prediction model


@st.cache_data
def load_prediction_model():
    return pickle.load(open('prediction_model.sav', 'rb'))

# Create a dictionary to map conditions to drugs


@st.cache_data
def create_condition_drugs_map(data):
    condition_drugs_map = {}
    for condition in data['Condition'].unique():
        drugs = data[data['Condition'] == condition]['Drug'].unique()
        condition_drugs_map[condition] = drugs.tolist()
    return condition_drugs_map


def main():
    st.set_page_config(layout='wide')
    data = load_data()
    condition_drugs_map = create_condition_drugs_map(data)

    title = '<h1 style="font-family:Arial; color:White;">Better Information, Better Health</h1>'
    st.markdown(title, unsafe_allow_html=True)
    subheader = '<h3 style="font-family:Arial; color:White;">Disease Prediction and Drug Recommendation</h3>'
    st.markdown(subheader, unsafe_allow_html=True)

    spacer = st.empty()
    spacer.write("\n\n")

    # Symptom Prediction
    prediction_model = load_prediction_model()
    symptoms = ['Itching', 'Skin Rash', 'Nodal Skin Eruptions', 'Continuous Sneezing', 'Shivering', 'Chills', 'Joint Pain',
                'Stomach Pain', 'Acidity', 'Ulcers On Tongue', 'Muscle Wasting', 'Vomiting', 'Burning Micturition',
                'Spotting  Urination', 'Fatigue', 'Weight Gain', 'Anxiety', 'Cold Hands And Feets', 'Mood Swings',
                'Weight Loss', 'Restlessness', 'Lethargy', 'Patches In Throat', 'Irregular Sugar Level', 'Cough',
                'High Fever', 'Sunken Eyes', 'Breathlessness', 'Sweating', 'Dehydration', 'Indigestion', 'Headache',
                'Yellowish Skin', 'Dark Urine', 'Nausea', 'Loss Of Appetite', 'Pain Behind The Eyes', 'Back Pain',
                'Constipation', 'Abdominal Pain', 'Diarrhoea', 'Mild Fever', 'Yellow Urine', 'Yellowing Of Eyes',
                'Acute Liver Failure', 'Fluid Overload', 'Swelling Of Stomach', 'Swelled Lymph Nodes', 'Malaise',
                'Blurred And Distorted Vision', 'Phlegm', 'Throat Irritation', 'Redness Of Eyes', 'Sinus Pressure',
                'Runny Nose', 'Congestion', 'Chest Pain', 'Weakness In Limbs', 'Fast Heart Rate',
                'Pain During Bowel Movements', 'Pain In Anal Region', 'Bloody Stool', 'Irritation In Anus', 'Neck Pain',
                'Dizziness', 'Cramps', 'Bruising', 'Obesity', 'Swollen Legs', 'Swollen Blood Vessels',
                'Puffy Face And Eyes', 'Enlarged Thyroid', 'Brittle Nails', 'Swollen Extremeties', 'Excessive Hunger',
                'Extra Marital Contacts', 'Drying And Tingling Lips', 'Slurred Speech', 'Knee Pain', 'Hip Joint Pain',
                'Muscle Weakness', 'Stiff Neck', 'Swelling Joints', 'Movement Stiffness', 'Spinning Movements',
                'Loss Of Balance', 'Unsteadiness', 'Weakness Of One Body Side', 'Loss Of Smell', 'Bladder Discomfort',
                'Foul Smell Of Urine', 'Continuous Feel Of Urine', 'Passage Of Gases', 'Internal Itching',
                'Toxic Look (Typhos)', 'Depression', 'Irritability', 'Muscle Pain', 'Altered Sensorium',
                'Red Spots Over Body', 'Belly Pain', 'Abnormal Menstruation', 'Dischromic  Patches', 'Watering From Eyes',
                'Increased Appetite', 'Polyuria', 'Family History', 'Mucoid Sputum', 'Rusty Sputum',
                'Lack Of Concentration', 'Visual Disturbances', 'Receiving Blood Transfusion',
                'Receiving Unsterile Injections', 'Coma', 'Stomach Bleeding', 'Distention Of Abdomen',
                'History Of Alcohol Consumption', 'Fluid Overload.1', 'Blood In Sputum', 'Prominent Veins On Calf',
                'Palpitations', 'Painful Walking', 'Pus Filled Pimples', 'Blackheads', 'Scurring', 'Skin Peeling',
                'Silver Like Dusting', 'Small Dents In Nails', 'Inflammatory Nails', 'Blister',
                'Red Sore Around Nose', 'Yellow Crust Ooze']

    st.sidebar.header("Navigation Bar")
    page = st.sidebar.selectbox("Choose an option", options=[
                                "Disease Predictor", "Customer Satisfaction", "Ease of Use and Effectiveness", "Customer Sentiment"])

    if page == "Disease Predictor":
        title = '<h2 style="font-family:Arial; color:Green;">Disease Predictor</h2>'
        st.markdown(title, unsafe_allow_html=True)
        selected_symptoms = st.multiselect("Select Symptoms:", symptoms)

        arr = []
        for symptom in selected_symptoms:
            arr.append(symptoms.index(symptom))

        model_input = [0] * len(symptoms)
        for element in arr:
            model_input[element] = 1

        model_input = [model_input]

        if st.button("Predict"):
            prediction = prediction_model.predict(model_input)
            st.write("Predicted Disease:", prediction[0])
            st.write(wikipedia.summary(prediction[0]))

    elif page == "Customer Satisfaction":
        st.header(":blue[Customer Satisfaction of different drugs]")
        selected_condition = st.selectbox(
            'Select Condition', list(condition_drugs_map.keys()))

        selected_drugs = condition_drugs_map[selected_condition]

        filtered_data = data[(data['Condition'] == selected_condition) & (
            data['Drug'].isin(selected_drugs))]

        vote_counts = filtered_data.groupby(
            ['Drug', 'Satisfaction']).size().reset_index(name='VoteCount')

        fig = go.Figure()

        for satisfaction_level in vote_counts['Satisfaction'].unique():
            level_data = vote_counts[vote_counts['Satisfaction']
                                     == satisfaction_level]
            fig.add_trace(go.Scatter(
                x=level_data['Drug'],
                y=[satisfaction_level] * len(level_data),
                mode='markers',
                marker=dict(
                    size=level_data['VoteCount'],
                    sizemode='diameter',
                    sizeref=0.1,
                    sizemin=5,
                    color=level_data['Satisfaction'],
                    colorscale='peach',
                    showscale=False
                ),
                text=level_data['Drug'],
                hovertemplate='Drug: %{text}<br>Satisfaction: %{y}<br>Vote Count: %{marker.size}<extra></extra>'
            ))

        fig.update_layout(
            title=f'Bubble Chart of Customer Satisfaction for different drugs available for {selected_condition}',
            xaxis_title='Drug',
            yaxis_title='Satisfaction',
            xaxis=dict(
                # Specify the font size for the x-axis label
                title_font=dict(size=24, color='black')
            ),
            yaxis=dict(
                # Specify the font size for the y-axis label
                title_font=dict(size=24, color='black')
            ),
            hovermode='closest',
            showlegend=False,
            height=800,  # Increase the height to fit the entire page
        )

        fig.update_layout(title_font=dict(size=24))

        st.plotly_chart(fig, use_container_width=True)

    elif page == "Ease of Use and Effectiveness":
        st.header(":blue[Ease of Use and Effectiveness]")
        selected_condition = st.selectbox(
            'Select Condition', list(condition_drugs_map.keys()))
        selected_drugs = condition_drugs_map[selected_condition]
        selected_drug = st.selectbox('Select Drug', selected_drugs)

        filtered_data = data[(data['Condition'] == selected_condition) & (
            data['Drug'] == selected_drug)]

        total_votes = len(filtered_data)
        ease_of_use_percentage = (
            filtered_data['EaseofUse'].value_counts() / total_votes) * 100
        effectiveness_percentage = (
            filtered_data['Effectiveness'].value_counts() / total_votes) * 100

        donut_chart_ease_of_use = go.Figure(data=go.Pie(
            values=ease_of_use_percentage,
            labels=filtered_data['EaseofUse'].unique(),
            hole=0.5,
            textinfo='label+percent',
            insidetextorientation='radial'
        ))

        donut_chart_ease_of_use.update_layout(
            title=f'Rating Distribution for Ease of Use of {selected_drug} for {selected_condition}',
            height=500
        )
        donut_chart_ease_of_use.update_layout(title_font=dict(size=24))

        donut_chart_effectiveness = go.Figure(data=go.Pie(
            values=effectiveness_percentage,
            labels=filtered_data['Effectiveness'].unique(),
            hole=0.5,
            textinfo='label+percent',
            insidetextorientation='radial'
        ))

        donut_chart_effectiveness.update_layout(
            title=f'Rating Distribution for Effectiveness of {selected_drug} for {selected_condition}',
            height=500
        )
        donut_chart_effectiveness.update_layout(title_font=dict(size=24))

        st.plotly_chart(donut_chart_ease_of_use, use_container_width=True)
        st.plotly_chart(donut_chart_effectiveness, use_container_width=True)

    elif page == "Customer Sentiment":
        st.header(":blue[Customer Sentiment]")
        selected_condition = st.selectbox(
            'Select Condition', list(condition_drugs_map.keys()))

        selected_drugs = condition_drugs_map[selected_condition]

        filtered_data = data[(data['Condition'] == selected_condition) & (
            data['Drug'].isin(selected_drugs))]

        trace_positive = go.Scatter(
            x=filtered_data[filtered_data['SentimentScore'] > 0]['Drug'],
            y=filtered_data[filtered_data['SentimentScore'] > 0]['SentimentScore'],
            mode='markers',
            marker=dict(
                color='green',
                symbol='circle',
                size=10,
                line=dict(
                    color='black',
                    width=1
                )
            ),
            name='Positive Sentiment',
            text=filtered_data[filtered_data['SentimentScore'] > 0]['Drug'],
            hovertemplate='Drug: %{text}<br>SentimentScore: %{y}<extra></extra>'
        )

        trace_negative = go.Scatter(
            x=filtered_data[filtered_data['SentimentScore'] < 0]['Drug'],
            y=filtered_data[filtered_data['SentimentScore'] < 0]['SentimentScore'],
            mode='markers',
            marker=dict(
                color='red',
                symbol='circle',
                size=10,
                line=dict(
                    color='black',
                    width=1
                )
            ),
            name='Negative Sentiment',
            text=filtered_data[filtered_data['SentimentScore'] < 0]['Drug'],
            hovertemplate='Drug: %{text}<br>SentimentScore: %{y}<extra></extra>'
        )

        trace_neutral = go.Scatter(
            x=filtered_data[filtered_data['SentimentScore'] == 0]['Drug'],
            y=filtered_data[filtered_data['SentimentScore'] == 0]['SentimentScore'],
            mode='markers',
            marker=dict(
                color='blue',
                symbol='circle',
                size=10,
                line=dict(
                    color='black',
                    width=1
                )
            ),
            name='Neutral Sentiment',
            text=filtered_data[filtered_data['SentimentScore'] == 0]['Drug'],
            hovertemplate='Drug: %{text}<br>SentimentScore: %{y}<extra></extra>'
        )

        fig = go.Figure(data=[trace_positive, trace_negative, trace_neutral])
        fig.update_layout(
            title=f'Customer sentiment for various drugs used to treat {selected_condition}',
            xaxis_title='Drug',
            yaxis_title='Sentiment Score',
            xaxis=dict(
                title_font=dict(size=24, color='black')
            ),
            yaxis=dict(
                title_font=dict(size=24, color='black')
            ),
            showlegend=True,
            height=800
        )
        fig.update_layout(title_font=dict(size=24))

        st.plotly_chart(fig, use_container_width=True)



if __name__ == "__main__":
    main()
