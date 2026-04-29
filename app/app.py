import streamlit as st
import pandas as pd
import pickle
import plotly.express as px
import random
from streamlit_autorefresh import st_autorefresh

# ------------------ PAGE CONFIG ------------------
st.set_page_config(
    page_title="Sentiment Dashboard",
    layout="wide"
)

# ------------------ DARK THEME ------------------
st.markdown("""
<style>
body {background-color: #0E1117; color: white;}
.metric-card {
    background-color: #1E1E1E;
    padding: 20px;
    border-radius: 10px;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

# ------------------ LOAD MODEL ------------------
model = pickle.load(open('models/model.pkl', 'rb'))
vectorizer = pickle.load(open('models/vectorizer.pkl', 'rb'))

# ------------------ SIDEBAR ------------------
st.sidebar.title("⚙️ Controls")
option = st.sidebar.selectbox("Choose Mode", ["Single Prediction", "Live Dashboard"])

# ------------------ INITIAL STATES ------------------
if "single_data" not in st.session_state:
    st.session_state.single_data = pd.read_csv("data/dataset.csv")

if "last_prediction" not in st.session_state:
    st.session_state.last_prediction = None

if "live_data" not in st.session_state:
    st.session_state.live_data = pd.read_csv("data/dataset.csv")

# ------------------ FAKE DATA ------------------
def generate_fake_data():
    samples = [
        ("I love this!", "positive"),
        ("Amazing service", "positive"),
        ("Worst experience", "negative"),
        ("Very bad", "negative"),
        ("It's okay", "neutral"),
        ("Average product", "neutral")
    ]
    return random.choice(samples)

# =====================================================
# ------------------ SINGLE PREDICTION ------------------
# =====================================================
if option == "Single Prediction":

      st.title("📊 Social Media Sentiment Analysis")

user_input = st.text_area("Enter your comment:")

if st.button("Analyze Sentiment"):

        vec = vectorizer.transform([user_input])
        result = model.predict(vec)[0]

        # Save prediction
        st.session_state.last_prediction = result

        # Add to dataset
        new_row = pd.DataFrame({
            "text": [user_input],
            "sentiment": [result]
        })

        st.session_state.single_data = pd.concat(
            [st.session_state.single_data, new_row],
            ignore_index=True
        )

    # ✅ SHOW RESULT ALWAYS (no disappearing)
        if st.session_state.last_prediction is not None:

         result = st.session_state.last_prediction

        if result == "positive":
            st.success("😊 Positive Sentiment")
        elif result == "negative":
            st.error("😡 Negative Sentiment")
        else:
            st.info("😐 Neutral Sentiment")

        # Chart
        st.subheader("📊 Sentiment Distribution")

        fig = px.pie(
            st.session_state.single_data,
            names='sentiment',
            hole=0.4,
            color='sentiment',
            color_discrete_map={
                'positive': '#00CC96',
                'negative': '#EF553B',
                'neutral': '#636EFA'
            }
        )

        st.plotly_chart(fig, use_container_width=True)

# =====================================================
# ------------------ LIVE DASHBOARD ------------------
# =====================================================
elif option == "Live Dashboard":

    # Auto refresh ONLY here
    st_autorefresh(interval=5000, key="refresh")

    st.title("📊 Live Sentiment Dashboard")

    # Add fake data
    new_text, new_sentiment = generate_fake_data()

    new_row = pd.DataFrame({
        "text": [new_text],
        "sentiment": [new_sentiment]
    })

    st.session_state.live_data = pd.concat(
        [st.session_state.live_data, new_row],
        ignore_index=True
    ).tail(100)

    df_live = st.session_state.live_data

    st.subheader("📂 Latest Data")
    st.dataframe(df_live.tail(10))

    # Metrics
    col1, col2, col3 = st.columns(3)

    col1.metric("😊 Positive", len(df_live[df_live['sentiment'] == 'positive']))
    col2.metric("😡 Negative", len(df_live[df_live['sentiment'] == 'negative']))
    col3.metric("😐 Neutral", len(df_live[df_live['sentiment'] == 'neutral']))

    # Pie chart
    st.subheader("📊 Live Sentiment Distribution")

    fig = px.pie(
        df_live,
        names='sentiment',
        hole=0.4,
        color='sentiment'
    )

    st.plotly_chart(fig, use_container_width=True)

    # Trend
    st.subheader("📈 Sentiment Trend")

    df_live['index'] = range(len(df_live))
    trend = df_live.groupby(['index', 'sentiment']).size().unstack().fillna(0)

    st.line_chart(trend)

# ------------------ MODEL EVALUATION ------------------
st.subheader("📉 Model Evaluation")

# Show confusion matrix image
st.image("outputs/confusion_matrix.png", caption="Confusion Matrix")

# Show classification report
with open("outputs/report.txt", "r") as f:
    report = f.read()

st.text("Classification Report")
st.code(report)