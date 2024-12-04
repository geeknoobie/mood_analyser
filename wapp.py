import streamlit as st
from transformers import pipeline
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
from collections import Counter
import textwrap

# Load models
@st.cache_resource
def load_models():
    emotion_model = pipeline("text-classification", model="joeddav/distilbert-base-uncased-go-emotions-student", top_k=None)
    sentiment_analyzer = SentimentIntensityAnalyzer()
    return emotion_model, sentiment_analyzer

emotion_model, sentiment_analyzer = load_models()

# App title
st.title("Mood Analyzer")

# User input
st.write("### Enter your text below:")
user_input = st.text_area("How are you feeling today?", "")

if st.button("Analyze"):
    if user_input.strip():
        # Analyze sentiment using VADER
        vader_scores = sentiment_analyzer.polarity_scores(user_input)
        sentiment_label = (
            "Positive" if vader_scores["compound"] > 0.05 else
            "Negative" if vader_scores["compound"] < -0.05 else
            "Neutral"
        )

        # Analyze emotions using Transformers
        emotion_scores = emotion_model(user_input)[0]
        dominant_emotion = max(emotion_scores, key=lambda x: x['score'])['label']

        # Display results
        st.write("## Results")
        st.write(f"**Sentiment:** {sentiment_label}")
        st.write(f"**Dominant Emotion:** {dominant_emotion}")

        # Prepare data for the graph
        emotion_labels = [emotion['label'] for emotion in emotion_scores]
        emotion_values = [emotion['score'] for emotion in emotion_scores]

        # Sort emotions by their scores and take the top 6
        sorted_emotions = sorted(zip(emotion_labels, emotion_values), key=lambda x: x[1], reverse=True)[:6]
        top_emotion_labels, top_emotion_values = zip(*sorted_emotions)

        # Create the graph
        fig, ax = plt.subplots(facecolor='none')

        # Detect system theme
        system_theme = st.get_option("theme.base")  # Returns "dark" or "light"
        
        # Set text color dynamically
        if system_theme == "dark":
            text_color = "white"
        else:
            text_color = "black"
        
        # Plot the bar chart
        bars = ax.bar(top_emotion_labels, top_emotion_values)
        
        # Annotate bars with percentage values
        for bar, value in zip(bars, top_emotion_values):
            percent = f"{value * 100:.1f}%"  # Convert to percentage with 1 decimal place
            ax.text(
                bar.get_x() + bar.get_width() / 2,  # X-coordinate
                bar.get_height() + 0.01,           # Y-coordinate (slightly above the bar)
                percent,                           # Annotation text
                ha='center',                       # Center align text
                color=text_color,                  # Dynamic text color
                fontsize=10                        # Font size for annotation
            )
        
        # Set x-axis labels with adjusted font size
        label_count = len(top_emotion_labels)
        fig_width = fig.get_figwidth()
        font_size = max(8, min(12, int(fig_width * 3 / label_count)))
        ax.set_xticks(range(len(top_emotion_labels)))
        ax.set_xticklabels(top_emotion_labels, rotation=0, ha='center', fontsize=font_size, color=text_color)
        
        # Set titles and labels
        ax.set_title("Emotion Analysis", color=text_color)
        ax.set_xlabel("Emotions", color=text_color)
        ax.set_ylabel("Scores", color=text_color)
        ax.set_ylim(0, max(top_emotion_values) + 0.1)
        
        # Make the background transparent
        fig.patch.set_alpha(0)  # Make the overall figure background transparent
        ax.set_facecolor("none")  # Make the axes (plot area) background transparent
        
        # Adjust layout to prevent label overlap
        plt.tight_layout()
        
        # Display the graph in Streamlit
        st.pyplot(fig)
    else:
        st.warning("Please enter some text to analyze.")
