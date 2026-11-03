import gradio as gr
import pandas as pd
import plotly.express as px
import nltk

from nltk.sentiment import SentimentIntensityAnalyzer

nltk.download("vader_lexicon")

sia = SentimentIntensityAnalyzer()


def analyze_movie_review(review):

    if not review or not review.strip():
        return "Please enter a movie review.", {}

    scores = sia.polarity_scores(review)

    compound = scores["compound"]

    if compound >= 0.05:
        sentiment = "Positive"
    elif compound <= -0.05:
        sentiment = "Negative"
    else:
        sentiment = "Neutral"

    result = {
        "Positive": scores["pos"],
        "Neutral": scores["neu"],
        "Negative": scores["neg"]
    }

    return sentiment, result

def analyze_multiple_reviews(text):

    reviews = [
        line.strip()
        for line in text.split("\n")
        if line.strip()
    ]

    if not reviews:
        return pd.DataFrame(), None

    data = []

    for review in reviews:

        scores = sia.polarity_scores(review)

        compound = scores["compound"]

        if compound >= 0.05:
            sentiment = "Positive"
        elif compound <= -0.05:
            sentiment = "Negative"
        else:
            sentiment = "Neutral"

        data.append({
            "Movie Review": review,
            "Sentiment": sentiment,
            "Compound Score": compound
        })

    df = pd.DataFrame(data)

    counts = df["Sentiment"].value_counts().reindex(
        ["Positive", "Neutral", "Negative"],
        fill_value=0
    )

    fig = px.pie(
        names=counts.index,
        values=counts.values,
        title="Movie Review Sentiment Distribution",
        hole=0.4,
        color=counts.index,
        color_discrete_map={
            "Positive": "#16A34A",
            "Neutral": "#F59E0B",
            "Negative": "#DC2626"
        }
    )

    return df, fig

with gr.Blocks() as demo:

    gr.Markdown(
        """
        # 🎬 AI Movie Review Analyzer

        ### NLP-Based Sentiment Analysis System

        Enter a movie review and discover its sentiment.
        """
    )

    with gr.Tab("Single Review Analysis"):

        review_input = gr.Textbox(
            label="Enter Movie Review",
            placeholder="Write your movie review here...",
            lines=5
        )

        analyze_btn = gr.Button(
            "Analyze Sentiment",
            variant="primary"
        )

        sentiment_output = gr.Textbox(
            label="Predicted Sentiment"
        )

        score_output = gr.JSON(
            label="Sentiment Scores"
        )

        analyze_btn.click(
            fn=analyze_movie_review,
            inputs=review_input,
            outputs=[
                sentiment_output,
                score_output
            ]
        )

    with gr.Tab("Multiple Reviews Analysis"):

        multiple_input = gr.Textbox(
            label="Enter Multiple Reviews",
            placeholder="Enter each review on a new line...",
            lines=8
        )

        multiple_btn = gr.Button(
            "Analyze All Reviews",
            variant="primary"
        )

        results_output = gr.Dataframe(
            label="Analysis Results"
        )

        chart_output = gr.Plot(
            label="Sentiment Dashboard"
        )

        multiple_btn.click(
            fn=analyze_multiple_reviews,
            inputs=multiple_input,
            outputs=[
                results_output,
                chart_output
            ]
        )

    gr.Markdown(
        """
        ---
        **Developed using:** Python | NLP | NLTK | VADER | Gradio
        """
    )

if __name__ == "__main__":
    demo.launch()