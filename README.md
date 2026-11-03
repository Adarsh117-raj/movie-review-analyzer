# Movie Review Analyzer (NLP Mini Project)

Sentiment analysis of movie reviews using **NLTK VADER**, with a **Plotly** dashboard and a **Gradio** web app.
Each review is classified as Positive, Neutral or Negative from its compound score (>= 0.05 Positive, <= -0.05 Negative).

## Features
- Load reviews from `movie_reviews.csv` (column `review`) or use built-in samples
- Text cleaning and VADER scoring (pos / neu / neg / compound)
- Dashboard: KPI cards, donut chart, bar chart, score histogram, results table
- Export results to CSV and Excel
- Gradio app with a single-review tab and a multiple-reviews tab

## Setup
```bash
pip install -r requirements.txt
python -c "import nltk; nltk.download('vader_lexicon')"
```

## Run
- Notebook: open `Movie_Review_Analyzer.ipynb` in Jupyter and run all cells
- Web app: `python app.py` then open the local URL shown

## Screenshots
![Input](screenshots/multiple-reviews-input.jpg)
![Results](screenshots/analysis-results.jpg)

## Project report
See `docs/Movie_Review_Analyzer_Project_Report.docx`.

**Tech:** Python, Pandas, NumPy, NLTK (VADER), Plotly, Gradio
