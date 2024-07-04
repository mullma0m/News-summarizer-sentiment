# News Summarizer and Sentiment Analyzer

This project provides a Streamlit application that fetches news articles based on a ticker symbol, summarizes the articles, and analyzes their sentiment. It integrates with FinnHub and NewsAPI to fetch news articles and uses pre-trained models from Hugging Face for summarization and sentiment analysis.

## Features

- Fetches news articles based on a ticker symbol from FinnHub and NewsAPI.
- Summarizes news articles using pre-trained models from Hugging Face.
- Analyzes the sentiment of the summarized news articles.
- Displays the summarized articles with sentiment analysis in a user-friendly interface.

## Installation

1. **Clone the repository**:
    ```bash
    git clone https://github.com/mullma0m/News-summarizer-sentiment.git
    cd News-summarizer-sentiment
    ```

2. **Create and activate a virtual environment**:
    ```bash
    python -m venv env
    source env/bin/activate   # On Windows, use `env\Scripts\activate`
    ```

3. **Install the required packages**:
    ```bash
    pip install -r requirements.txt
    ```

4. **Obtain API Keys**:
    - **FinnHub**: Sign up for a free API key at [FinnHub](https://finnhub.io/).
    - **NewsAPI**: Sign up for a free API key at [NewsAPI](https://newsapi.org/).

5. **Create a `.env` file** in the project directory and add your API keys:
    ```plaintext
    FINNHUB_API_KEY=your_finnhub_api_key
    NEWSAPI_API_KEY=your_newsapi_api_key
    ```

6. **Ensure the following directories and files are present**:
    - `streamlit_app.py` (your Streamlit application)
    - `model.ipynb` (your Jupyter notebook for creating the best model)
    - `group2_fin_model/` (the folder containing your trained model)
    - `.gitignore` (to exclude unnecessary files and folders)
    - `README.md` (description of your project)
    - `.env` (containing your API keys)
    - `requirements.txt` (to list all dependencies required for your project)

## Usage

1. **Run the Streamlit application**:
    ```bash
    streamlit run streamlit_app.py
    ```

2. **Open your web browser** and navigate to `http://localhost:8501`.

3. **Enter a ticker symbol** (e.g., AAPL, LCID, BTC, ETH) and specify the date range to fetch news articles.

4. **Choose summarization and sentiment analysis models** from the dropdowns.

5. **Click 'Get News'** to fetch, summarize, and analyze sentiment of the news articles.

## Deployed Application

You can also use the deployed application on Hugging Face Spaces:

[News Summarizer and Sentiment Analyzer](https://huggingface.co/spaces/Mulla88/News-summarizer-sentiment)

## Repository Structure

- `streamlit_app.py`: Main Streamlit application script.
- `model.ipynb`: Jupyter notebook used to create and train the sentiment analysis model.
- `group2_fin_model/`: Directory containing the saved model.
- `.gitignore`: Specifies files and directories to be ignored by Git.
- `README.md`: Description and instructions for the project.
- `.env`: Environment file containing API keys (not included in the repository).
- `requirements.txt`: List of dependencies required for the project.


