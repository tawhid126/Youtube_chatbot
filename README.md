# YouTube Chatbot

This project is a simple YouTube Q&A chatbot built with Streamlit, LangChain, Google Gemini, and FAISS. It lets you ask natural-language questions about the content of a YouTube video and get answers based on the video transcript.

## How It Works

- Takes a YouTube video **ID** (or URL that can be converted to an ID).
- Fetches the English transcript using `youtube-transcript-api`.
- Splits the transcript into chunks and creates embeddings using Google Gemini.
- Stores embeddings in a FAISS vector store for fast similarity search.
- For each user question, retrieves the most relevant chunks and sends them, along with the question, to the Gemini chat model.
- Displays the model's answer in the Streamlit UI.

## Requirements

- Python 3.10+ (recommended)
- Google API key with access to Gemini models (set in environment, e.g. `.env`).
- Dependencies (install via pip):
  - `streamlit`
  - `langchain-google-genai`
  - `langchain-community`
  - `langchain-text-splitters`
  - `youtube-transcript-api`
  - `faiss-cpu` (or another FAISS build)
  - `python-dotenv`

## Running the App

1. Create and activate a virtual environment (optional but recommended).
2. Install dependencies, for example:

   ```bash
   pip install -r requirements.txt
   ```

   or install the packages listed above manually.

3. Make sure your `.env` file contains the required Google API key (e.g. `GOOGLE_API_KEY`).
4. Start the Streamlit app:

   ```bash
   streamlit run youtube_chatboot.py
   ```

5. Open the URL shown in the terminal (usually `http://localhost:8501` or similar).
6. Enter the YouTube video ID (or URL, depending on how you use `extract_video_id`) and your question, then click **Find Ans**.

## Notes

- If a video has transcripts disabled or no English transcript, the app will not be able to answer questions about it.
- The quality of answers depends on both the transcript and the Gemini model.
