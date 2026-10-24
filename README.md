# DataChat AI

Chat with your CSV / Excel data in plain English. The app converts your question
into SQL using the Gemini API, runs it safely on DuckDB, and shows the answer,
SQL, result table and charts.

## Features
- Upload multiple CSV / Excel files (each Excel sheet becomes a table)
- Ask questions that combine files (JOINs)
- Chat history with follow-up questions ("now show it city-wise")
- Read-only SQL safety check (blocks DROP, DELETE, INSERT, etc.)
- Auto-generated explanation, downloadable result CSV, Bar / Line / Pie charts
- Automatic retry when the AI model is busy (503)
- Unit tests with pytest

## Tech stack
Python, Streamlit, DuckDB, Pandas, Plotly, Google Gemini API

## Project structure
```
app.py            Streamlit UI and chat flow
db_manager.py     File reading, table creation, schema text
chat_memory.py    Chat history for follow-up questions
llm_helper.py     Gemini calls (SQL + explanation) with retry
sql_safety.py     Read-only SQL validation
tests/            pytest unit tests
sample_data/      Example CSV
docs/             Architecture diagram
```

## Setup
```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```
Copy `.env.example` to `.env` and add your key from https://aistudio.google.com/apikey

```
GEMINI_API_KEY=your_key_here
GEMINI_MODEL=gemini-3.8-flash
```

## Run
```
streamlit run app.py
```

## Run tests
```
pytest
```

## Security
- `.env` is git-ignored; never upload your API key to GitHub.
- Only a single SELECT / WITH query is allowed; everything else is blocked.

## Future work
User login, saved chat sessions, database connections (MySQL/PostgreSQL), deployment.