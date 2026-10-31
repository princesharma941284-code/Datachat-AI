# DataChat AI - Day 2 + Day 4 (Day 3 code par bana hai)

## Day 2 - Semantic layer + Auto insights
Nayi files: `semantic_layer.py`, `auto_insights.py`. `app.py` aur `llm_helper.py` mein chhote badlav.

**Semantic layer** AI ko data ka *matlab* batati hai, sirf column ke naam nahi:
- har column ka role (id / date / category / number) aur category ki saari values
- tables ka link (orders.customer_id = customers.customer_id) - apne aap milta hai
- business metrics, jaise `revenue = SUM(total_amount)` (synonyms: sales, income)
- column/table descriptions
App mein: **🧠 Semantic layer** expander (Metrics, Descriptions, Table links, Save/Load JSON).
Metric save karne se pehle formula DuckDB par test hota hai (same safety filter se).

**Auto insights** (bina AI call ke, free): upload par data quality (missing/duplicates),
outliers, dominant category, month-wise trend, strong correlation, aur clickable
suggested questions. Optional "Write AI summary" button.

## Day 4 - Benchmark, tests, GitHub Actions
- `benchmark/` : 50 sawal + fixed sample database + runner + graph
- `tests/` : naye tests; `.github/workflows/` : ci.yml (auto) aur benchmark.yml (manual)

## Commands (project folder ke andar)
    pip install -r requirements.txt
    streamlit run app.py
    python -m pytest -q                                   # saare tests
    python -m benchmark.run_benchmark --mock              # API ke bina self-check
    python -m benchmark.run_benchmark --limit 5           # real AI, 5 sawal
    python -m benchmark.run_benchmark                     # real AI, 50 sawal (semantic layer ke saath)
    python -m benchmark.run_benchmark --compare           # plain vs semantic, before/after graph

`--compare` Day 2 ka saboot hai: `benchmark/results/comparison_chart.png` report/PPT mein lagao.
Real run ke liye `.env` mein `OPENAI_API_KEY` chahiye. `--compare` mein 100 API calls lagti hain.

## Email (Gmail) - ab har jagah se
- Har chat jawab ke neeche **📧 Email this answer** (result CSV attachment ke saath)
- **✨ Auto insights** mein **📧 Email these insights**
- Day 3 ke Email Report tab mein **🔧 Send a test email** (Gmail settings check karne ke liye)
- Ek saath 10 tak recipients (comma se alag)
- `.env` mein SMTP_HOST, SMTP_PORT, SMTP_USERNAME, SMTP_PASSWORD (Gmail **App Password**), SMTP_FROM, SMTP_USE_TLS
- `.env` badalne ke baad app dobara start karo (`Ctrl+C`, phir `streamlit run app.py`)
