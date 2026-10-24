import pandas as pd
import plotly.express as px
import streamlit as st

from chat_memory import add_turn, format_history_for_prompt
from db_manager import build_database, get_schema_text, make_table_name, read_data_file
from llm_helper import explain_results, generate_sql, has_api_key
from sql_safety import validate_read_only_sql


st.set_page_config(
    page_title="DataChat AI",
    page_icon="📊",
    layout="wide",
)

st.title("📊 DataChat AI")
st.caption(
    "Upload CSV or Excel files and chat with your data in plain English. "
    "You can ask follow-up questions and combine multiple files."
)


def show_chart(result: pd.DataFrame, key_prefix: str):
    """Create a chart when the result has usable numeric data."""

    numeric_columns = result.select_dtypes(include="number").columns.tolist()

    if len(result.columns) < 2 or not numeric_columns:
        return

    st.markdown("**Chart**")

    left, right, third = st.columns(3)

    x_column = left.selectbox(
        "X-axis",
        options=result.columns.tolist(),
        key=f"{key_prefix}_x",
    )

    y_column = right.selectbox(
        "Y-axis",
        options=numeric_columns,
        key=f"{key_prefix}_y",
    )

    chart_type = third.selectbox(
        "Chart type",
        options=["Bar", "Line", "Pie"],
        key=f"{key_prefix}_type",
    )

    chart_data = result.dropna(subset=[x_column, y_column])

    if chart_data.empty:
        st.info("No chart can be created from empty values.")
        return

    if chart_type == "Bar":
        figure = px.bar(
            chart_data,
            x=x_column,
            y=y_column,
        )

    elif chart_type == "Line":
        figure = px.line(
            chart_data,
            x=x_column,
            y=y_column,
            markers=True,
        )

    else:
        figure = px.pie(
            chart_data,
            names=x_column,
            values=y_column,
        )

    st.plotly_chart(
        figure,
        width="stretch",
        key=f"{key_prefix}_chart",
    )


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("Your data")

    uploaded_files = st.file_uploader(
        "Upload CSV or Excel files",
        type=["csv", "xlsx", "xls"],
        accept_multiple_files=True,
    )

    if st.button("Clear chat history"):
        st.session_state["history"] = []
        st.rerun()


# ============================================================
# FILE CHECK
# ============================================================

if not uploaded_files:
    st.info(
        "Upload one or more CSV / Excel files from the sidebar to begin."
    )
    st.stop()


# ============================================================
# LOAD FILES
# ============================================================

files_signature = tuple(
    (file.name, file.size)
    for file in uploaded_files
)

if st.session_state.get("files_signature") != files_signature:

    tables = {}
    used_names = set()
    load_errors = []

    for uploaded_file in uploaded_files:

        try:

            for label, dataframe in read_data_file(uploaded_file).items():

                if dataframe.empty:
                    load_errors.append(
                        f"{label}: file has no rows."
                    )
                    continue

                table_name = make_table_name(
                    label,
                    used_names,
                )

                tables[table_name] = dataframe

        except Exception as error:

            load_errors.append(
                f"{uploaded_file.name}: {error}"
            )

    st.session_state["files_signature"] = files_signature
    st.session_state["tables"] = tables
    st.session_state["load_errors"] = load_errors
    st.session_state["history"] = []


tables = st.session_state["tables"]


# ============================================================
# FILE ERRORS
# ============================================================

for message in st.session_state["load_errors"]:
    st.warning(message)


if not tables:
    st.error(
        "None of the uploaded files could be loaded."
    )
    st.stop()


# ============================================================
# DATABASE
# ============================================================

connection = build_database(tables)

schema_text = get_schema_text(
    connection,
    list(tables.keys()),
)


# ============================================================
# SIDEBAR TABLE INFORMATION
# ============================================================

with st.sidebar:

    st.subheader("Loaded tables")

    for table_name, dataframe in tables.items():

        st.write(
            f"**{table_name}** — "
            f"{len(dataframe)} rows, "
            f"{len(dataframe.columns)} columns"
        )


# ============================================================
# DATA PREVIEW
# ============================================================

with st.expander("Preview uploaded data"):

    for table_name, dataframe in tables.items():

        st.markdown(f"**{table_name}**")

        st.dataframe(
            dataframe.head(20),
            width="stretch",
        )


# ============================================================
# DATABASE SCHEMA
# ============================================================

with st.expander("View database schema"):

    st.code(
        schema_text,
        language="text",
    )


# ============================================================
# USER QUESTION
# ============================================================

error_message = None

question = st.chat_input(
    "Ask a question about your data..."
)


# ============================================================
# AI ANALYSIS
# ============================================================

if question:

    # Check OpenAI API key
    if not has_api_key():

        error_message = (
            "OpenAI API key is missing. "
            "Add OPENAI_API_KEY to your .env file."
        )

    else:

        try:

            with st.spinner(
                "Creating and running SQL..."
            ):

                # Get previous conversation
                history_text = format_history_for_prompt(
                    st.session_state["history"]
                )

                # Generate SQL using OpenAI
                generated_sql = generate_sql(
                    question,
                    schema_text,
                    history_text,
                )

                # Check SQL safety
                is_safe, checked_sql = validate_read_only_sql(
                    generated_sql
                )

                if not is_safe:

                    raise ValueError(
                        f"Unsafe SQL was blocked: {checked_sql}"
                    )

                # Execute SQL in DuckDB
                result = connection.execute(
                    checked_sql
                ).df()

                # Generate explanation using OpenAI
                explanation = explain_results(
                    question,
                    result,
                )

            # Save conversation
            add_turn(
                st.session_state["history"],
                question,
                checked_sql,
                explanation,
                result,
            )

        except Exception as error:

            error_message = (
                f"Analysis failed: {error}"
            )


# ============================================================
# SHOW CONVERSATION
# ============================================================

for index, turn in enumerate(
    st.session_state["history"]
):

    # User message
    with st.chat_message("user"):

        st.write(
            turn["question"]
        )


    # AI response
    with st.chat_message("assistant"):

        st.write(
            turn["explanation"]
        )


        # Generated SQL
        with st.expander("Generated SQL"):

            st.code(
                turn["sql"],
                language="sql",
            )


        # Query result
        st.dataframe(
            turn["result"],
            width="stretch",
        )


        # Download result
        st.download_button(
            "Download result as CSV",

            data=turn["result"].to_csv(
                index=False
            ),

            file_name=(
                f"datachat_result_{index + 1}.csv"
            ),

            mime="text/csv",

            key=f"download_{index}",
        )


        # Chart
        show_chart(
            turn["result"],
            key_prefix=f"turn{index}",
        )


# ============================================================
# ERROR MESSAGE
# ============================================================

if error_message:

    st.error(
        error_message
    )