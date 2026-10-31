import time

import pandas as pd
import plotly.express as px
import streamlit as st

from auth import MIN_PASSWORD_LENGTH, authenticate, create_user
from chat_memory import add_turn, format_history_for_prompt
from chat_store import (
    add_message,
    create_session,
    delete_session,
    list_sessions,
    load_messages,
)
from db_manager import (
    build_database,
    get_schema_text,
    make_table_name,
    read_data_file,
    run_query,
)
from llm_helper import explain_results, generate_sql, has_api_key
from sql_safety import validate_read_only_sql
from auto_insights import ai_summary, generate_insights, insights_to_text, suggest_questions
from semantic_layer import (
    apply_definitions,
    build_semantic_layer,
    check_metrics,
    clean_metrics,
    definitions_from_json,
    layer_to_json,
    semantic_schema_text,
)
from day3_features import (
    build_report,
    date_columns,
    detect_anomalies,
    email_is_configured,
    make_forecast,
    numeric_columns,
    send_email_report,
)


st.set_page_config(
    page_title="DataChat AI",
    page_icon="📊",
    layout="wide",
)

MAX_FAILED_LOGINS = 5
LOCKOUT_SECONDS = 60


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


def email_form(key: str, default_subject: str, build_text, dataframe=None, attachment_name="datachat_report.csv"):
    """Recipient box + Send button used everywhere in the app.

    build_text is a function; it is called only when the user presses Send.
    Several recipients can be typed, separated by commas.
    """
    if not email_is_configured():
        st.info(
            "Email is not set up yet. Fill the SMTP_ lines in your .env file "
            "(Gmail address + Gmail App Password) and restart the app."
        )
        return

    recipients = st.text_input(
        "Send to (several addresses: separate with commas)",
        value=st.session_state.get("email_default_recipient", ""),
        key=f"{key}_to",
        placeholder="name@example.com",
    )
    subject = st.text_input("Subject", value=default_subject, key=f"{key}_subject")

    if st.button("📧 Send email", key=f"{key}_send"):
        try:
            send_email_report(
                recipients,
                subject.strip() or default_subject,
                build_text(),
                dataframe,
                attachment_name,
            )
            st.session_state["email_default_recipient"] = recipients.strip()
            st.success("Email sent.")
        except Exception as error:
            st.error(f"Email failed: {error}")


# ============================================================
# LOGIN / SIGN UP
# ============================================================

def show_login_page():
    """Login and sign-up forms. Runs until the user is logged in."""

    st.title("📊 DataChat AI")
    st.caption("Log in to chat with your data. Your chats are saved to your account.")

    login_tab, signup_tab = st.tabs(["Log in", "Sign up"])

    with login_tab:
        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Log in")

        if submitted:
            locked_until = st.session_state.get("locked_until", 0)

            if time.time() < locked_until:
                st.error(
                    "Too many wrong attempts. "
                    f"Try again in {int(locked_until - time.time())} seconds."
                )
            else:
                user = authenticate(username, password)

                if user:
                    st.session_state["user"] = user
                    st.session_state["failed_logins"] = 0
                    st.rerun()

                failed = st.session_state.get("failed_logins", 0) + 1
                st.session_state["failed_logins"] = failed

                if failed >= MAX_FAILED_LOGINS:
                    st.session_state["locked_until"] = time.time() + LOCKOUT_SECONDS
                    st.session_state["failed_logins"] = 0

                st.error("Wrong username or password.")

    with signup_tab:
        with st.form("signup_form"):
            new_username = st.text_input("Choose a username")
            new_password = st.text_input(
                "Choose a password",
                type="password",
                help=f"At least {MIN_PASSWORD_LENGTH} characters.",
            )
            confirm_password = st.text_input("Repeat password", type="password")
            signup_submitted = st.form_submit_button("Create account")

        if signup_submitted:
            if new_password != confirm_password:
                st.error("The two passwords do not match.")
            else:
                created, message = create_user(new_username, new_password)

                if created:
                    st.success(message)
                else:
                    st.error(message)


if "user" not in st.session_state:
    show_login_page()
    st.stop()

user = st.session_state["user"]
user_id = user["id"]

st.title("📊 DataChat AI")
st.caption(
    "Upload CSV or Excel files and chat with your data in plain English. "
    "You can ask follow-up questions and combine multiple files."
)


# ============================================================
# CHAT SESSION STATE
# ============================================================

st.session_state.setdefault("history", [])
st.session_state.setdefault("session_id", None)


def start_new_chat():
    st.session_state["session_id"] = None
    st.session_state["history"] = []


def open_chat(session_id: int):
    st.session_state["session_id"] = session_id
    st.session_state["history"] = load_messages(user_id, session_id)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.write(f"Logged in as **{user['username']}**")

    if st.button("Log out"):
        st.session_state.clear()
        st.rerun()

    st.header("Your data")

    uploaded_files = st.file_uploader(
        "Upload CSV or Excel files",
        type=["csv", "xlsx", "xls"],
        accept_multiple_files=True,
    )

    st.header("Your chats")

    if st.button("➕ New chat"):
        start_new_chat()
        st.rerun()

    for saved in list_sessions(user_id):

        is_current = saved["id"] == st.session_state["session_id"]
        open_col, delete_col = st.columns([5, 1])

        label = ("▶ " if is_current else "") + saved["title"]

        if open_col.button(label, key=f"open_{saved['id']}"):
            open_chat(saved["id"])
            st.rerun()

        if delete_col.button("🗑", key=f"del_{saved['id']}"):
            delete_session(user_id, saved["id"])

            if is_current:
                start_new_chat()

            st.rerun()


# ============================================================
# LOAD FILES (only when the uploaded files change)
# ============================================================

files_signature = tuple(
    (file.name, file.size)
    for file in uploaded_files
)

previous_signature = st.session_state.get("files_signature")

if previous_signature != files_signature:

    tables = {}
    used_names = set()
    load_errors = []

    for uploaded_file in uploaded_files:

        try:

            for label, dataframe in read_data_file(uploaded_file).items():

                if dataframe.empty:
                    load_errors.append(f"{label}: file has no rows.")
                    continue

                table_name = make_table_name(label, used_names)
                tables[table_name] = dataframe

        except Exception as error:

            load_errors.append(f"{uploaded_file.name}: {error}")

    connection = build_database(tables) if tables else None

    st.session_state["files_signature"] = files_signature
    st.session_state["tables"] = tables
    st.session_state["load_errors"] = load_errors
    st.session_state["connection"] = connection
    st.session_state["schema_text"] = (
        get_schema_text(connection, list(tables.keys())) if tables else ""
    )

    # Day 2: semantic layer + automatic insights
    layer = build_semantic_layer(tables) if tables else None
    st.session_state["semantic_layer"] = layer
    st.session_state["layer_version"] = st.session_state.get("layer_version", 0) + 1
    st.session_state["insights"] = generate_insights(tables, layer) if tables else []
    st.session_state.pop("insight_summary_text", None)

    # Different data -> old SQL no longer makes sense -> start a fresh chat.
    if previous_signature:
        start_new_chat()


tables = st.session_state["tables"]
connection = st.session_state["connection"]
layer = st.session_state.get("semantic_layer")

# The AI reads the semantic layer (column meaning, joins, metrics),
# not just the plain column list.
schema_text = (
    semantic_schema_text(layer, connection)
    if layer and connection is not None
    else st.session_state["schema_text"]
)


# ============================================================
# FILE MESSAGES, TABLE INFO, PREVIEW
# ============================================================

for message in st.session_state["load_errors"]:
    st.warning(message)

if not uploaded_files:
    st.info(
        "Upload one or more CSV / Excel files from the sidebar to ask new "
        "questions. You can still read your saved chats."
    )
elif not tables:
    st.error("None of the uploaded files could be loaded.")

if tables:

    with st.sidebar:

        st.subheader("Loaded tables")

        for table_name, dataframe in tables.items():

            st.write(
                f"**{table_name}** — "
                f"{len(dataframe)} rows, "
                f"{len(dataframe.columns)} columns"
            )

    with st.expander("Preview uploaded data"):

        for table_name, dataframe in tables.items():

            st.markdown(f"**{table_name}**")
            st.dataframe(dataframe.head(20), width="stretch")

    with st.expander("View schema sent to the AI (with semantic layer)"):

        st.code(schema_text, language="text")


# ============================================================
# DAY 2: AUTO INSIGHTS + SEMANTIC LAYER
# ============================================================

if tables and layer:

    with st.expander("✨ Auto insights", expanded=True):

        insights = st.session_state["insights"]

        for table_name in tables:
            st.markdown(f"**{table_name}**")

            for item in insights:
                if item["table"] == table_name:
                    icon = "⚠️" if item["level"] == "warning" else "•"
                    st.write(f"{icon} {item['text']}")

        if has_api_key():
            if st.button("📝 Write AI summary", key="insight_summary_button"):
                try:
                    with st.spinner("Writing summary..."):
                        st.session_state["insight_summary_text"] = ai_summary(insights)
                except Exception as error:
                    st.error(f"Summary failed: {error}")

        if st.session_state.get("insight_summary_text"):
            st.info(st.session_state["insight_summary_text"])

        if st.checkbox("📧 Email these insights", key="email_insights_toggle"):
            email_form(
                "email_insights",
                "DataChat AI - Auto insights",
                lambda: "Auto insights\n=============\n\n" + insights_to_text(insights),
            )

        st.markdown("**Try asking:**")

        for number, suggestion in enumerate(suggest_questions(tables, layer)):
            if st.button(suggestion, key=f"suggest_{number}"):
                st.session_state["pending_question"] = suggestion
                st.rerun()

    with st.expander("🧠 Semantic layer — teach the AI your business words"):

        st.caption(
            "Define metrics like **revenue** once, add column descriptions and "
            "check table links. The AI then uses your exact definitions."
        )

        version = st.session_state["layer_version"]
        metrics_tab, columns_tab, links_tab, file_tab = st.tabs(
            ["📐 Metrics", "📝 Descriptions", "🔗 Table links", "💾 Save / load"]
        )

        with metrics_tab:

            metric_rows = pd.DataFrame(
                [
                    {
                        "name": m["name"],
                        "table": m["table"],
                        "formula": m["expression"],
                        "synonyms": ", ".join(m["synonyms"]),
                        "description": m["description"],
                    }
                    for m in layer["metrics"]
                ],
                columns=["name", "table", "formula", "synonyms", "description"],
            )

            edited_metrics = st.data_editor(
                metric_rows,
                num_rows="dynamic",
                width="stretch",
                key=f"metrics_editor_{version}",
                column_config={
                    "table": st.column_config.SelectboxColumn(
                        "table", options=list(tables.keys()), required=True
                    ),
                    "formula": st.column_config.TextColumn(
                        "formula", help="Example: SUM(total_amount)"
                    ),
                },
            )

            if st.button("Save metrics", key=f"save_metrics_{version}"):
                new_metrics = clean_metrics(
                    [
                        {
                            "name": row["name"],
                            "table": row["table"],
                            "expression": row["formula"],
                            "synonyms": row["synonyms"],
                            "description": row["description"],
                        }
                        for row in edited_metrics.fillna("").to_dict("records")
                    ],
                    layer,
                )

                problems = check_metrics({**layer, "metrics": new_metrics}, connection)

                if problems:
                    st.error("Not saved:\n\n" + "\n\n".join(problems))
                else:
                    layer["metrics"] = new_metrics
                    st.session_state["layer_version"] += 1
                    st.rerun()

        with columns_tab:

            chosen_table = st.selectbox(
                "Table", list(tables.keys()), key=f"desc_table_{version}"
            )
            table_info = layer["tables"][chosen_table]

            table_description = st.text_input(
                "What is this table about?",
                value=table_info["description"],
                key=f"desc_text_{version}_{chosen_table}",
            )

            column_rows = pd.DataFrame(
                [
                    {
                        "column": name,
                        "type": info["role"],
                        "description": info["description"],
                        "synonyms": ", ".join(info["synonyms"]),
                    }
                    for name, info in table_info["columns"].items()
                ]
            )

            edited_columns = st.data_editor(
                column_rows,
                disabled=["column", "type"],
                width="stretch",
                key=f"columns_editor_{version}_{chosen_table}",
            )

            if st.button("Save descriptions", key=f"save_desc_{version}_{chosen_table}"):
                apply_definitions(
                    layer,
                    {
                        "tables": {
                            chosen_table: {
                                "description": table_description,
                                "columns": {
                                    row["column"]: {
                                        "description": row["description"],
                                        "synonyms": row["synonyms"],
                                    }
                                    for row in edited_columns.fillna("").to_dict("records")
                                },
                            }
                        }
                    },
                )
                st.success("Saved.")

        with links_tab:

            st.caption(
                "Found automatically. Edit if a link is wrong or missing "
                "(\"many\" side → \"one\" side)."
            )

            link_rows = pd.DataFrame(
                layer["relationships"],
                columns=["from_table", "from_column", "to_table", "to_column"],
            )

            edited_links = st.data_editor(
                link_rows,
                num_rows="dynamic",
                width="stretch",
                key=f"links_editor_{version}",
            )

            if st.button("Save links", key=f"save_links_{version}"):
                wanted = edited_links.dropna().to_dict("records")
                apply_definitions(layer, {"relationships": wanted})

                if len(layer["relationships"]) < len(wanted):
                    st.warning("Some links were ignored because the table or column does not exist.")
                else:
                    st.success("Saved.")

        with file_tab:

            st.download_button(
                "⬇️ Download definitions (JSON)",
                data=layer_to_json(layer),
                file_name="datachat_semantic_layer.json",
                mime="application/json",
                key=f"download_layer_{version}",
            )

            definition_file = st.file_uploader(
                "Load definitions saved earlier",
                type=["json"],
                key=f"upload_layer_{version}",
            )

            if definition_file is not None and st.button("Apply file", key=f"apply_layer_{version}"):
                try:
                    definitions = definitions_from_json(
                        definition_file.getvalue().decode("utf-8")
                    )
                    trial = apply_definitions(build_semantic_layer(tables), definitions)
                    problems = check_metrics(trial, connection)

                    if problems:
                        st.error("File not applied:\n\n" + "\n\n".join(problems))
                    else:
                        st.session_state["semantic_layer"] = trial
                        st.session_state["layer_version"] += 1
                        st.rerun()

                except Exception as error:
                    st.error(f"Could not read the file: {error}")


# ============================================================
# DAY 3: ANOMALY DETECTION / FORECASTING / EMAIL REPORTS
# ============================================================

if tables:
    with st.expander("📈 Day 3 — Anomaly Detection, Forecasting & Email Reports"):
        st.caption(
            "Detect unusual values, create a simple trend forecast, and "
            "optionally email the generated report."
        )

        day3_table = st.selectbox(
            "Select a table",
            options=list(tables.keys()),
            key="day3_table",
        )
        day3_df = tables[day3_table]

        anomaly_result = None
        forecast_result = None

        anomaly_tab, forecast_tab, email_tab = st.tabs(
            ["🔎 Anomaly Detection", "🔮 Forecasting", "📧 Email Report"]
        )

        with anomaly_tab:
            anomaly_numbers = numeric_columns(day3_df)

            if not anomaly_numbers:
                st.info("This table has no numeric columns for anomaly detection.")
            else:
                anomaly_value = st.selectbox(
                    "Numeric column",
                    anomaly_numbers,
                    key="day3_anomaly_value",
                )
                anomaly_threshold = st.slider(
                    "Anomaly threshold (z-score)",
                    min_value=2.0,
                    max_value=5.0,
                    value=3.0,
                    step=0.5,
                    key="day3_anomaly_threshold",
                    help="Lower values flag more rows as unusual.",
                )

                if st.button("Detect anomalies", key="day3_detect"):
                    try:
                        anomaly_result = detect_anomalies(
                            day3_df,
                            anomaly_value,
                            anomaly_threshold,
                        )
                        flagged = anomaly_result[anomaly_result["_is_anomaly"]].copy()
                        st.session_state["day3_anomaly_result"] = flagged

                        st.success(
                            f"Found {len(flagged)} anomalous row(s) "
                            f"out of {len(anomaly_result)}."
                        )

                        if flagged.empty:
                            st.info("No anomalies were detected at this threshold.")
                        else:
                            st.dataframe(
                                flagged.drop(columns=["_is_anomaly"]),
                                width="stretch",
                            )
                            st.download_button(
                                "Download anomalies CSV",
                                data=flagged.drop(columns=["_is_anomaly"]).to_csv(
                                    index=False
                                ),
                                file_name="datachat_anomalies.csv",
                                mime="text/csv",
                                key="download_anomalies",
                            )
                    except Exception as error:
                        st.error(f"Anomaly detection failed: {error}")

                if "day3_anomaly_result" in st.session_state:
                    saved_anomalies = st.session_state["day3_anomaly_result"]
                    if not saved_anomalies.empty:
                        st.caption(
                            "Last detected anomalies are available for the Email Report tab."
                        )

        with forecast_tab:
            forecast_dates = date_columns(day3_df)
            forecast_numbers = numeric_columns(day3_df)

            if not forecast_dates or not forecast_numbers:
                st.info(
                    "Forecasting needs at least one date column and one numeric column."
                )
            else:
                forecast_date = st.selectbox(
                    "Date column",
                    forecast_dates,
                    key="day3_forecast_date",
                )
                forecast_value = st.selectbox(
                    "Value column",
                    forecast_numbers,
                    key="day3_forecast_value",
                )
                forecast_periods = st.number_input(
                    "Future periods",
                    min_value=1,
                    max_value=365,
                    value=7,
                    step=1,
                    key="day3_forecast_periods",
                )

                if st.button("Create forecast", key="day3_forecast"):
                    try:
                        forecast_result = make_forecast(
                            day3_df,
                            forecast_date,
                            forecast_value,
                            int(forecast_periods),
                        )
                        st.session_state["day3_forecast_result"] = forecast_result

                        st.success(
                            f"Forecast created for the next {forecast_periods} period(s)."
                        )
                        st.dataframe(forecast_result, width="stretch")

                        figure = px.line(
                            forecast_result,
                            x=forecast_date,
                            y=f"{forecast_value}_forecast",
                            markers=True,
                        )
                        st.plotly_chart(
                            figure,
                            width="stretch",
                            key="day3_forecast_chart",
                        )

                        st.download_button(
                            "Download forecast CSV",
                            data=forecast_result.to_csv(index=False),
                            file_name="datachat_forecast.csv",
                            mime="text/csv",
                            key="download_forecast",
                        )
                    except Exception as error:
                        st.error(f"Forecasting failed: {error}")

        with email_tab:
            report_options = []
            if "day3_anomaly_result" in st.session_state:
                report_options.append("Latest anomaly report")
            if "day3_forecast_result" in st.session_state:
                report_options.append("Latest forecast report")

            if st.checkbox("🔧 Send a test email (checks your Gmail settings)", key="email_test_toggle"):
                email_form(
                    "email_test",
                    "DataChat AI - test email",
                    lambda: "Hello from DataChat AI. Email sending is working.",
                )

            if not report_options:
                st.info(
                    "First run anomaly detection or forecasting, then you can "
                    "email that report."
                )
            else:
                report_type = st.selectbox(
                    "Report to send",
                    report_options,
                    key="day3_email_report_type",
                )
                recipient = st.text_input(
                    "Recipient email (several: separate with commas)",
                    key="day3_recipient",
                    placeholder="name@example.com",
                )
                subject = st.text_input(
                    "Email subject",
                    value="DataChat AI — Day 3 Report",
                    key="day3_email_subject",
                )

                if st.button("Send email report", key="day3_send_email"):
                    if not recipient.strip():
                        st.error("Enter a recipient email address.")
                    else:
                        try:
                            if report_type == "Latest anomaly report":
                                report_df = st.session_state["day3_anomaly_result"]
                            else:
                                report_df = st.session_state["day3_forecast_result"]

                            report_text = build_report(
                                report_type,
                                report_df,
                            )
                            send_email_report(
                                recipient.strip(),
                                subject.strip() or "DataChat AI Report",
                                report_text,
                                report_df,
                            )
                            st.success("Email report sent successfully.")
                        except Exception as error:
                            st.error(f"Email sending failed: {error}")


# ============================================================
# USER QUESTION
# ============================================================

error_message = None
needs_rerun = False

typed_question = st.chat_input(
    "Ask a question about your data...",
    disabled=not tables,
)

# A click on a suggested question (Day 2) works like typing it.
question = typed_question or st.session_state.pop("pending_question", None)


# ============================================================
# AI ANALYSIS
# ============================================================

if question and tables:

    if not has_api_key():

        error_message = (
            "OpenAI API key is missing. "
            "Add OPENAI_API_KEY to your .env file."
        )

    else:

        try:

            with st.spinner("Creating and running SQL..."):

                history_text = format_history_for_prompt(
                    st.session_state["history"]
                )

                generated_sql = generate_sql(
                    question,
                    schema_text,
                    history_text,
                )

                is_safe, checked_sql = validate_read_only_sql(generated_sql)

                if not is_safe:
                    raise ValueError(f"Unsafe SQL was blocked: {checked_sql}")

                result = run_query(connection, checked_sql)

                explanation = explain_results(question, result)

            # Save to the database first, then show it.
            if st.session_state["session_id"] is None:
                st.session_state["session_id"] = create_session(
                    user_id,
                    question,
                )

            add_message(
                user_id,
                st.session_state["session_id"],
                question,
                checked_sql,
                explanation,
                result,
            )

            add_turn(
                st.session_state["history"],
                question,
                checked_sql,
                explanation,
                result,
            )

            needs_rerun = True

        except Exception as error:

            error_message = f"Analysis failed: {error}"

# Rerun so the new chat title appears in the sidebar list.
if needs_rerun:
    st.rerun()


# ============================================================
# SHOW CONVERSATION
# ============================================================

for index, turn in enumerate(st.session_state["history"]):

    with st.chat_message("user"):
        st.write(turn["question"])

    with st.chat_message("assistant"):

        st.write(turn["explanation"])

        with st.expander("Generated SQL"):
            st.code(turn["sql"], language="sql")

        st.dataframe(turn["result"], width="stretch")

        st.download_button(
            "Download result as CSV",
            data=turn["result"].to_csv(index=False),
            file_name=f"datachat_result_{index + 1}.csv",
            mime="text/csv",
            key=f"download_{index}",
        )

        show_chart(turn["result"], key_prefix=f"turn{index}")

        if st.checkbox("📧 Email this answer", key=f"email_toggle_{index}"):
            email_form(
                f"email_turn{index}",
                f"DataChat AI - {turn['question'][:60]}",
                lambda turn=turn: (
                    f"Question: {turn['question']}\n\n"
                    f"Answer:\n{turn['explanation']}\n\n"
                    f"SQL used:\n{turn['sql']}\n\n"
                    + build_report("Result", turn["result"])
                ),
                turn["result"],
                f"datachat_result_{index + 1}.csv",
            )


# ============================================================
# ERROR MESSAGE
# ============================================================

if error_message:
    st.error(error_message)
