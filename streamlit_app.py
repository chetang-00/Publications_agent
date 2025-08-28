import streamlit as st
import os
import pandas as pd
from sql_agent_core import ask_llm_for_sql, run_sql_query, run_sql_query_with_params, db_path, summarize_abstracts, extract_and_count_keywords
import re

# Theme and Styling Setup
NYU_PURPLE = "#57068c"
NYU_LIGHT_PURPLE = "#a695bf"
NYU_ULTRA_VIOLET = "#8900e1"
NYU_RED = "#b41c18"
BG_COLOR = "#f9f9f9"


st.set_page_config(
    page_title="RA Publications",
    page_icon="📚",
    layout="wide"
)

# CSS with more specific selectors 
st.markdown(
    f"""
    <style>
    /* Global background */
    .stApp {{
        background-color: {BG_COLOR};
    }}
    
    /* Main content area */
    .main .block-container {{
        background-color: {BG_COLOR};
        padding-top: 2rem;
    }}
    
    /* Button styling */
    .stButton > button {{
        background-color: {NYU_PURPLE} !important;
        color: white !important;
        font-weight: 600 !important;
        border-radius: 6px !important;
        padding: 0.5rem 1rem !important;
        border: none !important;
        transition: all 0.3s ease !important;
    }}
    
    .stButton > button:hover {{
        background-color: {NYU_ULTRA_VIOLET} !important;
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2) !important;
    }}
    
    /* Form submit button */
    .stFormSubmitButton > button {{
        background-color: {NYU_PURPLE} !important;
        color: white !important;
        font-weight: 600 !important;
        border-radius: 6px !important;
        padding: 0.5rem 1rem !important;
        border: none !important;
        width: 200px !important;
    }}
    
    .stFormSubmitButton > button:hover {{
        background-color: {NYU_ULTRA_VIOLET} !important;
    }}
    
    /* Text input styling */
    .stTextInput input {{
        border: 2px solid {NYU_LIGHT_PURPLE} !important;
        padding: 0.6rem !important;
        border-radius: 6px !important;
        font-size: 1rem !important;
    }}
    
    .stTextInput input:focus {{
        border-color: {NYU_PURPLE} !important;
        box-shadow: 0 0 0 0.2rem rgba(87, 6, 140, 0.25) !important;
    }}
    
    /* Metric styling */
    .stMetric {{
        background-color: white !important;
        padding: 1rem !important;
        border-radius: 8px !important;
        border-left: 4px solid {NYU_PURPLE} !important;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1) !important;
    }}
    
    .stMetric > div > div:first-child {{
        color: {NYU_PURPLE} !important;
        font-weight: 600 !important;
    }}
    
    .stMetric > div > div:nth-child(2) {{
        color: {NYU_PURPLE} !important;
        font-size: 2rem !important;
        font-weight: 700 !important;
    }}
    
    /* Sidebar styling */
    .stSidebar {{
        background: linear-gradient(180deg, {NYU_PURPLE} 0%, {NYU_ULTRA_VIOLET} 100%) !important;
        border-right: 3px solid {NYU_PURPLE} !important;
    }}
    
    .stSidebar .stHeader {{
        font-size: 1.3rem !important;
        font-weight: 700 !important;
        color: white !important;
        margin-bottom: 1rem !important;
        text-shadow: 0 1px 2px rgba(0,0,0,0.3) !important;
    }}
    
    .stSidebar h2 {{
        color: white !important;
        font-size: 1.3rem !important;
        font-weight: 700 !important;
        margin-bottom: 1rem !important;
        text-shadow: 0 1px 2px rgba(0,0,0,0.3) !important;
    }}
    
    .stSidebar .stMarkdown {{
        color: white !important;
    }}
    
    .stSidebar .stMarkdown p {{
        color: rgba(255, 255, 255, 0.9) !important;
        font-size: 0.95rem !important;
        line-height: 1.6 !important;
    }}
    
    .stSidebar .stMarkdown strong {{
        color: white !important;
        font-weight: 600 !important;
    }}
    
    .stSidebar .stMarkdown ul {{
        color: rgba(255, 255, 255, 0.85) !important;
    }}
    
    .stSidebar .stMarkdown li {{
        color: rgba(255, 255, 255, 0.85) !important;
        margin-bottom: 0.3rem !important;
    }}
    
    /* Title styling */
    .stTitle {{
        color: {NYU_PURPLE} !important;
        font-weight: 700 !important;
        margin-bottom: 0.5rem !important;
        text-align: center !important;
    }}
    
    /* Subheader styling */
    .stSubheader {{
        color: {NYU_PURPLE} !important;
        font-weight: 600 !important;
        border-bottom: 2px solid {NYU_LIGHT_PURPLE} !important;
        padding-bottom: 0.5rem !important;
    }}
    
    /* General markdown styling */
    .stMarkdown p {{
        font-size: 1rem !important;
        line-height: 1.6 !important;
        color: #333333 !important;
    }}
    
    /* Chat message content styling */
    .stChatMessage .stMarkdown {{
        color: #333333 !important;
    }}
    
    .stChatMessage .stMarkdown p {{
        color: #333333 !important;
        font-size: 1rem !important;
        line-height: 1.6 !important;
    }}
    
    .stChatMessage .stMarkdown strong {{
        color: #333333 !important;
    }}
    
    .stChatMessage .stMarkdown li {{
        color: #333333 !important;
    }}
    
    .stChatMessage .stMarkdown ul {{
        color: #333333 !important;
    }}
    
    /* Code block styling */
    .stCodeBlock {{
        background-color: #f8f9fa !important;
        border: 1px solid {NYU_LIGHT_PURPLE} !important;
        border-radius: 6px !important;
    }}
    
    /* Data editor styling */
    .stDataFrame {{
        border: 1px solid {NYU_LIGHT_PURPLE} !important;
        border-radius: 6px !important;
    }}
    
    /* Success/error message styling */
    .stSuccess {{
        background-color: #d4edda !important;
        border-color: #c3e6cb !important;
        color: #155724 !important;
    }}
    
    /* Sidebar metric styling */
    .stSidebar .stMetric {{
        background-color: rgba(255, 255, 255, 0.15) !important;
        padding: 1rem !important;
        border-radius: 8px !important;
        border-left: 4px solid white !important;
        box-shadow: 0 2px 4px rgba(0,0,0,0.2) !important;
        backdrop-filter: blur(5px) !important;
    }}
    
    .stSidebar .stMetric > div > div:first-child {{
        color: white !important;
        font-weight: 600 !important;
    }}
    
    .stSidebar .stMetric > div > div:nth-child(2) {{
        color: white !important;
        font-size: 2rem !important;
        font-weight: 700 !important;
    }}
    
    .stError {{
        background-color: #f8d7da !important;
        border-color: #f5c6cb !important;
        color: {NYU_RED} !important;
    }}
    
    /* Sidebar error styling */
    .stSidebar .stError {{
        background-color: rgba(255, 69, 58, 0.9) !important;
        border-color: rgba(255, 69, 58, 0.8) !important;
        color: white !important;
        border-left: 4px solid white !important;
        backdrop-filter: blur(5px) !important;
    }}
    
    .stSidebar .stError > div {{
        color: white !important;
    }}
    
    .stSidebar .stError p {{
        color: white !important;
        font-weight: 500 !important;
    }}
    
    .stWarning {{
        background-color: #fff3cd !important;
        border-color: #ffeaa7 !important;
        color: #856404 !important;
    }}
    
    .stInfo {{
        background-color: #d1ecf1 !important;
        border-color: #bee5eb !important;
        color: #0c5460 !important;
    }}
    
    /* Spinner styling */
    .stSpinner > div {{
        border-color: {NYU_PURPLE} !important;
    }}
    
    /* Header with NYU branding */
    .nyu-header {{
        background: linear-gradient(135deg, {NYU_PURPLE} 0%, {NYU_LIGHT_PURPLE} 100%) !important;
        padding: 2rem 0 !important;
        margin: -1rem -1rem 2rem -1rem !important;
        text-align: center !important;
        color: white !important;
        border-radius: 0 0 15px 15px !important;
    }}
    
    .nyu-header h1 {{
        color: white !important;
        margin: 0 !important;
        font-size: 2.5rem !important;
        font-weight: 700 !important;
    }}
    
    .nyu-header p {{
        color: rgba(255, 255, 255, 0.9) !important;
        margin: 0.5rem 0 0 0 !important;
        font-size: 1.1rem !important;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

# NYU Header
st.markdown(
    f"""
    <div class="nyu-header">
        <h1>🎓 Research Publications Discovery</h1>
        <p>Ask questions about the publications database using natural language!</p>
    </div>
    """,
    unsafe_allow_html=True
)

def is_sql_query(response):
    sql_keywords = ["SELECT", "WITH", "PRAGMA", "EXPLAIN"]
    return any(response.strip().upper().startswith(k) for k in sql_keywords)

def is_followup_query(user_input):
    """
    Robustly detect if the user input is a follow-up referring to previous results.
    """
    followup_patterns = [
        r"\b(summarize|summary|summarise)\b",
        r"\b(above|previous|those|these|the\s+results|the\s+papers|the\s+list)\b",
        r"\b(add|apply|with|filter|restrict|limit|show only|only show|just show|just)\b.*",
        r"\bcontinue\b",
        r"\bnext\b",
        r"\bmore\b",
        r"\brefine\b",
        r"\bupdate\b",
        r"\bchange\b",
        r"\bremove\b",
        r"\bexclude\b",
        r"\binclude\b",
        r"\bexpand\b",
        r"\badd filter\b",
        r"\bshow papers\b",
        r"\bshow results\b",
        r"\bshow only\b",
        r"\bthe same\b",
        r"\bthe last\b",
        r"\bthe prior\b",
        r"\bthe earlier\b",
        r"\bthe current\b",
        r"\bthe output\b",
        r"\bthe table\b",
        r"\bthe dataframe\b",
        r"\bthe answer\b",
        r"\bthe query\b",
        r"\bthe selection\b",
        r"\bthe above papers\b",
        r"\bthe above results\b",
        r"\bthe above list\b",
        r"\bthe above table\b",
        r"\bthe above query\b",
        r"\bthe above answer\b",
        r"\bthe above selection\b",
        r"\bthe above output\b",
        r"\bthe above dataframe\b",
    ]
    for pattern in followup_patterns:
        if re.search(pattern, user_input, re.IGNORECASE):
            return True
    return False

def get_last_user_message(chat_history):
    for msg in reversed(chat_history):
        if msg["role"] == "user":
            return msg["content"]
    return None

def main():
    with st.sidebar:
        # NYU Logo at the top of sidebar
        if os.path.exists("nyu_logo.png"):
            st.image("nyu_logo.png", width=140)
            st.markdown("<br>", unsafe_allow_html=True)

        st.markdown("## 🕘 Chat History")
        if "saved_chats" not in st.session_state:
            st.session_state["saved_chats"] = []
        if "chat_history" not in st.session_state:
            st.session_state["chat_history"] = []
        if "current_chat_index" not in st.session_state:
            st.session_state["current_chat_index"] = 0

        if st.button("➕ New Chat"):
            if any(m["role"] == "user" for m in st.session_state["chat_history"]):
                st.session_state["saved_chats"].append(st.session_state["chat_history"].copy())
            st.session_state["chat_history"] = []
            st.session_state["current_chat_index"] = len(st.session_state["saved_chats"])
            st.rerun()

        for i, chat in enumerate(st.session_state["saved_chats"]):
            user_msgs = [m["content"] for m in chat if m["role"] == "user"]
            label = user_msgs[0][:30] + "..." if user_msgs else f"Chat {i+1}"
            if st.button(label, key=f"chat_{i}"):
                st.session_state["chat_history"] = chat.copy()
                st.session_state["current_chat_index"] = i
                st.rerun()

        if st.button("🗑️ Clear All"):
            st.session_state["saved_chats"] = []
            st.session_state["chat_history"] = []
            st.session_state["current_chat_index"] = 0
            st.rerun()
        
        st.divider()

        st.header("ℹ️ About")
        st.markdown("""
        This app allows you to query a publications database using natural language.
        
        **Available columns:**
        - authors
        - title
        - year
        - abstract
        - author_keywords
        - index_keywords
        - authors_with_affiliations
        - cluster_label
        
        **Example questions:**
        - Show me all papers about Drosophila
        - Find publications from 2020 or later
        - Which papers mention 'transcription' in the abstract?
        - List all authors and their papers
        """)
        if os.path.exists(db_path):
            import sqlite3
            conn = sqlite3.connect(db_path)
            count = pd.read_sql_query("SELECT COUNT(*) as count FROM publications", conn).iloc[0]['count']
            conn.close()
            st.metric("Total Publications", f"{count:,}")
        else:
            st.error("Database not found! Please run csv_to_sqlite.py first.")

    if "result_df" not in st.session_state:
        st.session_state["result_df"] = None
    if "last_sql" not in st.session_state:
        st.session_state["last_sql"] = None

    # Display chat history
    for msg in st.session_state["chat_history"]:
        avatar = None
        if msg["role"] == "assistant":
            avatar = "PilotGenAi.png"
        elif msg["role"] == "user":
            avatar = "User.png"
        with st.chat_message(msg["role"], avatar=avatar):
            st.markdown(msg["content"])
            # Show SQL query in an expander if present
            if "sql_query" in msg:
                with st.expander("Show/Hide SQL Query"):
                    st.code(msg["sql_query"], language="sql")
            # Show dataframes for aggregation and keyword analysis results in chat
            if msg.get("type") in ["aggregation", "keyword_analysis"] and msg.get("df") is not None:
                st.dataframe(msg["df"], use_container_width=True)

    # Prompt box at the bottom
    user_input = st.chat_input("Type your question and press Enter...")
    if user_input:
        st.session_state["chat_history"].append({"role": "user", "content": user_input})
        with st.chat_message("user", avatar="User.png"):
            st.markdown(user_input)
        with st.spinner("Thinking..."):
            # Hybrid context: explicit + chat history
            chat_history = st.session_state["chat_history"][-10:]  # last 10 turns
            explicit_context = user_input
            if is_followup_query(user_input) and st.session_state.get("last_sql"):
                last_user_msg = get_last_user_message(st.session_state["chat_history"][:-1])
                explicit_context = f"Previous SQL Query: {st.session_state['last_sql']}\nPrevious User Question: {last_user_msg}\nFollow-up: {user_input}"
            elif "last_clarify_context" in st.session_state:
                explicit_context = st.session_state["last_clarify_context"] + " " + user_input
            # else: explicit_context = user_input (already set)

            # If ask_llm_for_sql supports history, pass both; otherwise, concatenate
            try:
                intent, sql_query, clarify_message = ask_llm_for_sql(explicit_context, chat_history)
            except TypeError:
                # Fallback: concatenate chat history into context
                history_text = "\n".join([f"{m['role']}: {m['content']}" for m in chat_history])
                context = f"{history_text}\n{explicit_context}"
                intent, sql_query, clarify_message = ask_llm_for_sql(context)

            if intent == "clarify":
                st.session_state["last_clarify_context"] = user_input
                # Clear the result_df so previous results don't show below
                st.session_state["result_df"] = None
                st.session_state["chat_history"].append({"role": "assistant", "content": clarify_message})

            elif intent == "summary":
                df, error = run_sql_query(sql_query)
                if error:
                    st.session_state["chat_history"].append({"role": "assistant", "content": f"SQL Error: {error}"})
                elif len(df) > 0:
                    abstracts = df["abstract"].dropna().tolist()
                    summary = summarize_abstracts(abstracts, query_context=user_input)
                    st.session_state["chat_history"].append({"role": "assistant", "content": summary})
                    # Clear the result_df so previous results don't show below
                    st.session_state["result_df"] = None
                    if "last_clarify_context" in st.session_state:
                        del st.session_state["last_clarify_context"]
                else:
                    st.session_state["chat_history"].append({"role": "assistant", "content": "No abstracts found to summarize."})
                    # Clear the result_df so previous results don't show below
                    st.session_state["result_df"] = None

            elif intent == "keyword_analysis":
                # Parse filters from clarify_message
                author_filter = None
                year_filter = None
                
                if clarify_message:
                    filters = clarify_message.split("|")
                    for filter_str in filters:
                        if ":" in filter_str:
                            key, value = filter_str.split(":", 1)
                            if key.strip() == "author":
                                author_filter = value.strip()
                            elif key.strip() == "year":
                                year_filter = value.strip()
                
                # Extract keywords and counts
                keyword_df = extract_and_count_keywords(
                    db_path=db_path,
                    author_filter=author_filter,
                    year_filter=year_filter,
                    limit=20
                )
                
                if len(keyword_df) > 0:
                    # Format the response message
                    filter_desc = []
                    if author_filter:
                        filter_desc.append(f"author '{author_filter}'")
                    if year_filter:
                        filter_desc.append(f"year {year_filter}")
                    
                    filter_text = " for " + " and ".join(filter_desc) if filter_desc else ""
                    response_text = f"Most common research topics{filter_text}:"
                    
                    st.session_state["chat_history"].append({
                        "role": "assistant", 
                        "content": response_text,
                        "type": "keyword_analysis", 
                        "df": keyword_df
                    })
                    # Clear the result_df so previous results don't show below
                    st.session_state["result_df"] = None
                    if "last_clarify_context" in st.session_state:
                        del st.session_state["last_clarify_context"]
                else:
                    st.session_state["chat_history"].append({"role": "assistant", "content": "No keywords found for the specified criteria."})
                    # Clear the result_df so previous results don't show below
                    st.session_state["result_df"] = None

            elif intent == "sql":

                if is_sql_query(sql_query):
                    # Show SQL query
                    st.session_state["chat_history"].append({
                        "role": "assistant",
                        "content": "Generated SQL Query:",
                        "sql_query": sql_query
                    })
                    st.session_state["last_sql"] = sql_query
                    # Check for aggregation
                    if any(keyword in sql_query.upper() for keyword in ['GROUP BY', 'COUNT(', 'SUM(', 'AVG(', 'MAX(', 'MIN(']):
                        agg_result_df, agg_error = run_sql_query(sql_query)
                        if agg_error:
                            st.session_state["chat_history"].append({"role": "assistant", "content": f"SQL Error: {agg_error}"})
                        elif agg_result_df is not None and len(agg_result_df) > 0:
                            st.session_state["chat_history"].append({"role": "assistant", "content": "Aggregation Results:", "type": "aggregation", "df": agg_result_df})
                            st.session_state["result_df"] = None
                        else:
                            st.session_state["chat_history"].append({"role": "assistant", "content": "No aggregation results found."})
                            st.session_state["result_df"] = None
                    else:
                        result_df, error = run_sql_query(sql_query)
                        if error:
                            st.session_state["chat_history"].append({"role": "assistant", "content": f"SQL Error: {error}"})
                            st.session_state["result_df"] = None
                        elif result_df is not None:
                            st.session_state["chat_history"].append({"role": "assistant", "content": f"**Total Papers found: {len(result_df)}**", "type": "results", "df": result_df})
                            st.session_state["result_df"] = result_df
                            if "last_clarify_context" in st.session_state:
                                del st.session_state["last_clarify_context"]
                        else:
                            st.session_state["chat_history"].append({"role": "assistant", "content": "No results found for this query."})
                            st.session_state["result_df"] = None
                else:
                    st.session_state["chat_history"].append({"role": "assistant", "content": sql_query})
            st.rerun()

    # If the last result_df is present, allow editing cluster_label as before
    result_df = st.session_state.get("result_df", None)
    if result_df is not None and len(result_df) > 0:
        # Special case: only cluster_label column, just display as table
        if list(result_df.columns) == ["cluster_label"]:
            st.write("### Cluster Labels")
            st.dataframe(result_df, use_container_width=True)
        # Normal case: show edit form if id is present
        elif 'id' in result_df.columns:
            display_df = result_df.drop(columns=["id"]).reset_index(drop=True)
            with st.form("edit_form"):
                edited_df = st.data_editor(
                    display_df,
                    column_config={
                        "cluster_label": st.column_config.TextColumn("cluster_label", required=False)
                    },
                    disabled=[col for col in display_df.columns if col != "cluster_label"],
                    use_container_width=True,
                    key="editable_results"
                )
                save_button = st.form_submit_button("💾 Save Changes")
                if save_button:
                    # Clear the persistent success message when a new edit is submitted
                    st.session_state["show_label_update_success"] = False
                    from sql_agent_core import update_cluster_label, llm_suggest_label_correction
                    changes = 0
                    suggestions = []
                    # Always check and prompt for every changed label
                    for i, row in edited_df.iterrows():
                        orig_label = result_df.iloc[i]["cluster_label"]
                        new_label = row["cluster_label"]
                        if orig_label != new_label:
                            row_id = result_df.iloc[i]["id"]  # Use id for backend update
                            suggested_label, is_correction = llm_suggest_label_correction(new_label)
                            suggestions.append({
                                "row_index": i,
                                "row_id": row_id,
                                "orig_label": orig_label,
                                "user_label": new_label,
                                "suggestion": suggested_label,
                                "is_correction": is_correction
                            })
                    if suggestions:
                        st.session_state["pending_label_suggestions"] = suggestions
                        st.session_state["awaiting_label_confirmation"] = True
                        st.rerun()
                    else:
                        st.info("No changes to save.")
        else:
            st.warning("Query results must include the 'id' column for editing.")

    # Place the typo confirmation dialog at the very end of main()
    if st.session_state.get("awaiting_label_confirmation", False):
        suggestions = st.session_state.get("pending_label_suggestions", [])
        confirmed_updates = []
        typo_suggestions = [sug for sug in suggestions if sug["is_correction"] and sug["suggestion"] != sug["user_label"]]
        with st.container():
            for idx, sug in enumerate(typo_suggestions):
                key_yes = f"label_confirm_yes_{idx}"
                key_no = f"label_confirm_no_{idx}"
                st.info(f"Did you mean '{sug['suggestion']}' instead of '{sug['user_label']}' for row {sug['row_index']+1}?")
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("Yes", key=key_yes):
                        confirmed_updates.append({
                            "row_index": sug["row_index"],
                            "row_id": sug["row_id"],
                            "label": sug["suggestion"]
                        })
                        st.session_state[f"label_confirmed_{idx}"] = True
                with col2:
                    if st.button("No", key=key_no):
                        confirmed_updates.append({
                            "row_index": sug["row_index"],
                            "row_id": sug["row_id"],
                            "label": sug["user_label"]
                        })
                        st.session_state[f"label_confirmed_{idx}"] = True
            # For labels with no correction, auto-confirm
            for sug in suggestions:
                if not (sug["is_correction"] and sug["suggestion"] != sug["user_label"]):
                    confirmed_updates.append({
                        "row_index": sug["row_index"],
                        "row_id": sug["row_id"],
                        "label": sug["user_label"]
                    })
            # Check if all typo suggestions have been confirmed
            all_confirmed = all(st.session_state.get(f"label_confirmed_{i}") for i in range(len(typo_suggestions)))
            if all_confirmed:
                from sql_agent_core import update_cluster_label
                changes = 0
                for upd in confirmed_updates:
                    updated = update_cluster_label(int(upd["row_id"]), upd["label"])
                    if updated:
                        changes += 1
                    st.session_state["result_df"].at[upd["row_index"], "cluster_label"] = upd["label"]
                if changes > 0:
                    st.session_state["show_label_update_success"] = True
                else:
                    st.info("No changes to save.")
                st.session_state["awaiting_label_confirmation"] = False
                st.session_state["pending_label_suggestions"] = []
                for i in range(len(typo_suggestions)):
                    st.session_state.pop(f"label_confirmed_{i}", None)
                st.rerun()

    # Show a persistent success message after label update, if needed
    if st.session_state.get("show_label_update_success", False):
        st.success("Label(s) updated successfully!")
    # Do NOT clear the flag here; only clear it when a new edit is submitted

    # In the edit form, clear the flag when a new save is attempted
    # (insert this just before processing a new save)
    # ...
    # Inside the edit form, before processing save_button:
    # if save_button:
    #     st.session_state["show_label_update_success"] = False

if __name__ == "__main__":
    main() 