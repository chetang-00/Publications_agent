import os
import sqlite3
import pandas as pd
from portkey_ai import Portkey
from dotenv import load_dotenv
import re

# Load environment variables from .env
load_dotenv()

# Read Portkey API keys from environment
PORTKEY_API_KEY = os.getenv("PORTKEY_API_KEY", "")
PORTKEY_VIRTUAL_KEY = os.getenv("PORTKEY_VIRTUAL_KEY", "")
PORTKEY_BASE_URL = "https://ai-gateway.apps.cloud.rt.nyu.edu/v1/"

def get_portkey_client():
    return Portkey(
        api_key=PORTKEY_API_KEY,
        virtual_key=PORTKEY_VIRTUAL_KEY,
        base_url=PORTKEY_BASE_URL
    )

client = get_portkey_client()

db_path = './publications.db'
table_name = 'publications'

# Table schema for prompt context
schema = '''
Table: publications
Columns:
- id (INTEGER PRIMARY KEY AUTOINCREMENT)
- authors (TEXT)
- title (TEXT)
- year (INTEGER)
- abstract (TEXT)
- author_keywords (TEXT)
- index_keywords (TEXT)
- authors_with_affiliations (TEXT)
- cluster_label (TEXT)
'''

def clean_sql_response(response_text):
    """Extract clean SQL from LLM response, removing markdown formatting"""
    sql = re.sub(r'```sql\s*', '', response_text)
    sql = re.sub(r'```\s*$', '', sql)
    sql = re.sub(r'^```\s*', '', sql)
    sql = sql.strip()
    return sql

def enforce_case_insensitive(sql):
    """
    Post-process the generated SQL to enforce case-insensitive search for LIKE clauses.
    Replaces patterns like `column LIKE '%term%'` with `LOWER(column) LIKE LOWER('%term%')`.
    """
    # Regex to find patterns like: column LIKE '%term%'
    def repl(match):
        col = match.group(1)
        val = match.group(2)
        return f"LOWER({col}) LIKE LOWER({val})"
    # Handles both WHERE and AND/OR clauses
    sql = re.sub(r"([a-zA-Z_][a-zA-Z0-9_\"]*)\s+LIKE\s+('[^']*'|\?+)", repl, sql)
    return sql
import json 
def ask_llm_for_sql(question, history=None, model="gpt-4o-mini"):
    # Format history if provided
    history_text = ""
    if history:
        for msg in history:
            role = "User" if msg["role"] == "user" else "Assistant"
            history_text += f"{role}: {msg['content']}\n"
        history_text = f"Conversation history:\n{history_text}\n"

    prompt = f"""{history_text}
You are an expert assistant that helps translate user questions about a scientific publication database into SQL queries or helpful follow-up questions.

Here is the database schema:

{schema}

Your task:
1. Classify the user's `intent` as one of the following:
   - `"sql"` if the user wants to list, filter, or explore papers (e.g., see titles, years, authors).
   - `"summary"` only if the user explicitly asks for a summary, synthesis, or overview across multiple papers.
   - `"keyword_analysis"` if the user asks about common research topics, most frequent keywords, or research areas from index_keywords.
   - `"clarify"` if the question is too vague to generate a useful SQL query.

**Important:** Do NOT return `intent = "summary"` unless the user uses words like "summarize", "overview", "synthesis", or clearly implies summarization.

2. For `"clarify"` intent:
   - Leave `sql_query` empty.
   - Provide a helpful `clarify_message` to make the question more specific.

3. For `"summary"` intent:
   - Return a SQL query selecting only `id`, `abstract`.
   - Include filters such as keyword, author, or year if available.

4. For `"keyword_analysis"` intent:
   - Return "KEYWORD_ANALYSIS" as the sql_query (this will be handled by special keyword extraction logic).
   - Extract author and year filters from the question and include them in the clarify_message field formatted as: "author:AuthorName|year:>=2020" (or similar year format).

5. For `"sql"` intent:
   - Return a SQL query selecting:
     `id`, `title`, `year`, `authors`, `abstract`, `author_keywords`, `index_keywords`, `authors_with_affiliations`, `cluster_label`.

6. Always:
   - Use case-insensitive search: LOWER(column) LIKE LOWER('%term%')
   - Quote column names with spaces using double quotes.
   - Respond in this strict JSON format:

{{
  "intent": "sql" | "summary" | "keyword_analysis" | "clarify",
  "sql_query": "<SQL query or empty or KEYWORD_ANALYSIS>",
  "clarify_message": "<clarifying question or filter parameters for keyword_analysis>"
}}

7. If the user asks for all unique cluster labels, return:
SELECT DISTINCT cluster_label FROM publications WHERE cluster_label IS NOT NULL
Do NOT include the id column in this case.

Examples:

1. User question: "papers on biology"
→ Response:
{{
  "intent": "clarify",
  "sql_query": "",
  "clarify_message": "Could you please specify what aspects of biology you are interested in? For example: genetics, microbiology, ecology. Also, do you want a summary or a list of papers?"
}}

**Important Author Query Rules:**
- "papers by Author A and Author B" = papers where BOTH authors appear together (use AND)
- "papers by Author A or Author B" = papers where EITHER author appears (use OR)
- "papers by Author A, Author B" = papers where EITHER author appears (use OR)

2. User question: "Summarize genetics research from 2021"
→ Response:
{{
  "intent": "summary",
  "sql_query": "SELECT id, abstract FROM publications WHERE LOWER(index_keywords) LIKE LOWER('%genetics%') AND year = 2021",
  "clarify_message": ""
}}

3. User question: "Show me all papers on CRISPR from 2020"
→ Response:
{{
  "intent": "sql",
  "sql_query": "SELECT id, title, year, authors, abstract, author_keywords, index_keywords, authors_with_affiliations, cluster_label FROM publications WHERE LOWER(index_keywords) LIKE LOWER('%crispr%') AND year = 2020",
  "clarify_message": ""
}}

4. User question: "Now show only papers from 2022"
   (Assume previous SQL: SELECT ... WHERE ... )
→ Response:
{{
  "intent": "sql",
  "sql_query": "SELECT ... WHERE ... AND year = 2022",
  "clarify_message": ""
}}

5. User question: "What are the most common research topics for author Ranganath R since 2020?"
→ Response:
{{
  "intent": "keyword_analysis",
  "sql_query": "KEYWORD_ANALYSIS",
  "clarify_message": "author:Ranganath R|year:>=2020"
}}

6. User question: "Which research topics are most frequent in machine learning papers?"
→ Response:
{{
  "intent": "keyword_analysis", 
  "sql_query": "KEYWORD_ANALYSIS",
  "clarify_message": "keyword:machine learning"
}}

7. User question: "papers by Treisman J and Harris E"
→ Response:
{{
  "intent": "sql",
  "sql_query": "SELECT id, title, year, authors, abstract, author_keywords, index_keywords, authors_with_affiliations, cluster_label FROM publications WHERE LOWER(authors) LIKE LOWER('%treisman j%') AND LOWER(authors) LIKE LOWER('%harris e%')",
  "clarify_message": ""
}}

8. User question: "papers by Smith A or Johnson B"
→ Response:
{{
  "intent": "sql",
  "sql_query": "SELECT id, title, year, authors, abstract, author_keywords, index_keywords, authors_with_affiliations, cluster_label FROM publications WHERE LOWER(authors) LIKE LOWER('%smith a%') OR LOWER(authors) LIKE LOWER('%johnson b%')",
  "clarify_message": ""
}}

User question: {question}
Response:
"""

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=400,
        temperature=0
    )

    content = response['choices'][0]['message']['content'].strip()
    print("Raw LLM response:", content)

    try:
        parsed = json.loads(content)
        intent = parsed.get("intent", "clarify")
        sql = clean_sql_response(parsed.get("sql_query", "")).strip()
        sql = enforce_case_insensitive(sql)
        clarify_message = parsed.get("clarify_message", "")
        return intent, sql, clarify_message
    except Exception as e:
        print(f"[ERROR] Could not parse LLM JSON: {e}")
        return "clarify", "", "Could you clarify your request with specific topics, keywords, authors, or years?"


def run_sql_query(sql, db_path=db_path):
    conn = sqlite3.connect(db_path)
    try:
        df = pd.read_sql_query(sql, conn)
        return df, None
    except Exception as e:
        return None, f"SQL Error: {e}"
    finally:
        conn.close()

def run_sql_query_with_params(sql, params, db_path=db_path):
    """Execute SQL query with parameters (for IN clauses, etc.)"""
    conn = sqlite3.connect(db_path)
    try:
        df = pd.read_sql_query(sql, conn, params=params)
        return df, None
    except Exception as e:
        return None, f"SQL Error: {e}"
    finally:
        conn.close()


def update_cluster_label(id, new_label, db_path=db_path):
    id = int(id)  # Ensure id is a native Python int
    print(f"[DEBUG] update_cluster_label: id={id}, new_label={new_label}, db_path={db_path}")
    conn = sqlite3.connect(db_path)
    try:
        cur = conn.cursor()
        cur.execute("SELECT id, cluster_label FROM publications WHERE id = ?", (id,))
        row = cur.fetchone()
        print(f"[DEBUG] Row before update: {row}")
        cur.execute("""
            UPDATE publications
            SET cluster_label = ?
            WHERE id = ?
        """, (new_label, id))
        conn.commit()
        print(f"[DEBUG] Rows updated: {cur.rowcount}")
        return cur.rowcount  # number of rows updated
    except Exception as e:
        print(f"[ERROR] Exception in update_cluster_label: {e}")
        return 0
    finally:
        conn.close()

def get_all_cluster_labels(db_path=db_path):
    conn = sqlite3.connect(db_path)
    try:
        cur = conn.cursor()
        cur.execute("SELECT DISTINCT cluster_label FROM publications WHERE cluster_label IS NOT NULL")
        labels = [row[0] for row in cur.fetchall()]
        return labels
    finally:
        conn.close()

def suggest_cluster_label(label, db_path=db_path, cutoff=0.8):
    import difflib
    labels = get_all_cluster_labels(db_path)
    suggestions = difflib.get_close_matches(label, labels, n=1, cutoff=cutoff)
    return suggestions[0] if suggestions else None

def llm_suggest_label_correction(label, context=None, model="gpt-4o-mini"):
    """
    Use the LLM to check if the label is a typo or misspelling and suggest a correction if needed.
    Returns (suggested_label, is_correction: bool)
    """
    prompt = f"""
You are an expert assistant for scientific publication data. The user has entered the following label for a publication cluster: '{label}'.

If this label contains any typo, misspelling, or incorrect word (even minor), always suggest the correct label. Be strict about spelling and grammar. If the label is correct or you are unsure, return the label as is.

Respond in JSON format: {{\"suggested_label\": <string>, \"is_correction\": <true|false>}}
"""
    if context:
        prompt = f"Context: {context}\n" + prompt
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=50,
        temperature=0.2
    )
    import json
    raw = response.model_dump()['choices'][0]['message']['content'].strip()
    print(f"[DEBUG] LLM raw response for label '{label}': {raw}")
    try:
        result = json.loads(raw)
        return result.get("suggested_label", label), result.get("is_correction", False)
    except Exception as e:
        print(f"[ERROR] Could not parse LLM response: {e}")
        return label, False
    
def extract_and_count_keywords(db_path=db_path, author_filter=None, year_filter=None, limit=20):
    """
    Extract individual keywords from index_keywords field and count their frequency.
    
    Args:
        db_path: Path to the database
        author_filter: Optional author name to filter by (case-insensitive)
        year_filter: Optional year filter (e.g., ">=2020" or "=2021")
        limit: Number of top keywords to return
    
    Returns:
        DataFrame with columns: research topic, count
    """
    conn = sqlite3.connect(db_path)
    try:
        # Build query based on filters
        base_query = "SELECT index_keywords FROM publications WHERE index_keywords IS NOT NULL"
        params = []
        
        if author_filter:
            base_query += " AND LOWER(authors) LIKE LOWER(?)"
            params.append(f"%{author_filter}%")
        
        if year_filter:
            if year_filter.startswith(">="):
                year_val = int(year_filter[2:])
                base_query += " AND year >= ?"
                params.append(year_val)
            elif year_filter.startswith("<="):
                year_val = int(year_filter[2:])
                base_query += " AND year <= ?"
                params.append(year_val)
            elif year_filter.startswith("="):
                year_val = int(year_filter[1:])
                base_query += " AND year = ?"
                params.append(year_val)
            else:
                # Assume direct year value
                base_query += " AND year = ?"
                params.append(int(year_filter))
        
        # Execute query to get all index_keywords
        df = pd.read_sql_query(base_query, conn, params=params)
        
        # Process keywords
        keyword_counts = {}
        for keywords_str in df['index_keywords'].dropna():
            # Split on semicolons and commas, clean and normalize
            keywords = re.split(r'[;,]', str(keywords_str))
            for keyword in keywords:
                keyword = keyword.strip()
                if keyword and len(keyword) > 2:  # Skip very short terms
                    # Normalize to lowercase for counting
                    keyword_lower = keyword.lower()
                    keyword_counts[keyword_lower] = keyword_counts.get(keyword_lower, 0) + 1
        
        # Convert to DataFrame and sort by count
        result_df = pd.DataFrame(list(keyword_counts.items()), columns=['research topic', 'count'])
        result_df = result_df.sort_values('count', ascending=False).head(limit)
        result_df = result_df.reset_index(drop=True)
        
        return result_df
    
    except Exception as e:
        print(f"[ERROR] Keyword extraction failed: {e}")
        return pd.DataFrame(columns=['research topic', 'count'])
    finally:
        conn.close()

def summarize_abstracts(abstracts, query_context="", model="gpt-4o-mini"):
    """
    Summarizes a list of research paper abstracts into a cohesive narrative.
    Optionally includes context from the user query to focus the summary.
    """
    if not abstracts:
        return "No abstracts available to summarize."


    MAX_ABSTRACTS = 30
    selected_abstracts = abstracts[:MAX_ABSTRACTS]
    combined = "\n\n".join(selected_abstracts)

    prompt = f"""
You are a research assistant. Your task is to write a concise, factual summary using ONLY the information from the abstracts below.

Do NOT add any external knowledge, examples, or assumptions not found in the abstracts.

Context: {query_context}

Here are the abstracts:
{combined}

Write a well-structured paragraph or bullet-point summary that highlights common themes, contributions, and areas of focus across these papers.
"""


    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500,
            temperature=0.4
        )
        return response['choices'][0]['message']['content'].strip()
    except Exception as e:
        print(f"[ERROR] Summarization failed: {e}")
        return "An error occurred while generating the summary."
