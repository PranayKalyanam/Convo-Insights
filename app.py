"""Streamlit dashboard for the conversation-insights pipeline.

    streamlit run app.py
"""
from __future__ import annotations

import json
import uuid

import pandas as pd
import plotly.express as px
import streamlit as st

from src import config, db
from src.extractor import extract
from src.llm_client import LLMError, get_client

st.set_page_config(page_title="Conversation insights", page_icon="📞", layout="wide")

SENTIMENT_COLORS = {"negative": "#C8423B", "neutral": "#8C96A3", "positive": "#2F8F5B"}
STATUS_COLORS = {"unresolved": "#C8423B", "partially_resolved": "#D99A2B", "resolved": "#2F8F5B"}
QA_LABELS = {f"qa_{k}": v for k, v in config.QA_CHECKS.items()}
nice = lambda s: str(s).replace("_", " ").capitalize()  # noqa: E731


@st.cache_data(ttl=30)
def load() -> pd.DataFrame:
    return db.load_df()


def sample_transcript() -> str:
    path = config.DATA_DIR / "conversations.csv"
    if path.exists():
        return pd.read_csv(path).sample(1)["transcript"].iloc[0]
    return ("Agent: Hello, RentEasy support, this is Priya. How may I help you?\n"
            "Customer: I paid my rent of Rs 18500 by credit card yesterday but the owner says "
            "it's not credited. This is the third time I am calling.\n"
            "Agent: I'm sorry for the trouble. The transaction is pending with the bank; I cannot "
            "do anything until the bank updates it.\n"
            "Customer: This is useless. Nothing has been done after so many calls.")


# ------------------------------------------------------------------ sidebar
df_all = load()
st.title("Support conversation insights")

with st.sidebar:
    st.subheader("Filters")
    if st.button("Reload data"):
        load.clear()
        st.rerun()
    valid_df = df_all[df_all["valid"] == 1] if not df_all.empty else df_all
    def pick(label, col, options):
        present = [o for o in options if not valid_df.empty and o in set(valid_df[col])]
        return st.multiselect(label, present, default=present, format_func=nice)
    f_intent = pick("Intent", "intent", list(config.INTENTS))
    f_sent = pick("Sentiment", "sentiment", config.SENTIMENTS)
    f_status = pick("Resolution", "resolution_status", config.RESOLUTION_STATUSES)
    f_prio = pick("Priority", "priority", config.PRIORITIES)
    langs = sorted(valid_df["language"].dropna().unique()) if not valid_df.empty else []
    f_lang = st.multiselect("Language", langs, default=langs, format_func=nice)
    st.divider()
    st.caption(f"Extractor for new transcripts: `{config.LLM_PROVIDER} / {config.active_model()}`")
    if not df_all.empty:
        bad = int((df_all["valid"] == 0).sum())
        st.caption(f"{len(df_all)} rows in database, {bad} failed validation.")

if df_all.empty:
    st.info("The database is empty. Generate data and run the pipeline first:\n\n"
            "```\npython -m src.generate_data\npython -m src.pipeline\n```\n\n"
            "You can still analyse a single transcript in the third tab.")

df = valid_df
if not df.empty:
    df = df[df["intent"].isin(f_intent) & df["sentiment"].isin(f_sent)
            & df["resolution_status"].isin(f_status) & df["priority"].isin(f_prio)]
    if langs:
        df = df[df["language"].isin(f_lang) | df["language"].isna()]

tab_overview, tab_convs, tab_live, tab_eval = st.tabs(
    ["Overview", "Conversations", "Analyse a transcript", "Evaluation"])

# ------------------------------------------------------------------ overview
with tab_overview:
    if df.empty:
        st.write("No conversations match the current filters.")
    else:
        c = st.columns(5)
        c[0].metric("Conversations", f"{len(df):,}")
        c[1].metric("Unresolved", f"{(df['resolution_status'] == 'unresolved').mean():.0%}")
        c[2].metric("Negative customers", f"{(df['sentiment'] == 'negative').mean():.0%}")
        c[3].metric("High priority", int((df["priority"] == "high").sum()))
        c[4].metric("Avg agent QA score", f"{df['qa_score'].mean():.0f} / 100")

        left, right = st.columns(2)
        counts = (df.groupby(["intent", "resolution_status"]).size()
                  .reset_index(name="conversations"))
        counts["intent"] = counts["intent"].map(nice)
        order = counts.groupby("intent")["conversations"].sum().sort_values().index.tolist()
        fig = px.bar(counts, y="intent", x="conversations", color="resolution_status",
                     orientation="h", color_discrete_map=STATUS_COLORS,
                     category_orders={"intent": order[::-1],
                                      "resolution_status": config.RESOLUTION_STATUSES[::-1]},
                     title="What customers call about, and how it ends")
        fig.update_layout(legend_title_text="", yaxis_title="", margin=dict(l=0, r=0, t=50, b=0))
        left.plotly_chart(fig)

        sent = (df.groupby(["intent", "sentiment"]).size() / df.groupby("intent").size()).reset_index(name="share")
        sent["intent"] = sent["intent"].map(nice)
        fig = px.bar(sent, y="intent", x="share", color="sentiment", orientation="h",
                     color_discrete_map=SENTIMENT_COLORS, category_orders={"intent": order[::-1],
                     "sentiment": ["negative", "neutral", "positive"]},
                     title="Customer sentiment at end of call, by intent")
        fig.update_layout(legend_title_text="", yaxis_title="", xaxis_tickformat=".0%",
                          margin=dict(l=0, r=0, t=50, b=0))
        right.plotly_chart(fig)

        left, right = st.columns([1, 2])
        qa = pd.DataFrame({"check": [QA_LABELS[k] for k in QA_LABELS],
                           "pass_rate": [df[k].mean() for k in QA_LABELS]})
        fig = px.bar(qa, x="pass_rate", y="check", orientation="h", title="Agent checklist pass rate",
                     color_discrete_sequence=["#2364AA"])
        fig.update_layout(xaxis_tickformat=".0%", xaxis_range=[0, 1], yaxis_title="", xaxis_title="",
                          margin=dict(l=0, r=0, t=50, b=0))
        left.plotly_chart(fig)

        right.markdown("**Follow-up queue: high priority, not resolved**")
        queue = df[(df["priority"] == "high") & (df["resolution_status"] != "resolved")]
        right.dataframe(
            queue[["conversation_id", "intent", "summary", "next_action"]]
            .assign(intent=lambda d: d["intent"].map(nice)),
            hide_index=True, height=320)

# ------------------------------------------------------------------ conversations
with tab_convs:
    if df.empty:
        st.write("No conversations match the current filters.")
    else:
        q = st.text_input("Search transcripts and summaries", placeholder="e.g. deposit, UPI, packers")
        view = df
        if q:
            mask = (view["transcript"].str.contains(q, case=False, na=False)
                    | view["summary"].str.contains(q, case=False, na=False))
            view = view[mask]
        st.caption(f"{len(view)} conversations")
        st.dataframe(view[["conversation_id", "language", "intent", "sentiment", "resolution_status",
                           "priority", "qa_score", "summary"]],
                     hide_index=True, height=280)

        if not view.empty:
            cid = st.selectbox("Open a conversation", view["conversation_id"].tolist())
            row = view[view["conversation_id"] == cid].iloc[0]
            left, right = st.columns([3, 2])
            left.markdown("**Transcript** (personal data masked)")
            left.text(row["transcript"])
            right.markdown(f"**Summary**  \n{row['summary']}")
            right.markdown(
                f"**Intent:** {nice(row['intent'])}  \n"
                f"**Sentiment:** {nice(row['sentiment'])}  \n"
                f"**Resolution:** {nice(row['resolution_status'])}  \n"
                f"**Priority:** {nice(row['priority'])}")
            right.markdown(f"**Next action**  \n{row['next_action']}")
            right.markdown(f"**Agent checklist** ({row['qa_score']:.0f}/100)  \n" + "  \n".join(
                f"{'✅' if row[k] else '❌'} {v}" for k, v in QA_LABELS.items()))
            with right.expander("Processing details"):
                st.write(f"{row['provider']} / {row['model']}, {row['attempts']} attempt(s), "
                         f"{row['latency_ms']:.0f} ms, masked: {row['pii_counts']}")
                st.code(row["raw_output"], language="json")

# ------------------------------------------------------------------ live analysis
with tab_live:
    st.write("Paste any transcript (English, Hindi or Hinglish). Personal data is masked "
             "before it is sent to the model.")
    if "live_text" not in st.session_state:
        st.session_state.live_text = ""
    if st.button("Load a random example"):
        st.session_state.live_text = sample_transcript()
    text = st.text_area("Transcript", key="live_text", height=220,
                        placeholder="Agent: Hello...\nCustomer: ...")
    save = st.checkbox("Save the result to the database")
    if st.button("Analyse transcript", type="primary", disabled=not text.strip()):
        try:
            client = get_client()
        except LLMError as e:
            st.error(f"Could not start the model client: {e}. Check LLM_PROVIDER and your API key in .env.")
        else:
            with st.spinner("Analysing..."):
                res = extract(text, client)
            if not res.valid:
                st.error(f"The model output failed validation after {res.attempts} attempt(s): {res.error}")
                st.code(res.raw_output or "(empty response)")
            else:
                ins = res.insight
                st.success(f"Done in {res.latency_ms:.0f} ms with {client.provider}/{client.model}"
                           + (f", masked {res.pii_counts}" if res.pii_counts else ""))
                c = st.columns(4)
                c[0].metric("Intent", nice(ins.intent))
                c[1].metric("Sentiment", nice(ins.sentiment))
                c[2].metric("Resolution", nice(ins.resolution_status))
                c[3].metric("Priority", nice(ins.priority))
                st.markdown(f"**Summary**  \n{ins.summary}")
                st.markdown(f"**Next action**  \n{ins.next_action}")
                st.markdown(f"**Agent checklist** ({ins.agent_qa.score:.0f}/100)  \n" + "  \n".join(
                    f"{'✅' if getattr(ins.agent_qa, k) else '❌'} {v}" for k, v in config.QA_CHECKS.items()))
                with st.expander("Raw JSON"):
                    st.json(json.loads(ins.model_dump_json()))
            if save:
                conn = db.connect()
                cid = f"manual_{uuid.uuid4().hex[:8]}"
                db.save(conn, cid, res, client.provider, client.model, None)
                load.clear()
                st.caption(f"Saved as {cid}.")

# ------------------------------------------------------------------ evaluation
with tab_eval:
    report = config.REPORTS_DIR / "eval_report.md"
    if not report.exists():
        st.info("No evaluation yet. Run `python -m src.evaluate` and reload this page.")
    else:
        st.markdown(report.read_text())
        cm_path = config.REPORTS_DIR / "intent_confusion.csv"
        if cm_path.exists():
            cm = pd.read_csv(cm_path, index_col=0)
            fig = px.imshow(cm, text_auto=True, color_continuous_scale="Blues", aspect="auto",
                            labels=dict(x="Predicted", y="Gold", color="Count"),
                            title="Intent confusion matrix")
            st.plotly_chart(fig)
        err_path = config.REPORTS_DIR / "errors.csv"
        if err_path.exists():
            errs = pd.read_csv(err_path)
            with st.expander(f"Rows with at least one wrong label ({len(errs)})"):
                st.dataframe(errs, hide_index=True)
