"""mnemo live demo: chat with an assistant and watch its long-term memory work.

Run locally:   streamlit run demo/app.py
Deploy:        Streamlit Community Cloud -> main file `demo/app.py`, secret GROQ_API_KEY
"""

from __future__ import annotations

import os
import sys
import tempfile
import uuid
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # run without installing
from mnemo import Memory, Settings
from mnemo.agent import MemoryAgent
from mnemo.retrieval import HybridRetriever
from mnemo.store import Filters
from mnemo.types import EventType, MemoryType

load_dotenv(Path(__file__).resolve().parents[1] / ".env")  # local runs: key from .env

MAX_TURNS_PER_SESSION = 40
EVENT_STYLE = {
    EventType.ADD: ("green", "ADD"),
    EventType.UPDATE: ("orange", "UPDATE"),
    EventType.DELETE: ("red", "DELETE"),
    EventType.NONE: ("gray", "KNOWN"),
}
SAMPLE_CONVERSATION = [
    "Hi! I'm Aisha, a final-year CS student in Hyderabad.",
    "I'm vegetarian, and I'm training for my first half marathon in December.",
    "Please keep your answers short and use bullet points.",
    "Actually, I just moved to Bangalore for an internship at a fintech startup.",
]

st.set_page_config(page_title="mnemo: memory for LLM agents", layout="wide")


def _secret(name: str) -> str | None:
    try:
        return st.secrets.get(name) or os.getenv(name)
    except Exception:  # no secrets.toml
        return os.getenv(name)


@st.cache_resource(show_spinner="Loading embedding model...")
def get_memory(api_key: str) -> Memory:
    os.environ["GROQ_API_KEY"] = api_key
    db = Path(tempfile.gettempdir()) / "mnemo-demo.db"
    return Memory(Settings(_env_file=None, llm_provider="groq", sqlite_path=str(db)))


# --- session ---------------------------------------------------------------------------
state = st.session_state
state.setdefault("user_id", f"demo-{uuid.uuid4().hex[:8]}")
state.setdefault("history", [])  # list of {"role", "content", "turn"?}
state.setdefault("turns", 0)

with st.sidebar:
    st.title("mnemo")
    st.caption("Long-term memory for LLM agents")
    api_key = _secret("GROQ_API_KEY")
    if not api_key:
        api_key = st.text_input(
            "Groq API key", type="password", help="Free at console.groq.com. Not stored."
        )
    if not api_key:
        st.info("Add a Groq API key to start chatting.")
        st.stop()

    memory = get_memory(api_key)
    agent = MemoryAgent(memory)
    user_id = state.user_id
    st.markdown(f"**Session user:** `{user_id}`")
    st.caption("Each browser session gets its own private memory space.")
    st.metric("Memories stored", memory.count(user_id))

    if st.button("Load sample conversation", use_container_width=True):
        with st.spinner("Extracting and consolidating memories..."):
            for line in SAMPLE_CONVERSATION:
                memory.add(line, user_id)
        st.rerun()
    if st.button("Forget me (delete all)", use_container_width=True):
        memory.delete_all(user_id)
        state.history = []
        st.rerun()
    st.divider()
    st.markdown(
        "[GitHub](https://github.com/dhruv-deepak/mnemo) · "
        "[How it works](https://github.com/dhruv-deepak/mnemo#how-it-works)"
    )

chat_tab, search_tab, memories_tab, graph_tab = st.tabs(
    ["Chat", "Search playground", "Memory inspector", "Knowledge graph"]
)


# --- chat ------------------------------------------------------------------------------
def render_turn_details(turn) -> None:
    with st.expander(
        f"Memory activity · {len(turn.memories)} recalled · "
        f"retrieval {turn.retrieval_ms:.0f} ms · LLM {turn.generation_ms:.0f} ms"
    ):
        left, right = st.columns(2)
        with left:
            st.markdown("**Recalled**")
            if not turn.memories:
                st.caption("Nothing relevant yet.")
            for r in turn.memories:
                st.markdown(f"- {r.memory.content}  \n  `score {r.score:.2f}`")
        with right:
            st.markdown("**Written**")
            events = turn.write.events if turn.write else []
            if not events:
                st.caption("Nothing worth remembering.")
            for e in events:
                color, label = EVENT_STYLE[e.event]
                text = e.new_content or e.old_content
                st.markdown(f":{color}[**{label}**] {text}")
                if e.event is EventType.UPDATE and e.old_content:
                    st.caption(f"was: {e.old_content}")


with chat_tab:
    st.caption(
        "Tell the assistant about yourself, then start a new topic and ask about it later. "
        "Expand *Memory activity* under each reply to see what was recalled and written."
    )
    for message in state.history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("turn") is not None:
                render_turn_details(message["turn"])

    if prompt := st.chat_input("Say something about yourself, or ask a question"):
        if state.turns >= MAX_TURNS_PER_SESSION:
            st.warning("Demo limit reached for this session. Run it locally for unlimited use.")
            st.stop()
        state.turns += 1
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            plain_history = [{"role": m["role"], "content": m["content"]} for m in state.history]
            try:
                with st.spinner("Recalling and thinking..."):
                    turn = agent.respond(user_id, prompt, plain_history)
                st.markdown(turn.reply)  # show the reply before the slower memory write
                with st.spinner("Updating memory..."):
                    turn.write = agent.remember(user_id, prompt, turn.reply)
            except Exception as exc:  # provider outage, rate limit, bad key ...
                state.turns -= 1
                st.error(f"The model provider returned an error, please try again. ({exc})")
                st.stop()
            render_turn_details(turn)
        state.history += [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": turn.reply, "turn": turn},
        ]
        st.rerun()


# --- search playground ------------------------------------------------------------------
with search_tab:
    st.caption(
        "Hybrid retrieval fuses vector, keyword and knowledge-graph rankings with Reciprocal "
        "Rank Fusion, then adjusts for recency and importance. Toggle signals to compare."
    )
    query = st.text_input("Query", placeholder="Where does Aisha live?")
    c1, c2, c3 = st.columns(3)
    use_keyword = c1.toggle("Keyword (BM25)", value=True)
    use_graph = c2.toggle("Knowledge graph", value=True)
    limit = c3.slider("Results", 1, 20, 5)
    if query:
        # A throwaway retriever with its own settings, so toggles never affect other sessions.
        retriever = HybridRetriever(
            memory.store,
            memory.embedder,
            memory.settings.model_copy(update={"use_keyword": use_keyword, "use_graph": use_graph}),
        )
        results = retriever.search(query, Filters(user_id=user_id), limit)
        rows = [
            {
                "memory": r.memory.content,
                "score": round(r.score, 3),
                "vector rank": r.breakdown.vector_rank,
                "keyword rank": r.breakdown.keyword_rank,
                "graph rank": r.breakdown.graph_rank,
                "cosine": round(r.breakdown.vector_similarity or 0, 3),
                "recency": round(r.breakdown.recency, 2),
                "importance": r.memory.importance,
            }
            for r in results
        ]
        st.dataframe(rows, use_container_width=True, hide_index=True)


# --- memory inspector -------------------------------------------------------------------
with memories_tab:
    items = memory.get_all(user_id, limit=500)
    if not items:
        st.caption("No memories yet. Chat a little or load the sample conversation.")
    type_filter = st.multiselect(
        "Type", [t.value for t in MemoryType], default=[t.value for t in MemoryType]
    )
    for item in [i for i in items if i.type.value in type_filter]:
        with st.container(border=True):
            top, actions = st.columns([5, 1])
            top.markdown(f"**{item.content}**")
            top.caption(
                f"{item.type.value} · importance {item.importance:.1f} · "
                f"recalled {item.access_count}x · updated {item.updated_at:%d %b %Y %H:%M}"
            )
            if actions.button("Delete", key=f"del-{item.id}"):
                memory.delete(item.id)
                st.rerun()
            history = memory.history(item.id)
            if len(history) > 1:
                with st.expander(f"History ({len(history)} events)"):
                    for e in history:
                        st.markdown(
                            f"`{e.created_at:%d %b %H:%M}` **{e.event.value}** "
                            f"{e.old_content or ''} → {e.new_content or ''}"
                        )


# --- knowledge graph --------------------------------------------------------------------
with graph_tab:
    graph = memory.graph(user_id)
    if not graph["edges"]:
        st.caption("Relations between people, places and things appear here as you chat.")
    else:
        lines = ["digraph G {", "rankdir=LR;", 'node [shape=box, style="rounded"];']
        for edge in graph["edges"]:
            s, r, t = (edge[k].replace('"', "'") for k in ("source", "relation", "target"))
            lines.append(f'"{s}" -> "{t}" [label="{r}"];')
        lines.append("}")
        st.graphviz_chart("\n".join(lines), use_container_width=True)
        st.caption(f"{len(graph['nodes'])} entities · {len(graph['edges'])} relations")
