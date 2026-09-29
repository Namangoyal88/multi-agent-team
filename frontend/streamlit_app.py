"""Dark Streamlit control center for the Multi-Agent AI Research Team."""
from __future__ import annotations

import os
import sys
import time
from collections import Counter

import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from frontend.api_client import ResearchAPI  # noqa: E402
from frontend.theme import (  # noqa: E402
    ACCENT,
    ACCENT_SOFT,
    AMBER,
    BLUE,
    CYAN,
    GREEN_2,
    PANEL,
    ROSE,
    VIOLET,
    html_card,
    inject_css,
    metric_card,
    page_header,
    team_card,
    topbar,
)

st.set_page_config(
    page_title="Multi-Agent AI Research Team",
    page_icon="⌘",
    layout="wide",
    initial_sidebar_state="expanded",
)
inject_css()

PAGES = {
    "RESEARCH": [
        ("Overview", "⌂"),
        ("Live Research", "◉"),
        ("Agent Activity", "⌘"),
    ],
    "KNOWLEDGE": [
        ("Research Workspace", "□"),
        ("Papers / Evidence", "▤"),
        ("Experiments", "⚗"),
        ("Reports", "▧"),
        ("Memory", "◎"),
    ],
    "SYSTEM": [
        ("Model Routing", "⇄"),
        ("System Health", "⌁"),
    ],
}

TEAM = [
    ("Researcher", "Finds evidence. Connects the dots.", "GPT-OSS 20B", "⌕"),
    ("Mathematician", "Turns ideas into rigorous theory.", "GPT-OSS 120B", "∑"),
    ("ML Engineer", "Designs models and implementations.", "GPT-OSS 120B", "</>"),
    ("Experimenter", "Prompt Guard 86M safety gate.", "Llama Prompt Guard 2 86M", "⚗"),
    ("Reviewer", "Challenges methodology and evidence.", "GPT-OSS 20B", "◉"),
    ("Verifier", "Prompt Guard 22M evidence gate.", "Llama Prompt Guard 2 22M", "◇"),
]


def _set_page(page: str) -> None:
    st.session_state.page = page
    st.rerun()


def render_sidebar(runs: list[dict], backend_online: bool) -> None:
    with st.sidebar:
        st.markdown(
            '<div style="display:flex;gap:.8rem;align-items:center;padding:.2rem .15rem .95rem">'
            '<div style="width:48px;height:48px;border-radius:13px;background:#1d5b39;color:#bff2ca;display:flex;align-items:center;justify-content:center;font-size:1.35rem;font-weight:700">⌘</div>'
            '<div><div style="font-size:.88rem;font-weight:760;letter-spacing:.08em;color:#e7f1e9">MULTI AGENT AI</div>'
            '<div style="font-size:.76rem;font-weight:700;letter-spacing:.16em;color:#75b987;margin-top:.15rem">RESEARCH TEAM</div></div></div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            '<div class="panel panel-tight" style="margin-bottom:.8rem">'
            '<div style="display:flex;justify-content:space-between;align-items:center">'
            '<div><div class="tiny">PERSONAL WORKSPACE</div><div style="font-weight:650;color:#e0e9e2;margin-top:.2rem">Local research environment</div></div>'
            '<span style="color:#718178">⌄</span></div></div>',
            unsafe_allow_html=True,
        )

        if st.button("＋  New research", use_container_width=True, type="primary", key="new_research_nav"):
            _set_page("New Research")

        for section, items in PAGES.items():
            st.markdown(f'<div class="nav-section">{section}</div>', unsafe_allow_html=True)
            for label, icon in items:
                active = st.session_state.get("page", "Overview") == label
                if active:
                    st.markdown(f'<div class="nav-active">{icon}&nbsp;&nbsp;{label}</div>', unsafe_allow_html=True)
                elif st.button(f"{icon}  {label}", key=f"nav_{label}", use_container_width=True):
                    _set_page(label)

        if runs:
            st.markdown('<div class="nav-section">CURRENT RUN</div>', unsafe_allow_html=True)
            ids = [r["id"] for r in runs]
            current = st.session_state.get("rid")
            if current not in ids:
                current = ids[0]
                st.session_state.rid = current
            st.selectbox(
                "Research run",
                ids,
                index=ids.index(current),
                label_visibility="collapsed",
                format_func=lambda i: f"{i} · " + next(r["request"]["question"][:38] for r in runs if r["id"] == i),
                key="run_select",
                on_change=lambda: st.session_state.update(rid=st.session_state.run_select),
            )

        st.markdown("<div style='height:1.7rem'></div>", unsafe_allow_html=True)
        status_class = "service-on" if backend_online else "service-off"
        status = "Backend connected" if backend_online else "Backend offline"
        st.markdown(
            f'<div class="panel panel-tight"><div class="tiny">WORKSPACE STATUS</div><div class="{status_class}" style="margin-top:.38rem;font-weight:650">● {status}</div>'
            f'<div class="tiny" style="margin-top:.35rem">Private, local-first research workspace</div></div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            '<div style="margin-top:1rem;color:#5f7268;font-size:.65rem;display:flex;justify-content:space-between">'
            '<span>Workspace v1.0</span><span>⌗</span></div>',
            unsafe_allow_html=True,
        )


def load_context():
    api = ResearchAPI(os.environ.get("RESEARCH_API_URL", "http://127.0.0.1:8000"))
    try:
        runs = api.list()
        return api, runs, True, None
    except Exception as exc:
        return api, [], False, exc


def run_context(api: ResearchAPI, runs: list[dict]) -> str | None:
    if not runs:
        return st.session_state.get("rid")
    ids = [r["id"] for r in runs]
    rid = st.session_state.get("rid")
    if rid not in ids:
        rid = ids[0]
        st.session_state.rid = rid
    return rid


def quick_prompts(api: ResearchAPI, backend_online: bool) -> None:
    st.markdown('<div class="tiny" style="margin:.9rem 0 .45rem;text-transform:uppercase;letter-spacing:.14em">NEED A STARTING POINT?</div>', unsafe_allow_html=True)
    cols = st.columns(3)
    examples = [
        ("Agentic RAG & hallucinations", "Investigate whether agentic RAG reduces hallucinations compared with conventional RAG."),
        ("Compare ML architectures", "Compare modern retrieval and ranking model architectures for research systems."),
        ("Explore a research gap", "Identify open research gaps in evidence-grounded multi-agent AI research."),
    ]
    for col, (label, question) in zip(cols, examples):
        with col:
            if st.button(f"{label}  →", use_container_width=True, key=f"quick_{label}", disabled=not backend_online):
                result = api.start(question, False)
                st.session_state.rid = result["research_id"]
                st.session_state.page = "Live Research"
                st.rerun()


def render_composer(api: ResearchAPI, backend_online: bool, compact: bool = False) -> None:
    st.markdown(
        '<div class="composer-title"><div><div class="composer-kicker">NEW RESEARCH</div>'
        '<div class="composer-question">What would you like to discover?</div>'
        '<div class="composer-help">Give your research team a question. They will plan, delegate, verify, and connect the work.</div></div>'
        '<span class="pill">LOCAL-FIRST</span></div>',
        unsafe_allow_html=True,
    )
    question = st.text_area(
        "Research question",
        height=128 if compact else 156,
        placeholder="Ask a big question. Explore an idea. Challenge a hypothesis...",
        label_visibility="collapsed",
        max_chars=8000,
        key="research_question",
        disabled=not backend_online,
    )
    cols = st.columns([1.15, .85, .7])
    with cols[0]:
        mode = st.selectbox("Mode", ["Automatic", "Evidence-first", "Experiment-first"], index=0, label_visibility="collapsed", key="research_mode", disabled=not backend_online)
    with cols[1]:
        depth = st.selectbox("Depth", ["Standard depth", "Deep research", "Fast synthesis"], index=0, label_visibility="collapsed", key="research_depth", disabled=not backend_online)
    with cols[2]:
        st.markdown('<div class="tiny" style="padding-top:.72rem;text-align:right">Advanced</div>', unsafe_allow_html=True)
    with st.expander("Advanced controls", expanded=False):
        exp = st.checkbox("Allow experiment execution (backend must also enable code execution)", value=False, disabled=not backend_online)
        st.caption(f"Mode: {mode} · Depth: {depth} · Maximum question length: 8,000 characters")
    footer_cols = st.columns([1, .9])
    with footer_cols[0]:
        st.markdown('<div class="tiny" style="padding-top:.8rem">⌕ Private to your local workspace</div>', unsafe_allow_html=True)
    with footer_cols[1]:
        if st.button("✦  Start research  →", type="primary", use_container_width=True, disabled=(not backend_online or len(question.strip()) < 8), key=f"start_{'compact' if compact else 'main'}"):
            result = api.start(question.strip(), exp)
            st.session_state.rid = result["research_id"]
            st.session_state.page = "Live Research"
            st.rerun()
    quick_prompts(api, backend_online)


def render_overview(api: ResearchAPI, runs: list[dict], backend_online: bool) -> None:
    topbar(backend_online)
    h1, h2 = st.columns([1, .22])
    with h1:
        page_header("YOUR RESEARCH, REIMAGINED", "Big questions. <span class='accent'>Better discoveries.</span>", "Your autonomous research team, ready when curiosity strikes.")
    with h2:
        st.markdown("<div style='height:1.8rem'></div>", unsafe_allow_html=True)
        if st.button("＋  New research", type="secondary", use_container_width=True, key="top_new_research"):
            _set_page("New Research")

    rid = run_context(api, runs)
    completed = sum(r.get("status") == "completed" for r in runs)
    active_agents = 0
    paper_count = 0
    if rid and backend_online:
        try:
            status = api.status(rid)
            active_agents = sum(a.get("status") in {"running", "working", "queued"} for a in status.get("agents", []))
            paper_count = len(api.papers(rid))
        except Exception:
            pass
    cols = st.columns(4)
    with cols[0]: metric_card("Research runs", str(len(runs)), "Every question is a new beginning", "⚗")
    with cols[1]: metric_card("Active agents", f"{active_agents} / 6", "Your research team is on standby" if not active_agents else "Agents working on this run", "⌘")
    with cols[2]: metric_card("Papers collected", str(paper_count), "Grounded in real-world evidence", "▤")
    with cols[3]: metric_card("Reports generated", str(completed), "Knowledge, ready to share", "▧")

    st.markdown("<div style='height:1.25rem'></div>", unsafe_allow_html=True)
    left, right = st.columns([1.55, .72], gap="large")
    with left:
        with st.container(border=True):
            render_composer(api, backend_online)
    with right:
        st.markdown(
            '<div class="mind-card"><div class="panel-label">● &nbsp;THE MIND BEHIND YOUR TEAM</div>'
            '<div class="mind-title">One question.<br>An entire research team.</div>'
            '<div class="panel-muted">Your director plans, delegates, verifies, and connects the work. Deep reasoning is invoked when the problem actually needs it.</div>'
            '<div style="margin-top:.85rem;color:#829b87;font-size:.7rem">✦ Your research question</div>'
            '<div class="director"><div class="director-name">▦ &nbsp; Deep Research Director</div><div class="director-model">NVIDIA Nemotron 3 Ultra</div></div>'
            '<div class="tree-line"></div>'
            '<div class="mini-agents"><div class="mini-agent"><span class="mi">⌕</span>Researcher</div><div class="mini-agent"><span class="mi">&lt;/&gt;</span>ML Engineer</div><div class="mini-agent"><span class="mi">◇</span>Verifier</div></div>'
            '<div class="mind-footer"><span>Deep reasoning, only when needed</span><span>↗</span></div></div>',
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
    st.markdown('<div style="display:flex;align-items:center;justify-content:space-between"><div><h2 style="margin-bottom:.15rem">Your research team</h2><div class="page-sub" style="margin:0">Different expertise. Shared intelligence. One goal. <span class="pill" style="margin-left:.4rem">6 specialists</span></div></div><span class="tiny">Meet the team&nbsp; →</span></div>', unsafe_allow_html=True)
    st.markdown("<div style='height:.75rem'></div>", unsafe_allow_html=True)
    cols = st.columns(6, gap="small")
    for col, (name, desc, model, icon) in zip(cols, TEAM):
        with col:
            team_card(name, desc, model, icon)

    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
    status_text = "Research service is online. Your interface is connected to FastAPI." if backend_online else "Research service is offline. Your interface is ready; start FastAPI to create persistent research runs."
    status_icon = "✓" if backend_online else "!"
    service_class = 'service-on' if backend_online else 'service-off'
    st.markdown(
        f'<div class="service-banner"><span><span class="{service_class}" style="font-weight:800;margin-right:.45rem">{status_icon}</span><span class="strong">{status_text}</span></span><span>View setup&nbsp; →</span></div>',
        unsafe_allow_html=True,
    )
    st.markdown('<div class="source-row"><span>CONNECTED TO THE WORLD OF KNOWLEDGE</span><span>◌ <b>arXiv</b></span><span>▤ <b>Semantic Scholar</b></span><span>PubMed</span><span>⌘ <b>GitHub</b></span><span>◉ <b>Web search</b></span><span style="margin-left:auto">Retrieved on demand ↗</span></div>', unsafe_allow_html=True)
    st.markdown('<div class="footer"><span>◫ Built for curiosity. Grounded in evidence.</span><span>Deep Agents&nbsp;&nbsp;·&nbsp;&nbsp;LangGraph&nbsp;&nbsp;·&nbsp;&nbsp;Local-first</span></div>', unsafe_allow_html=True)


def render_new_research(api: ResearchAPI, backend_online: bool) -> None:
    topbar(backend_online)
    page_header("NEW RESEARCH", "Start with a question.", "Your director turns the question into a research plan and routes work to the right specialists.")
    st.markdown("<div style='height:.8rem'></div>", unsafe_allow_html=True)
    left, right = st.columns([1.4, .65], gap="large")
    with left:
        with st.container(border=True):
            render_composer(api, backend_online)
    with right:
        st.markdown('<div class="panel-soft"><div class="panel-label">HOW IT WORKS</div><h3 style="margin:.65rem 0">Director → specialists → evidence → report</h3><div class="panel-muted">Nemotron handles planning and escalation. Groq powers specialist work. Every factual claim is checked before it can enter the final report.</div><div style="height:1rem"></div><div class="pill">NVIDIA · PLANNER</div><div style="height:.5rem"></div><div class="pill">GROQ · SPECIALISTS</div><div style="height:.5rem"></div><div class="pill">VERIFICATION · REQUIRED</div></div>', unsafe_allow_html=True)


def badge(status: str) -> str:
    labels = {"done": "✓", "running": "◌", "failed": "!", "completed": "✓", "queued": "•"}
    cls = f"status-{status}"
    return f'<span class="{cls}" style="font-weight:750">{labels.get(status, "·")} {status}</span>'


def render_live(api: ResearchAPI, rid: str | None, backend_online: bool) -> None:
    topbar(backend_online)
    page_header("RESEARCH", "Live research run", "Watch the plan, evidence collection, agent activity, and verification as the team works.")
    if not backend_online:
        st.warning("FastAPI is not reachable. Start the backend to inspect live runs.")
        return
    if not rid:
        st.info("Start a research run from Overview or New Research first.")
        return
    s = api.status(rid)
    top = st.columns([1.4, 1, 1, 1])
    with top[0]:
        st.markdown(f'<div class="status-card"><div class="tiny">RUN STATUS</div><div style="font-size:1.15rem;margin-top:.35rem">{badge(s["status"])}</div><div class="panel-muted" style="margin-top:.45rem">{s["question"][:90]}</div></div>', unsafe_allow_html=True)
    with top[1]: st.markdown(f'<div class="status-card"><div class="tiny">PROGRESS</div><div style="font-size:1.6rem;font-weight:720;margin-top:.4rem">{round(s.get("progress", 0)*100)}%</div></div>', unsafe_allow_html=True)
    with top[2]: st.markdown(f'<div class="status-card"><div class="tiny">STAGES DONE</div><div style="font-size:1.6rem;font-weight:720;margin-top:.4rem">{len(s.get("stages_done", []))}</div></div>', unsafe_allow_html=True)
    with top[3]: st.markdown(f'<div class="status-card"><div class="tiny">ISSUES</div><div style="font-size:1.6rem;font-weight:720;margin-top:.4rem">{len(s.get("errors", []))}</div></div>', unsafe_allow_html=True)
    st.progress(s.get("progress", 0.0))
    left, right = st.columns([1.35, .8], gap="large")
    with left:
        st.markdown('<div class="panel"><div class="panel-title">Agent event stream</div><div class="panel-muted" style="margin:.25rem 0 .8rem">Newest activity first.</div>', unsafe_allow_html=True)
        events = api.events(rid)
        for e in events[-45:][::-1]:
            stamp = time.strftime("%H:%M:%S", time.localtime(e["ts"]))
            st.markdown(f'<div class="event-row"><div><span class="event-time">{stamp}</span><span class="event-agent">{e["agent"]}</span><span class="event-kind">{e["kind"]}</span></div><div class="event-msg">{e["message"]}</div></div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with right:
        st.markdown('<div class="panel"><div class="panel-title">Research plan</div><div class="panel-muted" style="margin:.3rem 0 .75rem">The director selects stages dynamically; verification, reporting, and evaluation remain mandatory.</div></div>', unsafe_allow_html=True)
        if s.get("plan"):
            st.markdown("**Planned stages**")
            for stage in s["plan"].get("stages", []):
                done = stage in s.get("stages_done", [])
                stage_class = "service-on" if done else "panel-muted"
                stage_icon = "✓" if done else "○"
                st.markdown(f'<div class="event-row"><span class="{stage_class}">{stage_icon}</span> &nbsp; {stage}</div>', unsafe_allow_html=True)
        if s.get("errors"):
            st.error("Warnings / errors are exposed instead of hidden.")
            with st.expander("Inspect errors"):
                st.json(s["errors"])
        if st.button("Refresh run", use_container_width=True):
            st.rerun()


def render_agent_activity(api: ResearchAPI, rid: str | None, backend_online: bool) -> None:
    topbar(backend_online)
    page_header("TEAM", "Agent activity", "See which specialist is working, which model it is using, and whether NVIDIA escalation was triggered.")
    if not backend_online or not rid:
        st.info("Start a research run to inspect agent activity.")
        return
    s = api.status(rid)
    rows = []
    for a in s.get("agents", []):
        rows.append({"Agent": a["agent"], "Provider": a["provider"].upper(), "Model": a["model"], "Task": a["task"], "Status": a["status"], "Progress": a["progress"], "Tools": ", ".join(a["tools_used"]) or "-", "NVIDIA escalation": "Yes" if a["escalated"] else "No", "Duration (s)": a["duration_s"]})
    st.dataframe(rows, use_container_width=True, hide_index=True)


def render_workspace(api: ResearchAPI, rid: str | None, backend_online: bool) -> None:
    topbar(backend_online)
    page_header("KNOWLEDGE", "Research workspace", "Inspect artifacts produced by every stage without losing the raw work behind the final report.")
    if not backend_online or not rid:
        st.info("No research run selected.")
        return
    artifacts = api.artifacts(rid)
    for a in artifacts:
        if a["kind"] == "state":
            continue
        with st.expander(f"{a['kind']}  ·  {a['name']}"):
            st.json(a["data"]) if not isinstance(a["data"], str) else st.markdown(a["data"])


def render_papers(api: ResearchAPI, rid: str | None, backend_online: bool) -> None:
    topbar(backend_online)
    page_header("EVIDENCE", "Papers & claim verification", "A source index paired with the verification record for factual claims.")
    if not backend_online or not rid:
        st.info("No research run selected.")
        return
    papers = api.papers(rid)
    if papers:
        st.dataframe([{"ID": p["pid"], "Title": p["title"], "Year": p.get("year"), "Source": p.get("source"), "URL": p.get("url")} for p in papers], use_container_width=True, hide_index=True)
    else:
        st.info("No papers collected yet.")
    ver = api.artifacts(rid, "verification")
    if ver:
        st.markdown("### Claim verification")
        st.dataframe([{k: c.get(k) for k in ("id", "text", "citations", "status", "rationale")} for c in ver[0]["data"]], use_container_width=True, hide_index=True)


def render_experiments(api: ResearchAPI, rid: str | None, backend_online: bool) -> None:
    topbar(backend_online)
    page_header("LAB", "Experiments", "Executed experiments are isolated in the research sandbox and their metrics are kept alongside the experiment artifact.")
    if not backend_online or not rid:
        st.info("No research run selected.")
        return
    ex = api.artifacts(rid, "experiment")
    if not ex:
        st.info("No experiment artifacts for this run. Code execution is off unless explicitly enabled.")
    for a in ex:
        st.markdown(f'<div class="panel" style="margin-bottom:.7rem"><div class="panel-title">{a["name"]}</div></div>', unsafe_allow_html=True)
        if a["name"] == "code":
            st.code(a["data"], language="python")
        else:
            st.json(a["data"])


def render_reports(api: ResearchAPI, rid: str | None, backend_online: bool) -> None:
    topbar(backend_online)
    page_header("KNOWLEDGE", "Reports", "Final outputs are generated from verified evidence, not raw unsupported stage material.")
    if not backend_online or not rid:
        st.info("No research run selected.")
        return
    try:
        md = api.report(rid)
        cols = st.columns([1, .2])
        with cols[0]: st.markdown("### Final report")
        with cols[1]: st.download_button("Download .md", md, file_name=f"research_{rid}.md", use_container_width=True)
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown(md)
        st.markdown('</div>', unsafe_allow_html=True)
    except Exception:
        st.info("Report not ready yet.")


def render_memory(api: ResearchAPI, backend_online: bool) -> None:
    topbar(backend_online)
    page_header("KNOWLEDGE", "Memory", "Search prior research context that the team has stored in the local workspace.")
    if not backend_online:
        st.info("Memory search is available when the backend is running.")
        return
    q = st.text_input("Semantic search over previous research", placeholder="e.g. retrieval grounding, hallucination evaluation")
    if q:
        r = api.memory(q)
        stats = r.get("stats", {})
        cols = st.columns(3)
        with cols[0]: metric_card("Stored memories", str(stats.get("items", stats.get("count", 0))), "Local workspace memory", "◎")
        with cols[1]: metric_card("Graph enabled", "Yes" if stats.get("graph_enabled", True) else "No", "Knowledge graph setting", "⌘")
        with cols[2]: metric_card("Matches", str(len(r.get("results", []))), "Retrieved for this query", "⌕")
        for x in r.get("results", []):
            html_card(f'<div class="tiny">{x["kind"]} · score {x["score"]} · run {x["job_id"]}</div><div style="margin-top:.35rem;color:#dce8e0">{x["text"][:500]}</div>', "panel panel-tight")


def render_models(api: ResearchAPI, backend_online: bool) -> None:
    topbar(backend_online)
    page_header("SYSTEM", "Model routing", "Groq powers the specialist team; NVIDIA Nemotron runs the deep-research director and escalation path.")
    if not backend_online:
        st.info("Connect the backend to inspect the live routing table.")
        return
    st.dataframe(api.models(), use_container_width=True, hide_index=True)
    st.markdown('<div class="panel-soft"><div class="panel-label">ROUTING PRINCIPLE</div><div class="panel-title" style="margin-top:.5rem">Use deep reasoning only when the research problem needs it.</div><div class="panel-muted" style="margin-top:.3rem">Specialists stay on Groq for focused work. Low-confidence results can escalate to NVIDIA Nemotron 3 Ultra through the Deep Research Director.</div></div>', unsafe_allow_html=True)


def render_health(api: ResearchAPI, backend_online: bool) -> None:
    topbar(backend_online)
    page_header("SYSTEM", "System health", "Provider availability, tool status, memory, and local execution policy in one place.")
    if not backend_online:
        st.error(f"Backend unreachable at {api.c.base_url}. Start FastAPI, then refresh this page.")
        return
    health = api.health()
    cols = st.columns(4)
    with cols[0]: metric_card("API", "Online", "FastAPI responding", "✓")
    with cols[1]: metric_card("NVIDIA", "Ready" if health.get("nvidia_available") else "Off", "Director / escalation", "⌘")
    with cols[2]: metric_card("Groq", "Configured" if health.get("groq_configured") else "Missing", "Specialist provider", "↗")
    with cols[3]: metric_card("Code execution", "Enabled" if health.get("code_execution_enabled") else "Disabled", "Sandbox policy", "⚗")
    st.markdown("### Tool status")
    st.json(health.get("tools", {}))
    st.markdown("### Provider checks")
    if st.button("Check provider model IDs against live /models", key="provider_check"):
        st.json(api.providers(check=True))
    else:
        st.json(api.providers())


def main() -> None:
    if "page" not in st.session_state:
        st.session_state.page = "Overview"
    api, runs, backend_online, backend_error = load_context()
    render_sidebar(runs, backend_online)
    rid = run_context(api, runs)

    page = st.session_state.page
    if page == "Overview":
        render_overview(api, runs, backend_online)
    elif page == "New Research":
        render_new_research(api, backend_online)
    elif page == "Live Research":
        render_live(api, rid, backend_online)
    elif page == "Agent Activity":
        render_agent_activity(api, rid, backend_online)
    elif page == "Research Workspace":
        render_workspace(api, rid, backend_online)
    elif page == "Papers / Evidence":
        render_papers(api, rid, backend_online)
    elif page == "Experiments":
        render_experiments(api, rid, backend_online)
    elif page == "Reports":
        render_reports(api, rid, backend_online)
    elif page == "Memory":
        render_memory(api, backend_online)
    elif page == "Model Routing":
        render_models(api, backend_online)
    elif page == "System Health":
        render_health(api, backend_online)
    else:
        st.session_state.page = "Overview"
        st.rerun()


if __name__ == "__main__":
    main()
