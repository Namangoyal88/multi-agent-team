"""Dark UI theme and reusable presentation helpers for the Streamlit client."""
from __future__ import annotations

from html import escape

import streamlit as st


BG = "#0a0f0d"
PANEL = "#111814"
PANEL_2 = "#151d18"
BORDER = "#26332b"
TEXT = "#eef5f0"
MUTED = "#8fa39a"
ACCENT = "#46d67f"
ACCENT_SOFT = "#183424"
GREEN_2 = "#a8e4b8"
VIOLET = "#a893ea"
BLUE = "#78b8ff"
AMBER = "#edc273"
ROSE = "#ea8eab"
CYAN = "#61d6d1"


def inject_css() -> None:
    st.markdown(
        f"""
        <style>
        :root {{
            --bg: {BG};
            --panel: {PANEL};
            --panel-2: {PANEL_2};
            --border: {BORDER};
            --text: {TEXT};
            --muted: {MUTED};
            --accent: {ACCENT};
            --accent-soft: {ACCENT_SOFT};
        }}

        .stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"] {{
            background: var(--bg) !important;
            color: var(--text) !important;
        }}
        [data-testid="stHeader"] {{ background: transparent !important; }}
        [data-testid="stToolbar"] {{ visibility: hidden; height: 0; }}
        [data-testid="stDecoration"] {{ display: none; }}
        [data-testid="stSidebar"] {{
            background: #0c120f !important;
            border-right: 1px solid var(--border);
        }}
        [data-testid="stSidebar"] > div:first-child {{ padding-top: 1rem; }}
        [data-testid="stSidebarContent"] {{ padding: 0 18px 18px 18px; }}
        [data-testid="stSidebarUserContent"] {{ padding-bottom: 1rem; }}
        .block-container {{
            max-width: 1420px;
            padding: 1.1rem 2.2rem 3rem 2.2rem;
        }}
        .stMarkdown, .stText, label, p, span, div {{ font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }}
        h1, h2, h3, h4 {{ color: var(--text) !important; letter-spacing: -0.02em; }}
        h1 {{ font-size: 2.5rem !important; line-height: 1.05 !important; }}
        h2 {{ font-size: 1.45rem !important; }}
        h3 {{ font-size: 1.08rem !important; }}
        .stCaption, small {{ color: var(--muted) !important; }}
        [data-testid="stVerticalBlockBorderWrapper"] {{
            background: var(--panel) !important;
            border: 1px solid var(--border) !important;
            border-radius: 18px !important;
            padding: .35rem .25rem !important;
        }}
        [data-testid="stMetric"] {{
            background: var(--panel) !important;
            border: 1px solid var(--border) !important;
            border-radius: 16px !important;
            padding: 1rem !important;
            box-shadow: 0 10px 30px rgba(0,0,0,.12);
        }}
        [data-testid="stMetricLabel"] {{ color: var(--muted) !important; }}
        [data-testid="stMetricValue"] {{ color: var(--text) !important; }}
        [data-testid="stMetricDelta"] {{ color: var(--GREEN_2) !important; }}

        textarea, input, [data-baseweb="select"] > div, [data-baseweb="input"] > div {{
            background: #0d1410 !important;
            color: var(--text) !important;
            border-color: var(--border) !important;
        }}
        textarea::placeholder, input::placeholder {{ color: #62746a !important; }}
        [data-testid="stTextArea"] > div, [data-testid="stTextInput"] > div {{ border-radius: 12px; }}
        [data-baseweb="select"] *, [data-baseweb="input"] * {{ color: var(--text) !important; }}
        [data-baseweb="popover"] {{ background: #101812 !important; }}
        [role="option"] {{ background: #101812 !important; color: var(--text) !important; }}
        [role="option"]:hover {{ background: #17221b !important; }}

        div.stButton > button {{
            border: 1px solid var(--border) !important;
            background: #101611 !important;
            color: var(--text) !important;
            border-radius: 10px !important;
            min-height: 40px;
            font-weight: 600 !important;
            transition: all .15s ease;
        }}
        div.stButton > button:hover {{
            border-color: #3b5144 !important;
            background: #172019 !important;
            transform: translateY(-1px);
        }}
        div.stButton > button[kind="primary"] {{
            background: var(--accent) !important;
            color: #06110a !important;
            border-color: var(--accent) !important;
        }}
        div.stButton > button[kind="primary"]:hover {{ background: #5be08d !important; }}
        [data-testid="stSidebar"] div.stButton > button {{
            width: 100%;
            justify-content: flex-start;
            text-align: left;
            border-color: transparent !important;
            background: transparent !important;
            color: #a6b5ad !important;
            box-shadow: none !important;
            font-weight: 500 !important;
        }}
        [data-testid="stSidebar"] div.stButton > button:hover {{
            background: #131c17 !important;
            color: var(--text) !important;
            transform: none;
        }}
        .nav-active {{
            background: #183224 !important;
            color: #bff2ca !important;
            border: 1px solid #28543a !important;
            border-radius: 10px;
            padding: .68rem .8rem;
            margin: .12rem 0;
            font-weight: 650;
        }}
        .nav-spacer {{ height: .25rem; }}
        .nav-section {{
            color: #64786d;
            font-size: .67rem;
            font-weight: 700;
            letter-spacing: .18em;
            text-transform: uppercase;
            margin: 1rem 0 .45rem .6rem;
        }}

        .topbar {{
            display:flex; align-items:center; justify-content:space-between;
            border-bottom:1px solid var(--border); padding:.15rem 0 1rem;
            margin-bottom:2rem;
        }}
        .crumbs {{ color:#718179; font-size:.86rem; }}
        .crumbs b {{ color:#d6e2db; font-weight:600; }}
        .top-actions {{ display:flex; align-items:center; gap:1.1rem; color:#819289; font-size:.8rem; }}
        .dot {{ width:7px; height:7px; border-radius:50%; background:var(--accent); display:inline-block; box-shadow:0 0 10px rgba(70,214,127,.55); margin-right:.35rem; }}
        .avatar {{ width:34px; height:34px; border-radius:50%; background:#e6ddcf; color:#725f4e; display:flex; align-items:center; justify-content:center; font-weight:700; }}

        .eyebrow {{
            color:#7fa48c; font-size:.68rem; font-weight:700; letter-spacing:.22em;
            text-transform:uppercase; margin-bottom:.55rem;
        }}
        .hero-title {{ font-size:2.7rem; font-weight:730; line-height:1.03; margin:0; color:#eff6f1; }}
        .hero-title .accent {{ color:#70c889; }}
        .hero-subtitle {{ color:#809188; font-size:1rem; margin:.65rem 0 0; }}

        .panel {{
            background:var(--panel); border:1px solid var(--border); border-radius:18px;
            padding:1.25rem; box-shadow:0 14px 36px rgba(0,0,0,.16);
        }}
        .panel-tight {{ padding:1rem 1.1rem; }}
        .panel-soft {{ background:#131d16; border:1px solid #29392f; border-radius:18px; padding:1.25rem; }}
        .panel-title {{ font-weight:680; font-size:1rem; color:#ebf4ee; }}
        .panel-muted {{ color:#829188; font-size:.83rem; line-height:1.55; }}
        .panel-label {{ color:#6e8176; font-size:.66rem; letter-spacing:.17em; text-transform:uppercase; font-weight:700; }}

        .stat-card {{
            background:var(--panel); border:1px solid var(--border); border-radius:16px; padding:1.05rem 1.1rem;
            min-height:124px;
        }}
        .stat-head {{ display:flex; justify-content:space-between; align-items:center; color:#8a9b92; font-size:.78rem; }}
        .stat-icon {{ width:32px; height:32px; display:flex; align-items:center; justify-content:center; border-radius:10px; background:#17241c; font-size:1rem; }}
        .stat-value {{ font-size:2rem; font-weight:700; margin-top:1rem; color:#edf5ef; }}
        .stat-value .slash {{ font-size:1rem; color:#63766c; font-weight:500; margin-left:.25rem; }}
        .stat-note {{ color:#718078; font-size:.72rem; margin-top:.2rem; }}

        .composer-title {{ display:flex; justify-content:space-between; gap:1rem; align-items:start; }}
        .composer-kicker {{ color:#6d8779; font-size:.65rem; letter-spacing:.15em; text-transform:uppercase; font-weight:700; }}
        .composer-question {{ font-size:1.23rem; font-weight:690; color:#eff6f0; margin:.4rem 0 .25rem; }}
        .composer-help {{ color:#7f9187; font-size:.83rem; }}
        .tiny {{ font-size:.68rem; color:#718178; }}
        .pill {{ display:inline-flex; padding:.25rem .48rem; border-radius:999px; font-size:.63rem; border:1px solid #2a4333; background:#142119; color:#9bc9a6; }}

        .mind-card {{
            background:linear-gradient(160deg,#13251b 0%,#0f1a14 100%); border:1px solid #2a4734;
            border-radius:18px; padding:1.35rem; min-height:430px; position:relative; overflow:hidden;
        }}
        .mind-card:after {{ content:""; position:absolute; width:190px; height:190px; right:-70px; top:-70px; border-radius:50%; border:1px solid rgba(93,155,111,.15); }}
        .mind-title {{ font-size:1.55rem; line-height:1.06; font-weight:700; margin:.75rem 0 .9rem; color:#c9e3ce; max-width:290px; }}
        .director {{ background:#e7f2e8; color:#12221a; border-radius:12px; padding:.8rem .85rem; margin:1.45rem 0 .55rem; border:1px solid #7f9e83; }}
        .director-name {{ font-weight:700; font-size:.86rem; }}
        .director-model {{ color:#66806c; font-size:.68rem; margin-top:.18rem; }}
        .tree-line {{ height:28px; border-left:1px solid #52715a; margin-left:28px; }}
        .mini-agents {{ display:grid; grid-template-columns:repeat(3,1fr); gap:.55rem; }}
        .mini-agent {{ background:#111b14; border:1px solid #304733; border-radius:11px; padding:.65rem .45rem; text-align:center; min-height:72px; color:#8ea996; font-size:.62rem; }}
        .mini-agent .mi {{ font-size:1rem; display:block; color:#a5cfae; margin-bottom:.25rem; }}
        .mind-footer {{ border-top:1px solid #304733; margin-top:1.15rem; padding-top:.7rem; color:#819986; font-size:.68rem; display:flex; justify-content:space-between; }}

        .team-card {{
            background:var(--panel); border:1px solid var(--border); border-radius:16px; padding:.95rem; min-height:205px;
        }}
        .team-top {{ display:flex; justify-content:space-between; align-items:start; gap:.4rem; }}
        .team-icon {{ width:36px; height:36px; border-radius:10px; display:flex; align-items:center; justify-content:center; font-size:1.15rem; background:#16231b; }}
        .standby {{ color:#88a393; font-size:.63rem; }}
        .standby i {{ width:6px; height:6px; border-radius:50%; display:inline-block; background:#7fb48b; margin-right:4px; }}
        .team-name {{ font-size:.92rem; font-weight:700; color:#edf4ef; margin-top:.9rem; }}
        .team-desc {{ color:#778a80; font-size:.69rem; line-height:1.45; margin:.45rem 0 .8rem; min-height:42px; }}
        .team-model {{ border-top:1px solid #253229; padding-top:.6rem; display:flex; gap:.35rem; align-items:center; color:#8fa198; font-size:.67rem; }}
        .groq {{ color:#f3c267; font-weight:800; text-transform:lowercase; }}

        .service-banner {{ background:#111b14; border:1px solid #324536; border-radius:14px; padding:.8rem 1rem; display:flex; justify-content:space-between; align-items:center; gap:1rem; color:#94a899; font-size:.74rem; }}
        .service-banner .strong {{ color:#bdcfbf; }}
        .service-on {{ color:#78de92; }}
        .service-off {{ color:#d89a70; }}

        .source-row {{ display:flex; flex-wrap:wrap; gap:2.2rem; color:#768980; font-size:.75rem; align-items:center; padding:.55rem 0; }}
        .source-row span {{ white-space:nowrap; }}
        .source-row b {{ color:#a4b2aa; font-weight:550; }}

        .page-title {{ font-size:2rem; font-weight:720; margin:0; }}
        .page-sub {{ color:#778a80; margin-top:.35rem; margin-bottom:1.25rem; font-size:.88rem; }}
        .status-card {{ background:var(--panel); border:1px solid var(--border); border-radius:16px; padding:1rem; }}
        .status-running {{ color:#e7c56c; }} .status-completed {{ color:#69dd8d; }} .status-failed {{ color:#ef8f8f; }} .status-queued {{ color:#91b8f0; }}
        .event-row {{ background:#0e1511; border:1px solid #202c25; border-radius:11px; padding:.68rem .8rem; margin-bottom:.45rem; }}
        .event-time {{ color:#617168; font-size:.62rem; margin-right:.6rem; }}
        .event-agent {{ color:#a7ceb0; font-weight:650; font-size:.7rem; margin-right:.55rem; }}
        .event-kind {{ color:#6f8478; font-size:.63rem; text-transform:uppercase; letter-spacing:.08em; }}
        .event-msg {{ color:#c4cec7; font-size:.76rem; margin-top:.25rem; }}

        .footer {{ border-top:1px solid var(--border); padding-top:1rem; margin-top:2rem; color:#65766e; font-size:.67rem; display:flex; justify-content:space-between; }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def html_card(inner: str, classes: str = "panel") -> None:
    st.markdown(f'<div class="{classes}">{inner}</div>', unsafe_allow_html=True)


def metric_card(label: str, value: str, note: str, icon: str, icon_class: str = "") -> None:
    st.markdown(
        f'''<div class="stat-card"><div class="stat-head"><span>{escape(label)}</span><span class="stat-icon {icon_class}">{icon}</span></div>
        <div class="stat-value">{escape(value)}</div><div class="stat-note">{escape(note)}</div></div>''',
        unsafe_allow_html=True,
    )


def topbar(backend_online: bool) -> None:
    state = "Local workspace" if backend_online else "Service offline"
    state_class = "service-on" if backend_online else "service-off"
    st.markdown(
        f'''<div class="topbar"><div class="crumbs"><b>Workspace</b><span style="margin:0 .55rem">›</span><b>Overview</b></div>
        <div class="top-actions"><span class="{state_class}"><span class="dot"></span>{state}</span><span>Documentation ↗</span><span style="font-size:1.05rem">♧</span><span class="avatar">R</span></div></div>''',
        unsafe_allow_html=True,
    )


def page_header(eyebrow: str, title: str, subtitle: str) -> None:
    st.markdown(f'<div class="eyebrow">{escape(eyebrow)}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="hero-title">{title}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="hero-subtitle">{escape(subtitle)}</div>', unsafe_allow_html=True)


def team_card(name: str, description: str, model: str, icon: str) -> None:
    st.markdown(
        f'''<div class="team-card"><div class="team-top"><span class="team-icon">{icon}</span><span class="standby"><i></i>Standby</span></div>
        <div class="team-name">{escape(name)}</div><div class="team-desc">{escape(description)}</div>
        <div class="team-model"><span class="groq">groq</span><span>{escape(model)}</span></div></div>''',
        unsafe_allow_html=True,
    )
