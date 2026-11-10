"""GradMit's visual system: palette, page CSS and the HTML building blocks.

The layout language follows the team's Data Vortex Round 4 dashboard (a branded
top bar, the project pipeline as navigation, stat tiles, chart cards with an
interpretation note, eyebrow headings and a method flowchart beside the files it
reads and writes). GradMit keeps its own palette: graphite with mint for the
selected model and gold for the baselines.

Nothing here computes a result. Every number passed in comes from ``outputs/``
or ``artifacts/``.
"""

from __future__ import annotations

import html
import itertools
from pathlib import Path

import streamlit as st

# --------------------------------------------------------------------------- palette
BG, PANEL, PANEL_2 = "#0C1316", "#141E22", "#1A262B"
BORDER, GRID = "rgba(186,214,220,.15)", "rgba(186,214,220,.10)"
TXT, MUT, DIM = "#EEF3F4", "#A3B5B9", "#71878C"
MINT, GOLD, SKY, CORAL, SLATE, LILAC = "#5EC4B0", "#E2B45A", "#6FA8DC", "#E07A5F", "#8B9BA3", "#A99BE0"
MINT_SOFT = "#2F6F65"

REPO = "https://github.com/SaiGrover/Gradmit-ML-Web-App"
REPO_LABEL = "SaiGrover/Gradmit-ML-Web-App"

_GITHUB = (
    '<svg viewBox="0 0 16 16" aria-hidden="true"><path fill="currentColor" d="M8 0C3.58 0 0 3.58 0 8c0 '
    "3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48"
    "-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07"
    "-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27"
    " 2-.27s1.36.09 2 .27c1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75"
    "-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.01 8.01 0 0 0 16 8c0-4.42-3.58"
    '-8-8-8z"/></svg>'
)

CSS = f"""
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@500;600;700&family=Inter:wght@400;500;600;700&display=swap');

.stApp {{ background:
    radial-gradient(1100px 520px at 88% -8%, rgba(94,196,176,.13), transparent 62%),
    radial-gradient(900px 480px at -6% 4%, rgba(226,180,90,.07), transparent 60%),
    {BG}; }}
[data-testid="stHeader"] {{ background: transparent; }}
[data-testid="stHeaderActionElements"], #MainMenu, footer {{ display:none !important; }}
.block-container {{ padding-top: 1rem; max-width: 1560px; }}

h1, h2, h3, h4 {{ letter-spacing: -.02em; }}
.stMarkdown p, .stMarkdown li {{ line-height: 1.62; }}
div[data-testid="stCaptionContainer"] p {{ color:{DIM} !important; font-size:.8rem !important; }}
div[data-testid="stWidgetLabel"] p {{ color:{MUT} !important; font-size:.72rem !important;
   letter-spacing:.1em; text-transform:uppercase; font-weight:600; }}
code {{ color:{MINT} !important; background:rgba(94,196,176,.10) !important; }}

/* ---------------------------------------------------------------- top bar */
.topbar {{ position:relative; display:flex; align-items:center; justify-content:space-between;
   flex-wrap:wrap; gap:1rem; padding:.4rem .2rem 1rem .2rem; margin-bottom:.9rem; }}
.topbar:after {{ content:""; position:absolute; left:0; right:0; bottom:0; height:1px;
   background:linear-gradient(90deg, transparent, {MINT}, {SKY}, {GOLD}, transparent);
   background-size:240% 100%; animation:sweep 14s linear infinite; opacity:.85; }}
@keyframes sweep {{ from {{ background-position:0% 0; }} to {{ background-position:240% 0; }} }}
.brand {{ display:flex; align-items:center; gap:.9rem; }}
.mark {{ width:48px; height:48px; display:grid; place-items:center; border-radius:15px;
   border:1px solid rgba(186,214,220,.22);
   background:radial-gradient(circle at 30% 25%, rgba(94,196,176,.32), rgba(111,168,220,.08) 60%, transparent 76%);
   box-shadow:0 0 32px -10px rgba(94,196,176,.8), inset 0 1px 0 rgba(255,255,255,.07); }}
.mark svg {{ width:32px; height:32px; }}
.wm {{ font-family:'Outfit','Inter',sans-serif; font-size:1.6rem; font-weight:700; letter-spacing:-.035em;
   line-height:1.1; background:linear-gradient(100deg, {TXT} 0%, {MINT} 30%, {SKY} 55%, {TXT} 85%);
   background-size:260% 100%; -webkit-background-clip:text; background-clip:text; color:transparent;
   -webkit-text-fill-color:transparent; animation:flow 13s ease-in-out infinite; }}
@keyframes flow {{ 0%,100% {{ background-position:0% 50%; }} 50% {{ background-position:100% 50%; }} }}
.tag {{ color:{MUT}; font-size:.8rem; margin-top:.2rem; }}
.tag b {{ color:{MINT}; font-weight:600; }}
.who {{ text-align:right; line-height:1.4; }}
.who .badge {{ display:inline-block; margin-bottom:.3rem; padding:.14rem .55rem; border-radius:999px;
   font-size:.7rem; font-weight:600; color:{GOLD}; background:rgba(226,180,90,.12);
   border:1px solid rgba(226,180,90,.4); }}
.who .t {{ font-family:'Outfit','Inter',sans-serif; font-size:.95rem; font-weight:600; color:{TXT}; }}
.repo {{ display:inline-flex; align-items:center; gap:.36rem; margin-top:.3rem; padding:.2rem .6rem;
   border-radius:999px; border:1px solid {BORDER}; background:rgba(255,255,255,.035);
   color:{MUT} !important; font-size:.74rem; text-decoration:none !important; transition:.15s ease; }}
.repo:hover {{ color:{TXT} !important; border-color:{MINT}; background:rgba(94,196,176,.1); }}
.repo svg {{ width:13px; height:13px; }}
@media (max-width: 640px) {{ .who {{ text-align:left; }} .wm {{ font-size:1.35rem; }} }}

/* ---------------------------------------------------------------- pipeline rail */
div[data-testid="stLayoutWrapper"]:has(> div.st-key-navrail) {{ position:sticky; top:0; z-index:99; }}
div.st-key-navrail {{ margin:-.2rem 0 1.4rem 0; padding:.55rem 0 .7rem 0;
   background:rgba(12,19,22,.88); backdrop-filter:blur(12px) saturate(1.15);
   border-bottom:1px solid {BORDER}; box-shadow:0 14px 22px -18px rgba(0,0,0,.95); }}
div.st-key-navrail button {{ width:100% !important; min-height:72px; background:rgba(255,255,255,.03) !important;
   border:1px solid {BORDER}; border-radius:13px !important; padding:.5rem .55rem !important;
   position:relative; overflow:hidden; transition:.16s ease; justify-content:flex-start; text-align:left; }}
div.st-key-navrail button:before {{ content:""; position:absolute; top:0; left:0; right:0; height:2px; }}
div.st-key-navrail button p {{ color:{DIM} !important; font-size:.68rem !important; line-height:1.45;
   margin:0; white-space:normal; text-align:left; }}
div.st-key-navrail button strong {{ display:inline-block; color:{TXT} !important; font-size:.9rem;
   font-weight:650; line-height:1.25; margin:.08rem 0; }}
div.st-key-navrail button em {{ display:inline-block; color:{MUT} !important; font-style:normal; font-size:.72rem; }}
div.st-key-navrail button:focus-visible {{ outline:2px solid {MINT}; outline-offset:2px; }}
.flowarrow {{ text-align:center; color:{DIM}; font-size:1rem; opacity:.75; }}
@media (max-width: 900px) {{
   div[data-testid="stLayoutWrapper"]:has(> div.st-key-navrail) {{ position:static; }}
   div.st-key-navrail div[data-testid="stHorizontalBlock"] {{ flex-wrap:nowrap !important; overflow-x:auto;
      gap:.45rem; padding-bottom:.3rem; scrollbar-width:thin; }}
   div.st-key-navrail div[data-testid="stColumn"] {{ min-width:150px !important; flex:0 0 150px !important; }}
   div.st-key-navrail div[data-testid="stColumn"]:has(.flowarrow) {{ display:none; }}
}}

/* ---------------------------------------------------------------- hero */
.kicker {{ color:{MINT}; font-size:.85rem; font-weight:600; margin:.6rem 0 .3rem 0; }}
.lede {{ color:{MUT}; font-size:1.06rem; line-height:1.65; max-width:62ch; }}

/* ---------------------------------------------------------------- tiles */
.tiles {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(170px,1fr)); gap:.75rem; margin:.4rem 0 1.1rem 0; }}
.tile {{ background:linear-gradient(160deg, rgba(255,255,255,.055), rgba(255,255,255,.015));
   border:1px solid {BORDER}; border-radius:14px; padding:.85rem .95rem; position:relative; overflow:hidden; }}
.tile:before {{ content:""; position:absolute; top:0; left:0; width:100%; height:2px;
   background:linear-gradient(90deg,var(--c),transparent); }}
.tile .l {{ color:{MUT}; font-size:.74rem; font-weight:600; }}
.tile .v {{ font-family:'Outfit','Inter',sans-serif; font-size:1.75rem; font-weight:700; line-height:1.1;
   margin:.3rem 0 .15rem 0; letter-spacing:-.025em; color:var(--c); font-variant-numeric:tabular-nums; }}
.tile .n {{ color:{DIM}; font-size:.78rem; line-height:1.4; }}
div[data-testid="stMetric"] {{ background:linear-gradient(160deg, rgba(255,255,255,.05), rgba(255,255,255,.015));
   border-radius:14px; }}
div[data-testid="stMetricValue"] {{ font-family:'Outfit','Inter',sans-serif; color:{MINT}; }}

/* ---------------------------------------------------------------- cards and panels */
div[class*="st-key-card"] {{ background:{PANEL}; border:1px solid {BORDER}; border-radius:16px;
   padding:.8rem .9rem .45rem .9rem; height:100%; }}
div[class*="st-key-card"] div[data-testid="stExpander"] {{ border:0; background:transparent;
   border-top:1px solid {BORDER}; border-radius:0; }}
div[class*="st-key-card"] div[data-testid="stExpander"] details {{ border:0; }}
.cardtitle {{ font-family:'Outfit','Inter',sans-serif; font-size:1rem; font-weight:600; color:{TXT}; margin:.1rem 0 0 0; }}
.cardsub {{ color:{DIM}; font-size:.8rem; margin:.05rem 0 .2rem 0; }}
.panel {{ background:{PANEL}; border:1px solid {BORDER}; border-radius:16px; padding:1rem 1.15rem; min-height:10.5rem;
   position:relative; overflow:hidden; }}
.panel:before {{ content:""; position:absolute; top:0; left:0; width:100%; height:2px;
   background:linear-gradient(90deg,var(--c),transparent); }}
.panel h4 {{ font-family:'Outfit','Inter',sans-serif; margin:0 0 .3rem 0; font-size:1.04rem; font-weight:600; color:{TXT}; }}
.panel p {{ color:{MUT}; font-size:.88rem; line-height:1.62; margin:0; }}
.panel .ico {{ width:34px; height:34px; border-radius:10px; display:grid; place-items:center; margin-bottom:.6rem;
   color:var(--c); border:1px solid var(--c); background:rgba(255,255,255,.035); font-weight:700;
   font-family:'Outfit','Inter',sans-serif; }}
.verdict {{ border:1px solid rgba(226,180,90,.38); background:rgba(226,180,90,.07); border-radius:14px;
   padding:.85rem 1rem; color:{MUT}; font-size:.92rem; line-height:1.6; }}
.verdict b {{ color:{GOLD}; }}
div[data-testid="stVerticalBlockBorderWrapper"]:has(> div > div[data-testid="stVerticalBlock"]) {{ border-color:{BORDER}; }}
div[data-testid="stExpander"] {{ border:1px solid {BORDER}; border-radius:12px; background:rgba(255,255,255,.02); }}
div[data-testid="stExpander"] summary p {{ font-size:.86rem !important; color:{MUT} !important; }}

/* section headings from st.subheader share the eyebrow look */
[data-testid="stMain"] [data-testid="stHeading"] h3 {{ padding-left:.7rem; border-left:3px solid transparent;
   border-image:linear-gradient(180deg,{MINT},{SKY}) 1 100%; margin-top:1.2rem; }}
div[data-testid="stVerticalBlockBorderWrapper"] {{ background:{PANEL}; }}

/* ---------------------------------------------------------------- eyebrow */
.eyebrow {{ display:flex; align-items:baseline; flex-wrap:wrap; gap:.15rem .65rem; margin:1.8rem 0 .8rem 0;
   padding-left:.7rem; border-left:3px solid transparent; border-image:linear-gradient(180deg,{MINT},{SKY}) 1 100%; }}
.eyebrow h3 {{ font-family:'Outfit','Inter',sans-serif; margin:0; padding:0; font-size:1.25rem; font-weight:600; color:{TXT}; }}
.eyebrow span {{ color:{DIM}; font-size:.86rem; }}

/* ---------------------------------------------------------------- method flow */
.flow {{ margin:.2rem 0 1rem 0; }}
.flow .step {{ position:relative; display:grid; grid-template-columns:44px 1fr; gap:1rem; padding-bottom:1rem; }}
.flow .step:last-child {{ padding-bottom:0; }}
.flow .step:not(:last-child):after {{ content:""; position:absolute; left:21px; top:50px; bottom:-2px; width:2px;
   background:linear-gradient(180deg, var(--c), rgba(186,214,220,.15)); }}
.flow .node {{ width:44px; height:44px; border-radius:13px; display:grid; place-items:center;
   font-family:'Outfit','Inter',sans-serif; font-weight:700; font-size:.92rem; color:var(--c);
   background:rgba(255,255,255,.04); border:1.5px solid var(--c); box-shadow:0 0 20px -7px var(--c); }}
.flow .sbody {{ background:{PANEL}; border:1px solid {BORDER}; border-radius:14px; padding:.7rem .9rem .8rem .9rem; }}
.flow h5 {{ font-family:'Outfit','Inter',sans-serif; margin:0 0 .2rem 0; padding:0; font-size:.98rem; font-weight:600; color:{TXT}; }}
.flow p {{ margin:0; color:{MUT}; font-size:.86rem; line-height:1.6; }}
.flow .chip {{ display:inline-block; margin-top:.45rem; padding:.18rem .6rem; border-radius:999px;
   font-size:.74rem; font-weight:600; color:var(--c); background:rgba(255,255,255,.045); border:1px solid var(--c); }}
.iopanel {{ background:{PANEL}; border:1px solid {BORDER}; border-radius:16px; padding:1rem 1.1rem; }}
.io-h {{ font-family:'Outfit','Inter',sans-serif; font-size:1rem; font-weight:600; color:{TXT}; margin-bottom:.7rem; }}
.io-k {{ font-size:.76rem; font-weight:700; color:var(--c); margin:.1rem 0 .35rem 0; }}
.io-list {{ display:flex; flex-direction:column; gap:.3rem; }}
.io-row {{ display:flex; justify-content:space-between; align-items:baseline; gap:.7rem;
   background:rgba(255,255,255,.035); border:1px solid {BORDER}; border-radius:9px; padding:.38rem .6rem; }}
.io-n {{ font-size:.8rem; color:{TXT}; line-height:1.35; word-break:break-word; }}
.io-s {{ font-size:.72rem; color:{DIM}; white-space:nowrap; font-variant-numeric:tabular-nums; }}
.io-arrow {{ text-align:center; color:var(--c); font-size:1.05rem; margin:.45rem 0; }}
.io-engine {{ text-align:center; font-family:'Outfit','Inter',sans-serif; font-weight:600; font-size:.92rem;
   color:var(--c); border:1.5px solid var(--c); border-radius:12px; padding:.6rem .5rem;
   background:rgba(255,255,255,.04); box-shadow:0 0 26px -10px var(--c); }}
.io-f {{ margin-top:.75rem; font-size:.72rem; color:{DIM}; }}

/* ---------------------------------------------------------------- footer */
.foot {{ margin-top:2.4rem; padding-top:1rem; border-top:1px solid {BORDER}; color:{DIM}; font-size:.8rem;
   display:flex; flex-wrap:wrap; justify-content:space-between; gap:.6rem; }}
.foot a {{ color:{MUT} !important; }}

@media (prefers-reduced-motion: reduce) {{ .topbar:after, .wm {{ animation:none !important; }} }}
"""


def inject_css() -> None:
    st.html(f"<style>{CSS}</style>")


def _esc(text) -> str:
    return html.escape(str(text))


def topbar() -> None:
    # st.markdown, not st.html: st.html strips inline SVG, which the mark and the GitHub icon need.
    markup = f"""
<div class="topbar">
  <div class="brand">
    <div class="mark">
      <svg viewBox="0 0 40 40" aria-hidden="true">
        <path d="M6 31 C12 31 13 10 20 10 C27 10 28 31 34 31" fill="none" stroke="{MINT}" stroke-width="2.8"
              stroke-linecap="round"/>
        <line x1="5" y1="21" x2="35" y2="21" stroke="{GOLD}" stroke-width="2.2" stroke-dasharray="3.5 2.5"/>
      </svg>
    </div>
    <div>
      <div class="wm">GradMit</div>
      <div class="tag">Where does a graduate application profile stand? &middot; <b>a reproducible ML study</b></div>
    </div>
  </div>
  <div class="who">
    <div class="badge">JIIT Noida</div>
    <div class="t">Saanvi Grover</div>
    <a class="repo" href="{REPO}" target="_blank" rel="noopener">{_GITHUB}<span>{REPO_LABEL}</span></a>
  </div>
</div>"""
    st.markdown("".join(line.strip() for line in markup.splitlines()), unsafe_allow_html=True)


def rail_css(colours: list[str], active: int) -> str:
    rules = []
    for i, c in enumerate(colours):
        rules.append(
            f"div.st-key-nav{i} button:hover {{ border-color:{c}aa !important; box-shadow:0 6px 20px -9px {c}99 !important; }}"
            f"div.st-key-nav{i} button:before {{ background:linear-gradient(90deg,{c},{c}22); }}"
        )
        if i == active:
            rules.append(
                f"div.st-key-nav{i} button {{ background:{c}1f !important; border-color:{c} !important;"
                f" box-shadow:0 0 0 1px {c}55, 0 10px 28px -14px {c} !important; }}"
                f"div.st-key-nav{i} button strong {{ color:{c} !important; }}"
                f"div.st-key-nav{i} button:before {{ height:3px; }}"
            )
    return "<style>" + "".join(rules) + "</style>"


def hero(kicker: str, lede: str) -> None:
    st.html(f'<div class="kicker">{_esc(kicker)}</div>')
    if lede:
        st.html(f'<p class="lede">{lede}</p>')


def tiles(items: list[tuple[str, str, str, str]]) -> None:
    """Stat tiles: (label, value, note, colour)."""
    cells = "".join(
        f'<div class="tile" style="--c:{c}"><div class="l">{_esc(label)}</div>'
        f'<div class="v">{_esc(value)}</div><div class="n">{_esc(note)}</div></div>'
        for label, value, note, c in items
    )
    st.html(f'<div class="tiles">{cells}</div>')


def eyebrow(title: str, sub: str = "") -> None:
    st.html(f'<div class="eyebrow"><h3>{_esc(title)}</h3><span>{_esc(sub)}</span></div>')


def panel(title: str, body: str, colour: str = MINT) -> None:
    st.html(f'<div class="panel" style="--c:{colour}"><h4>{_esc(title)}</h4><p>{body}</p></div>')


def verdict(body: str) -> None:
    st.html(f'<div class="verdict">{body}</div>')


_CARD = itertools.count()


def card(title: str | None = None, sub: str | None = None):
    """A dark card container; use as ``with style.card("Title"):``."""
    box = st.container(key=f"card{next(_CARD)}")
    if title:
        box.html(f'<div class="cardtitle">{_esc(title)}</div>' + (f'<div class="cardsub">{_esc(sub)}</div>' if sub else ""))
    return box


def file_size(path: Path) -> str:
    if not path.exists():
        return "missing"
    n = path.stat().st_size
    for unit, div in (("MB", 1_048_576), ("KB", 1024)):
        if n >= div:
            return f"{n / div:.1f} {unit}"
    return f"{n} B"


def method(steps: list[tuple[str, str, str]], engine: str, ins: list[tuple[str, Path]],
           outs: list[tuple[str, Path]], accent: str = MINT) -> None:
    """The project pipeline as a vertical flowchart, beside the files it reads and writes."""
    left, right = st.columns([1.35, 1], gap="large")
    n = len(steps)
    with left:
        rows = []
        for i, (title, body, chip) in enumerate(steps):
            c = GOLD if i == n - 1 else accent
            rows.append(
                f'<div class="step" style="--c:{c}"><div class="node">{i + 1:02d}</div>'
                f'<div class="sbody"><h5>{_esc(title)}</h5><p>{body}</p>'
                + (f'<span class="chip">{_esc(chip)}</span>' if chip else "")
                + "</div></div>"
            )
        st.html('<div class="flow">' + "".join(rows) + "</div>")

    def block(kind: str, items: list[tuple[str, Path]], c: str) -> str:
        body = "".join(
            f'<div class="io-row"><span class="io-n">{_esc(label)}</span><span class="io-s">{file_size(p)}</span></div>'
            for label, p in items
        )
        return f'<div class="io-k" style="--c:{c}">{kind}</div><div class="io-list">{body}</div>'

    with right:
        st.html(
            '<div class="iopanel"><div class="io-h">Files in and out</div>'
            f'{block("Reads", ins, SKY)}'
            f'<div class="io-arrow" style="--c:{accent}">&#8595;</div>'
            f'<div class="io-engine" style="--c:{accent}">{_esc(engine)}</div>'
            f'<div class="io-arrow" style="--c:{GOLD}">&#8595;</div>'
            f'{block("Writes", outs, GOLD)}'
            '<div class="io-f">Sizes are read from disk when the page loads.</div></div>'
        )


def footer() -> None:
    st.html(
        '<div class="foot"><span>GradMit estimates a proxy label derived from the dataset&rsquo;s own '
        "Chance of Admit score. It does not predict real admission decisions.</span>"
        f'<span>Built by Saanvi Grover, JIIT Noida &middot; <a href="{REPO}" target="_blank" rel="noopener">source</a></span></div>'
    )
