"""
FRAUD SHIELD – Bàn làm việc chống gian lận thanh toán số (bản Streamlit)
Festival Chuyển đổi số 2026 · Bảng B · Digital Payment Fraud Shield

Chạy trên máy:   streamlit run streamlit_app.py
Triển khai:      Streamlit Community Cloud – chọn tệp chính là streamlit_app.py
Lõi mô hình:     fraud_core.py (giống hệt pipeline trong hồ sơ dự thi)
"""
import hashlib
import html
import io
import json
import re
import sqlite3
import tempfile
import time
import zipfile
from contextlib import closing
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import precision_recall_curve

import fraud_core as F

# ----------------------------------------------------------------------------- cấu hình chung
TEAM_NAME = ""  # điền tên đội, ví dụ "Đội ABC" – để trống thì không hiển thị
BASE = Path(__file__).parent
SAMPLE_DIR = BASE / "data"
VN_TZ = timezone(timedelta(hours=7))
LEVELS = ["CRITICAL", "HIGH", "MEDIUM", "LOW"]
LEVEL_COLOR = {"CRITICAL": "#C92A2A", "HIGH": "#E8590C", "MEDIUM": "#D99A00", "LOW": "#98A2B3"}
NAVY, BLUE, BLUE_SOFT, INK, MUTED, GRID = "#0B1B33", "#2557D6", "#BFD0F7", "#0F172A", "#5B6577", "#EDF1F7"
DECISIONS = ["Xác nhận gian lận", "Báo động giả", "Cần thêm thông tin"]
STATUSES = ["Chưa xử lý", "Đang xác minh", "Xác nhận gian lận", "Báo động giả"]
STATUS_COLOR = {"Chưa xử lý": "#D99A00", "Đang xác minh": "#7C5CD6", "Xác nhận gian lận": "#C92A2A",
                "Báo động giả": "#2B8A3E"}
DONE = ("Xác nhận gian lận", "Báo động giả")
LEVEL_INFO = {"CRITICAL": ("Khẩn cấp", "Vượt ngưỡng và có chuỗi thất bại→thành công hoặc dồn dập. Tạm dừng, gọi xác minh ngay"),
              "HIGH": ("Cao", "Vượt ngưỡng cảnh báo. Kiểm tra trong ngày, xác thực lại khách"),
              "MEDIUM": ("Theo dõi", "Từ 60% ngưỡng đến dưới ngưỡng. Không chặn, đưa vào danh sách theo dõi"),
              "LOW": ("Bình thường", "Dưới 60% ngưỡng. Không can thiệp")}
DEC2STATUS = {"Xác nhận gian lận": "Xác nhận gian lận", "Xác nhận rủi ro": "Xác nhận gian lận",
              "Báo động giả": "Báo động giả", "Cần thêm thông tin": "Đang xác minh"}
DB_PATH = Path(tempfile.gettempdir()) / "fraudshield_decisions.sqlite"
LOG_COLS = ["transaction_id", "case_id", "decision", "note", "user", "time", "model", "threshold"]

st.set_page_config(page_title="Fraud Shield · Chống gian lận thanh toán số", page_icon="🛡️",
                   layout="wide", initial_sidebar_state="collapsed")

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Be+Vietnam+Pro:wght@400;500;600;700;800&display=swap');
:root{--navy:#0B1B33;--navy2:#16325F;--blue:#2557D6;--blue-soft:#E8EEFC;--ink:#0F172A;--muted:#5B6577;
--line:#E3E8F0;--bg:#F4F6FA;--card:#FFFFFF;--crit:#C92A2A;--high:#E8590C;--med:#D99A00;--ok:#2B8A3E}
html,body,.stApp,.stApp p,.stApp li,.stApp label,.stApp input,.stApp textarea,.stApp button,.stApp h1,.stApp h2,
.stApp h3,.stApp h4,.stMarkdown,.fs,.fs div,.fs span,.fs td,.fs th{font-family:'Be Vietnam Pro',system-ui,-apple-system,'Segoe UI',sans-serif}
.stApp{background:#F4F6FA}
header[data-testid="stHeader"]{background:transparent}
footer{visibility:hidden}
.block-container,[data-testid="stMainBlockContainer"]{max-width:1320px;padding-top:1.6rem;padding-bottom:3rem}
/* thẻ nội dung */
[class*="st-key-card"]{box-sizing:border-box;background:var(--card);border:1px solid var(--line);border-radius:16px;padding:18px 20px 14px;
box-shadow:0 1px 2px rgba(15,23,42,.04)}
/* tabs */
.stTabs [data-baseweb="tab-list"]{gap:4px;border-bottom:1px solid var(--line);flex-wrap:wrap}
.stTabs [data-baseweb="tab"]{padding:10px 16px;font-weight:600;border-radius:10px 10px 0 0}
.stTabs [data-baseweb="tab"] p{font-size:14.5px;font-weight:600}
.stTabs [aria-selected="true"] p{color:var(--blue)}
/* nút */
.stButton button,.stDownloadButton button,.stFormSubmitButton button{border-radius:10px;font-weight:600;min-height:42px}
/* hero */
.fs-hero{background:radial-gradient(1200px 400px at 85% -20%,rgba(59,130,246,.45),transparent 60%),
linear-gradient(135deg,#0B1B33 0%,#13294B 55%,#1C3F86 100%);color:#fff;border-radius:20px;padding:30px 34px 28px;
position:relative;overflow:hidden;margin-bottom:18px}
.fs-hero .brand{display:flex;align-items:center;gap:14px}
.fs-logo{width:46px;height:46px;border-radius:12px;background:linear-gradient(135deg,#4F8BFF,#2557D6);display:flex;
align-items:center;justify-content:center;font-weight:800;font-size:17px;color:#fff;box-shadow:0 6px 18px rgba(37,87,214,.45);flex:none}
.fs-hero .t1{font-size:30px;font-weight:800;letter-spacing:-.02em;line-height:1.15}
.fs-hero .t2{font-size:15px;opacity:.82;margin-top:2px}
.fs-hero .lead{font-size:15.5px;line-height:1.6;opacity:.92;max-width:760px;margin:18px 0 16px}
.fs-chip{display:inline-block;padding:5px 12px;border-radius:999px;background:rgba(255,255,255,.1);
border:1px solid rgba(255,255,255,.18);font-size:12.5px;font-weight:500;margin:0 6px 6px 0;color:#fff}
.fs-hero .event{position:absolute;right:28px;top:26px;font-size:12px;opacity:.7;text-align:right}
/* thanh trên cùng khi đã có kết quả */
.fs-top{display:flex;align-items:center;gap:14px;background:linear-gradient(135deg,#0B1B33,#16325F);color:#fff;
border-radius:16px;padding:14px 18px}
.fs-top .name{font-weight:800;font-size:18px;line-height:1.2}
.fs-top .meta{font-size:12.5px;opacity:.78;margin-top:2px}
/* KPI */
.fs-kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(215px,1fr));gap:14px;margin:4px 0 14px}
.fs-kpis.sm{grid-template-columns:repeat(auto-fit,minmax(120px,1fr));gap:10px}
.fs-kpis.sm .fs-kpi{padding:12px 14px}.fs-kpis.sm .val{font-size:25px}
.fs-kpi{background:var(--card);border:1px solid var(--line);border-radius:16px;padding:16px 18px;
box-shadow:0 1px 2px rgba(15,23,42,.04);position:relative}
.fs-kpi .lab{font-size:12.5px;color:var(--muted);font-weight:600;letter-spacing:.01em}
.fs-kpi .val{font-size:31px;font-weight:700;color:var(--ink);letter-spacing:-.02em;line-height:1.15;margin-top:8px;
font-variant-numeric:tabular-nums}
.fs-kpi .sub{font-size:12.5px;color:var(--muted);margin-top:6px;line-height:1.45}
.fs-kpi.accent{border-left:4px solid var(--crit)}
.fs-kpi.accent .val{color:var(--crit)}
.fs-kpi.dark{background:linear-gradient(160deg,#0B1B33,#16325F);border-color:#1E3A66}
.fs-kpi.dark .lab,.fs-kpi.dark .sub{color:rgba(255,255,255,.72)}
.fs-kpi.dark .val{color:#fff}
.fs-pill{display:inline-block;padding:2px 8px;border-radius:999px;font-size:11px;font-weight:700;margin-left:4px}
.fs-pill.ok{background:rgba(43,138,62,.16);color:#2B8A3E}
.fs-kpi.dark .fs-pill.ok{background:rgba(110,231,140,.18);color:#8CE99A}
/* tiêu đề mục */
.fs-h{font-size:16.5px;font-weight:700;color:var(--ink);margin:2px 0 2px}
.fs-sub{font-size:13px;color:var(--muted);margin-bottom:10px;line-height:1.5}
.fs-sec{font-size:21px;font-weight:800;color:var(--ink);letter-spacing:-.01em;margin:8px 0 2px}
/* huy hiệu mức rủi ro */
.fs-badge{display:inline-block;padding:2px 10px;border-radius:999px;font-size:11.5px;font-weight:700;letter-spacing:.03em;white-space:nowrap}
.b-CRITICAL{background:#FDE8E8;color:#B42318}.b-HIGH{background:#FFEDD5;color:#C2410C}
.b-MEDIUM{background:#FEF3C7;color:#A16207}.b-LOW{background:#EEF1F5;color:#5B6577}
.b-ĐẠT{background:#E6F4EA;color:#2B8A3E}.b-CẢNH{background:#FEF3C7;color:#A16207}
.b-LỖI{background:#FDE8E8;color:#B42318}.b-THIẾU{background:#EEF1F5;color:#5B6577}
/* bảng HTML */
.fs-tw{overflow-x:auto;border:1px solid var(--line);border-radius:12px}
.fs-table{width:100%;border-collapse:collapse;font-size:13px;background:#fff}
.fs-table th{background:#F3F6FB;color:#475467;font-weight:600;text-align:left;padding:9px 12px;border-bottom:1px solid var(--line);white-space:nowrap}
.fs-table td{padding:9px 12px;border-bottom:1px solid #EEF1F5;color:#1F2937;vertical-align:top}
.fs-table tr:last-child td{border-bottom:none}
.fs-table .num{text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap}
.fs-table tr.hl td{font-weight:700;background:#F3F7FF}
.fs-table .best{color:var(--blue);font-weight:700}
.fs-table .new{color:var(--high);font-weight:700}
/* mức rủi ro – thanh */
.fs-lv{display:flex;align-items:center;gap:10px;margin:9px 0}
.fs-lv .b{width:88px;flex:none}
.fs-lv .bar{flex:1;height:10px;background:#EEF1F5;border-radius:999px;overflow:hidden}
.fs-lv .bar i{display:block;height:100%;border-radius:999px}
.fs-lv .n{width:118px;text-align:right;font-size:13px;color:#1F2937;font-variant-numeric:tabular-nums;flex:none}
/* case / giao dịch */
.fs-case{display:flex;flex-wrap:wrap;gap:22px;align-items:flex-start}
.fs-case .item .k{font-size:12px;color:var(--muted);font-weight:600}
.fs-case .item .v{font-size:16px;font-weight:700;color:var(--ink);margin-top:3px;font-variant-numeric:tabular-nums}
.fs-reason{background:#F6F8FC;border:1px solid var(--line);border-left:4px solid var(--blue);border-radius:10px;
padding:11px 14px;font-size:13.5px;line-height:1.55;color:#1F2937;margin-top:14px}
.fs-flag{display:inline-block;padding:5px 11px;border-radius:9px;font-size:12.5px;margin:0 6px 6px 0;border:1px solid var(--line);background:#fff}
.fs-flag.on{background:#FFF4ED;border-color:#FCC9A8;color:#9A3412;font-weight:600}
.fs-flag.off{color:#98A2B3}
.fs-kv{display:grid;grid-template-columns:repeat(auto-fit,minmax(230px,1fr));gap:0 22px}
.fs-kv div{display:flex;justify-content:space-between;gap:12px;border-bottom:1px dashed #E6EAF1;padding:7px 0;font-size:13px}
.fs-kv span{color:var(--muted)}
.fs-kv b{font-weight:600;color:#111827;text-align:right}
.fs-step{background:#fff;border:1px solid var(--line);border-radius:16px;padding:16px 18px}
.fs-step .no{width:28px;height:28px;border-radius:8px;background:var(--blue-soft);color:var(--blue);font-weight:800;
display:flex;align-items:center;justify-content:center;font-size:13px}
.fs-step .tt{font-weight:700;margin:10px 0 4px;color:var(--ink);font-size:14.5px}
.fs-step .dd{font-size:13px;color:var(--muted);line-height:1.5}
.fs-steps{display:grid;grid-template-columns:repeat(auto-fit,minmax(230px,1fr));gap:14px;margin-top:6px}
.fs-foot{margin-top:26px;padding-top:14px;border-top:1px solid var(--line);font-size:12px;color:var(--muted);
display:flex;justify-content:space-between;flex-wrap:wrap;gap:8px}
.fs-prog{display:flex;height:14px;border-radius:999px;overflow:hidden;background:#EEF1F5;margin:6px 0 8px}
.fs-prog i{display:block;height:100%}
.fs-legend{display:flex;flex-wrap:wrap;gap:14px;font-size:12.5px;color:#334155}
.fs-legend span{display:inline-flex;align-items:center;gap:6px}
.fs-legend b{font-weight:700;font-variant-numeric:tabular-nums}
.fs-dot{width:10px;height:10px;border-radius:3px;display:inline-block}
.fs-kpi.warn{border-left:4px solid var(--med)}
.fs-kpi.warn .val{color:#B07800}
.fs-kpi.good{border-left:4px solid var(--ok)}
.fs-side{display:flex;align-items:center;gap:10px;margin:2px 0 6px}
.fs-side .fs-logo{width:36px;height:36px;font-size:14px;border-radius:10px}
.fs-side .nm{font-weight:800;font-size:16px;color:var(--ink);line-height:1.15}
.fs-side .sb{font-size:12px;color:var(--muted)}
.stApp{background:#F4F6FA}
[data-baseweb="tag"]{background:#E8EEFC!important}
[data-baseweb="tag"] span,[data-baseweb="tag"] svg{color:#1E3A8A!important;fill:#1E3A8A!important}
button[data-testid="stBaseButton-primary"],button[data-testid="stBaseButton-primaryFormSubmit"]{background:#2557D6!important;border-color:#2557D6!important;color:#fff!important}
button[data-testid="stBaseButton-pillsActive"]{background:#E8EEFC!important;border-color:#2557D6!important;color:#1E3A8A!important}
.stTabs [data-baseweb="tab-highlight"]{background-color:#2557D6!important}
.fs-head{display:flex;align-items:center;gap:14px;margin:0 0 6px}
.fs-head .t{font-size:24px;font-weight:800;color:var(--ink);letter-spacing:-.02em;line-height:1.15}
.fs-head .m{font-size:13px;color:var(--muted);margin-top:3px}
.fs-tiers{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px;margin:4px 0 14px}
.fs-tier{background:#fff;border:1px solid var(--line);border-top:5px solid var(--c);border-radius:14px;padding:14px 16px}
.fs-tier.off{opacity:.45}
.fs-tier .top{display:flex;justify-content:space-between;align-items:center}
.fs-tier .lv{font-weight:800;font-size:13px;color:var(--c);letter-spacing:.03em}
.fs-tier .vn{font-size:12px;color:var(--muted);font-weight:600}
.fs-tier .num{font-size:28px;font-weight:700;color:var(--ink);margin-top:6px;font-variant-numeric:tabular-nums}
.fs-tier .pc{font-size:12.5px;color:var(--muted)}
.fs-tier .act{font-size:12.5px;color:#334155;margin-top:8px;line-height:1.45;border-top:1px dashed #E3E8F0;padding-top:8px}
.fs-tier .open{display:inline-block;margin-top:6px;font-size:12px;font-weight:700;color:var(--c)}
.fs-kpis.six{grid-template-columns:repeat(auto-fit,minmax(170px,1fr))}
.fs-kpis.six .val{font-size:27px}
@media (max-width:900px){.fs-tiers{grid-template-columns:repeat(2,minmax(0,1fr))}}
@media (max-width:640px){.fs-hero{padding:22px 20px}.fs-hero .t1{font-size:24px}.fs-hero .event{position:static;text-align:left;margin-top:10px}
.fs-kpi .val{font-size:26px}.fs-lv .n{width:96px}}
</style>
"""
st.markdown(re.sub(r"\s*\n\s*", " ", CSS), unsafe_allow_html=True)
if not st.get_option("theme.primaryColor"):
    st.warning("Thiếu tệp .streamlit/config.toml trong repo nên màu giao diện chưa đúng – hãy tải cả thư mục "
               ".streamlit lên GitHub (xem README).", icon=":material/palette:")


# ----------------------------------------------------------------------------- tiện ích hiển thị
def H(s):
    """Hiển thị HTML (gộp dòng để Markdown không biến phần thụt lề thành khối code)."""
    st.markdown(re.sub(r"\s*\n\s*", " ", s), unsafe_allow_html=True)


def e(x):
    return html.escape("" if x is None else str(x))


def n0(x):
    return f"{int(round(float(x))):,}".replace(",", ".")


def d(x, k=3):
    return f"{float(x):.{k}f}".replace(".", ",")


def pct(x, k=1):
    return f"{float(x) * 100:.{k}f}%".replace(".", ",")


def money(x):
    return n0(x) + " ₫"


def dt(x):
    return pd.Timestamp(x).strftime("%d/%m/%Y %H:%M")


def badge(level):
    lv = str(level)
    cls = lv.split()[0] if lv else "LOW"
    return f'<span class="fs-badge b-{e(cls)}">{e(lv)}</span>'


def kpi(label, value, sub="", kind=""):
    return (f'<div class="fs-kpi {kind}"><div class="lab">{label}</div><div class="val">{value}</div>'
            f'<div class="sub">{sub}</div></div>')


def kpis(cards, small=False):
    H(f'<div class="fs fs-kpis{" sm" if small else ""}">' + "".join(cards) + "</div>")


def title(text, sub=None):
    H(f'<div class="fs-h">{text}</div>' + (f'<div class="fs-sub">{sub}</div>' if sub else ""))


def table(rows, cols, num_cols=(), hl_last=False, best=None, raw_cols=()):
    """rows: list[dict]; cols: list[(key, header)]."""
    head = "".join(f'<th class="{"num" if k in num_cols else ""}">{e(h)}</th>' for k, h in cols)
    body = []
    for i, r in enumerate(rows):
        cls = ' class="hl"' if hl_last and i == len(rows) - 1 else ""
        tds = []
        for k, _ in cols:
            v = r.get(k, "")
            txt = v if k in raw_cols else e(v)
            c = ["num"] if k in num_cols else []
            if best and best.get(k) == i:
                c.append("best")
            tds.append(f'<td class="{" ".join(c)}">{txt}</td>')
        body.append(f"<tr{cls}>" + "".join(tds) + "</tr>")
    H(f'<div class="fs fs-tw"><table class="fs-table"><thead><tr>{head}</tr></thead><tbody>{"".join(body)}</tbody></table></div>')


def card(key):
    return st.container(border=False, key=f"card_{key}")


def style(fig, h=320, legend=True, **kw):
    kw.setdefault("bargap", 0.28)
    fig.update_layout(
        height=h, margin=dict(l=6, r=10, t=30 if legend else 8, b=6),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", separators=",.",
        font=dict(family="Be Vietnam Pro, system-ui, sans-serif", size=12, color="#334155"),
        hoverlabel=dict(bgcolor=NAVY, bordercolor=NAVY, font=dict(color="#fff", family="Be Vietnam Pro, sans-serif")),
        showlegend=legend,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, bgcolor="rgba(0,0,0,0)", font=dict(size=11.5)),
        **kw)
    fig.update_xaxes(gridcolor=GRID, zeroline=False, linecolor="#D5DCE6", tickfont=dict(size=11.5), automargin=True)
    fig.update_yaxes(gridcolor=GRID, zeroline=False, linecolor="#D5DCE6", tickfont=dict(size=11.5), automargin=True)
    return fig


def chart(fig, key=None):
    st.plotly_chart(fig, theme=None, key=key, config={"displayModeBar": False, "responsive": True})


def grid(df, **kw):
    return st.dataframe(df, hide_index=True, width="stretch", **kw)


def money_short(x):
    x = float(x)
    if x >= 1e9:
        return f"{x / 1e9:.2f} tỷ ₫".replace(".", ",")
    if x >= 1e6:
        return f"{x / 1e6:.1f} triệu ₫".replace(".", ",")
    return money(x)


# ----------------------------------------------------------------------------- lưu quyết định xử lý (SQLite)
def _db():
    con = sqlite3.connect(DB_PATH, timeout=10)
    con.execute("CREATE TABLE IF NOT EXISTS decisions (ds TEXT, transaction_id TEXT, case_id TEXT, decision TEXT, "
                "note TEXT, user TEXT, time TEXT, model TEXT, threshold REAL)")
    return con


def save_decisions(ds, rows):
    with closing(_db()) as con:
        con.executemany("INSERT INTO decisions VALUES (?,?,?,?,?,?,?,?,?)",
                        [(ds, r["transaction_id"], r.get("case_id") or "", r["decision"], r.get("note") or "",
                          r.get("user") or "chuyên viên", r["time"], F.VERSION, float(r["threshold"])) for r in rows])
        con.commit()


def load_log(ds):
    try:
        with closing(_db()) as con:
            return pd.read_sql_query("SELECT transaction_id, case_id, decision, note, user, time, model, threshold "
                                     "FROM decisions WHERE ds = ? ORDER BY rowid", con, params=(ds,))
    except Exception:  # noqa
        return pd.DataFrame(columns=LOG_COLS)


def clear_decisions(ds):
    with closing(_db()) as con:
        con.execute("DELETE FROM decisions WHERE ds = ?", (ds,))
        con.commit()


def now_vn():
    return datetime.now(VN_TZ).strftime("%d/%m/%Y %H:%M:%S")


def status_frame(res):
    """Mọi giao dịch đáng ngờ (HIGH + CRITICAL) kèm trạng thái xử lý.
    Đã xử lý = có kết luận: nhãn đã xác minh trong Data Pack hoặc quyết định chuyên viên ghi trong ứng dụng."""
    df = res["df"]
    a = df[df.predicted_fraud == 1].copy()
    a["status"] = "Chưa xử lý"
    a["source"] = ""
    a["handler"] = ""
    a["handled_at"] = ""
    a["note"] = ""
    lv = res.get("lab_view")
    if lv is not None:
        m = a.transaction_id.map(dict(zip(lv.transaction_id, lv.y)))
        a.loc[m == 1, "status"] = "Xác nhận gian lận"
        a.loc[m == 0, "status"] = "Báo động giả"
        a.loc[m.notna(), "source"] = "Nhãn BTC (đã xác minh)"
    log = load_log(res["ds"])
    if len(log):
        last = log.drop_duplicates("transaction_id", keep="last").set_index("transaction_id")
        hit = a.transaction_id.isin(last.index)
        ids = a.loc[hit, "transaction_id"]
        a.loc[hit, "status"] = ids.map(last["decision"]).map(DEC2STATUS).fillna("Đang xác minh").values
        a.loc[hit, "source"] = "Chuyên viên"
        a.loc[hit, "handler"] = ids.map(last["user"]).values
        a.loc[hit, "handled_at"] = ids.map(last["time"]).values
        a.loc[hit, "note"] = ids.map(last["note"]).values
    return a


# ----------------------------------------------------------------------------- chạy mô hình (có cache)
@st.cache_data(show_spinner=False, max_entries=8, ttl=6 * 3600)
def engine(files: dict):
    t0 = time.perf_counter()
    tables, checks, ok = F.validate(files)
    if not ok:
        return {"ok": False, "checks": checks}
    res = F.run(files, tables=tables)
    res.update(ok=True, checks=checks, seconds=round(time.perf_counter() - t0, 1))
    return res


@st.cache_data(show_spinner=False)
def sample_files():
    return F.collect_dir(SAMPLE_DIR)


def store_result(res, files, notes, source):
    checks = [{"group": "Tệp tải lên", "check": n, "status": "CẢNH BÁO", "detail": ""} for n in notes] + res["checks"]
    ds = hashlib.sha1(files.get("transactions.csv", b"")).hexdigest()[:16]
    st.session_state.res = {**res, "checks": checks, "source": source, "ds": ds,
                            "started": datetime.now(VN_TZ).strftime("%H:%M · %d/%m/%Y"),
                            "rid": f"{abs(hash((source, time.time()))) % 10**8:08d}"}
    st.session_state.pop("failed", None)


def start_run(files, notes, source):
    with st.status("Đang chạy Fraud Shield…", expanded=True) as s:
        st.write("Kiểm tra cấu trúc tệp, đối chiếu manifest và SHA-256")
        st.write("Dựng hồ sơ hành vi, chấm điểm Rule – Logistic Regression – Isolation Forest")
        st.write("Huấn luyện/đánh giá 5-fold (nếu có nhãn), xếp hạng và gom case")
        try:
            res = engine(files)
        except Exception as ex:  # noqa
            s.update(label="Lỗi khi chạy mô hình", state="error")
            st.error(f"Lỗi khi chạy mô hình: {ex}")
            return
        if not res["ok"]:
            s.update(label="Dữ liệu chưa đạt các kiểm tra bắt buộc", state="error")
            st.session_state.failed = {"checks": [*({"group": "Tệp tải lên", "check": n, "status": "CẢNH BÁO",
                                                      "detail": ""} for n in notes), *res["checks"]]}
            return
        s.update(label="Hoàn tất", state="complete")
    store_result(res, files, notes, source)
    st.rerun()


def auto_load_sample():
    """Mở link là có Dashboard ngay: tự chạy Data Pack BTC có sẵn (kết quả được cache cho mọi người xem)."""
    if not (SAMPLE_DIR / "transactions.csv").exists():
        return False
    with st.spinner("Đang chấm điểm Data Pack BTC… (lần đầu mất vài giây)"):
        files = sample_files()
        res = engine(files)
    if not res.get("ok"):
        return False
    store_result(res, files, [], "Data Pack BTC (có sẵn trong ứng dụng)")
    return True


# ----------------------------------------------------------------------------- màn hình bắt đầu
def landing():
    team = f" · {e(TEAM_NAME)}" if TEAM_NAME else ""
    H(f"""
    <div class="fs fs-hero">
      <div class="event">Festival Chuyển đổi số 2026<br>Bảng B · Digital Payment Fraud Shield{team}</div>
      <div class="brand"><div class="fs-logo">FS</div>
        <div><div class="t1">Fraud Shield</div><div class="t2">Bàn làm việc chống gian lận thanh toán số</div></div></div>
      <div class="lead">Chấm điểm rủi ro từng giao dịch, giải thích bằng lời vì sao bị cảnh báo và gom các cảnh báo
        liên quan thành case để chuyên viên xử lý theo thứ tự ưu tiên. Toàn bộ chạy trực tiếp trên Data Pack của
        Ban tổ chức.</div>
      <span class="fs-chip">Hồ sơ hành vi từng khách</span><span class="fs-chip">Rule có giải thích</span>
      <span class="fs-chip">Logistic Regression</span><span class="fs-chip">Isolation Forest</span>
      <span class="fs-chip">Gom case · nhật ký quyết định</span>
    </div>""")

    c1, c2 = st.columns(2, gap="medium")
    with c1, card("start_sample"):
        title("Chạy ngay với Data Pack của BTC",
              "Bộ dữ liệu chính thức đã có sẵn trong ứng dụng: 8.000 giao dịch (01–06/2026), 2.000 nhãn công khai, "
              "kèm manifest SHA-256 và answer key mẫu. Mô hình huấn luyện và đánh giá lại từ đầu trong vài giây.")
        sample_ok = SAMPLE_DIR.exists() and (SAMPLE_DIR / "transactions.csv").exists()
        if st.button("Chạy với Data Pack BTC", type="primary", icon=":material/play_arrow:",
                     disabled=not sample_ok, width="stretch", key="btn_sample"):
            start_run(sample_files(), [], "Data Pack BTC (có sẵn trong ứng dụng)")
        if not sample_ok:
            st.caption("Bản triển khai này không kèm Data Pack mẫu – hãy tải dữ liệu lên ở khung bên cạnh.")
    with c2, card("start_upload"):
        title("Kiểm tra Data Pack của bạn",
              "Tải lên tệp .zip của BTC hoặc chọn nhiều tệp .csv cùng lúc. Bắt buộc: transactions, customers, "
              "merchants, devices. Có fraud_labels_sample thì huấn luyện + đánh giá; không có thì chấm điểm "
              "bằng mô hình học sẵn.")
        ups = st.file_uploader("Data Pack (.zip hoặc .csv)", type=["zip", "csv"], accept_multiple_files=True,
                               label_visibility="collapsed", key="uploader")
        if st.button("Kiểm tra & chạy dữ liệu tải lên", icon=":material/upload_file:", disabled=not ups,
                     width="stretch", key="btn_upload"):
            files, notes = F.collect_files([(u.name, u.getvalue()) for u in ups])
            src = ", ".join(u.name for u in ups) if len(ups) <= 2 else f"{len(ups)} tệp CSV tải lên"
            start_run(files, notes, src[:160])

    if "failed" in st.session_state:
        st.error("Dữ liệu chưa đạt các kiểm tra bắt buộc – xem chi tiết bên dưới.")
        checks_table(st.session_state.failed["checks"])

    H("""<div class="fs-sec" style="margin-top:22px">Ứng dụng làm gì</div>
    <div class="fs fs-steps">
      <div class="fs-step"><div class="no">1</div><div class="tt">Kiểm tra dữ liệu</div><div class="dd">Đủ tệp, đủ cột,
        đối chiếu số dòng – số cột – SHA-256 với manifest của BTC, kiểm tra khoá liên kết giữa các bảng.</div></div>
      <div class="fs-step"><div class="no">2</div><div class="tt">Hồ sơ hành vi</div><div class="dd">So mỗi giao dịch với
        thói quen của chính khách hàng đó: thiết bị, tỉnh, IP, kênh, mức tiền (leave-one-out, không dùng nhãn).</div></div>
      <div class="fs-step"><div class="no">3</div><div class="tt">Chấm điểm 3 tầng</div><div class="dd">risk = 0,6·ML +
        0,3·Rule + 0,1·Bất thường. Ngưỡng chọn bằng 5-fold lồng nhau, không nhìn tập kiểm tra.</div></div>
      <div class="fs-step"><div class="no">4</div><div class="tt">Case & giải thích</div><div class="dd">Mức LOW → CRITICAL,
        lý do bằng tiếng Việt, gom cảnh báo cùng khách trong 30 phút thành case, ghi nhật ký quyết định.</div></div>
    </div>""")


# ----------------------------------------------------------------------------- bảng kiểm tra dữ liệu
def checks_table(checks):
    rows = [{"group": c["group"], "check": c["check"], "status": badge(c["status"]), "detail": c.get("detail", "")}
            for c in checks]
    table(rows, [("group", "Nhóm"), ("check", "Hạng mục"), ("status", "Kết quả"), ("detail", "Chi tiết")],
          raw_cols=("status",))


# ----------------------------------------------------------------------------- các tab kết quả
def model_summary(res):
    df, rep = res["df"], res["report"]
    thr, n = rep["threshold"], len(df)
    if res["mode"] == "train":
        b = rep["comparison_nested_cv"]["Gộp 3 tầng – có hồ sơ (B, chính thức)"]
        ok = '<span class="fs-pill ok">đạt</span>'
        kpis([
            kpi("F1-score", d(b["f1"]), f"Mục tiêu ≥ 0,75 {ok if b['f1'] >= .75 else ''}", "dark"),
            kpi("Recall", d(b["recall"]), f"{b['tp']}/{b['tp'] + b['fn']} gian lận được bắt · mục tiêu ≥ 0,80 "
                f"{ok if b['recall'] >= .8 else ''}", "dark"),
            kpi("Precision", d(b["precision"]), f"{b['fp']} báo nhầm trên {n0(rep['model_info']['n_labels'] - rep['model_info']['n_fraud'])} giao dịch bình thường", "dark"),
            kpi("PR-AUC", d(b["pr_auc"]), "Đánh giá 5-fold phân tầng, ngưỡng lồng – không rò rỉ nhãn", "dark"),
        ])
    else:
        mi = rep["model_info"]
        st.info(f"Data Pack này không có nhãn nên được chấm điểm bằng mô hình học sẵn ({mi['source']}). "
                "Các chỉ số F1/Recall chỉ có khi tải kèm fraud_labels_sample.csv.", icon=":material/info:")

    c1, c2 = st.columns(2, gap="medium")
    with c1, card("levels"):
        title("Phân bố mức rủi ro", f"Trên toàn bộ {n0(n)} giao dịch")
        cnt = rep["risk_level_counts"]
        mx = max(cnt.get(l, 0) for l in LEVELS if l != "LOW") or 1
        rows = []
        for l in LEVELS:
            c = cnt.get(l, 0)
            w = 100 if l == "LOW" else c / mx * 100
            rows.append(f'<div class="fs-lv"><div class="b">{badge(l)}</div><div class="bar"><i style="width:{w:.1f}%;'
                        f'background:{LEVEL_COLOR[l]}"></i></div><div class="n"><b>{n0(c)}</b> · {pct(c / n)}</div></div>')
        H('<div class="fs">' + "".join(rows) + "</div>")
        H(f'<div class="fs-sub" style="margin-top:10px">HIGH/CRITICAL: điểm ≥ ngưỡng {d(thr, 4)}; CRITICAL khi kèm '
          f'chuỗi thất bại→thành công (R5) hoặc ≥ 3 giao dịch/5 phút (R6). MEDIUM: điểm ≥ 60% ngưỡng – theo dõi. '
          f'Thanh LOW không theo tỉ lệ để các mức nhỏ dễ nhìn.</div>')
    with c2, card("hist"):
        title("Phân bố điểm rủi ro", "Trục dọc theo thang log – phần lớn giao dịch có điểm rất thấp")
        s = df.risk_score.values
        fig = go.Figure()
        fig.add_histogram(x=s[s < thr], xbins=dict(start=0, end=1, size=0.02), name="Dưới ngưỡng",
                          marker_color="#B8C2D3", hovertemplate="Điểm %{x}<br>%{y} giao dịch<extra></extra>")
        fig.add_histogram(x=s[s >= thr], xbins=dict(start=0, end=1, size=0.02), name="Cảnh báo",
                          marker_color=LEVEL_COLOR["CRITICAL"], hovertemplate="Điểm %{x}<br>%{y} giao dịch<extra></extra>")
        fig.add_vline(x=thr, line=dict(color=NAVY, dash="dash", width=1.5))
        fig.add_annotation(x=thr, y=1, yref="paper", text=f"Ngưỡng {d(thr, 4)}", showarrow=False, xanchor="right",
                           xshift=-6, yanchor="top", font=dict(size=11.5, color=NAVY))
        style(fig, 330, barmode="overlay", bargap=0.05)
        fig.update_yaxes(type="log", title=None, tickvals=[1, 10, 100, 1000], ticktext=["1", "10", "100", "1.000"])
        fig.update_xaxes(range=[0, 1], dtick=0.1)
        chart(fig, "hist")



# ----------------------------------------------------------------------------- DASHBOARD giám sát
def status_legend(counts, total):
    bar = "".join(f'<i style="width:{counts.get(k, 0) / max(1, total) * 100:.2f}%;background:{STATUS_COLOR[k]}" '
                  f'title="{k}"></i>' for k in STATUSES)
    leg = "".join(f'<span><i class="fs-dot" style="background:{STATUS_COLOR[k]}"></i>{k} <b>{n0(counts.get(k, 0))}</b></span>'
                  for k in STATUSES)
    return f'<div class="fs"><div class="fs-prog">{bar}</div><div class="fs-legend">{leg}</div></div>'


def by_status(frame, key, order):
    t = frame.groupby([key, "status"]).size().unstack(fill_value=0).reindex(order, fill_value=0)
    return t.reindex(columns=STATUSES, fill_value=0)


def stacked(tab, horizontal=False, h=300, key=None, xcat=True, legend=True):
    fig = go.Figure()
    for stt in STATUSES:
        vals = tab[stt].values
        if horizontal:
            fig.add_bar(y=tab.index.astype(str), x=vals, name=stt, orientation="h", marker_color=STATUS_COLOR[stt],
                        hovertemplate="%{y}<br>" + stt + ": %{x}<extra></extra>")
        else:
            fig.add_bar(x=tab.index.astype(str), y=vals, name=stt, marker_color=STATUS_COLOR[stt],
                        hovertemplate="%{x}<br>" + stt + ": %{y}<extra></extra>")
    style(fig, h, barmode="stack", legend=legend)
    fig.update_layout(legend=dict(traceorder="normal"))
    if not horizontal and xcat:
        fig.update_xaxes(type="category", tickangle=0, nticks=10)
    if horizontal:
        fig.update_yaxes(autorange="reversed")
    chart(fig, key)


def hbar(series, key, h=300, color=BLUE, unit="giao dịch"):
    sr = series.sort_values()
    fig = go.Figure(go.Bar(x=sr.values, y=sr.index.astype(str), orientation="h", marker_color=color, text=sr.values,
                           textposition="outside", cliponaxis=False,
                           hovertemplate="%{y}<br>%{x} " + unit + "<extra></extra>"))
    style(fig, h, legend=False)
    fig.update_xaxes(range=[0, max(1, sr.max() if len(sr) else 1) * 1.18])
    chart(fig, key)


def dash_filters(res):
    df, rid = res["df"], res["rid"]
    d0, d1 = df.datetime.min().date(), df.datetime.max().date()
    ch_opts = sorted(df.payment_channel.dropna().astype(str).unique().tolist())
    with card("d_filter"):
        c1, c2, c3 = st.columns([1.15, 2.3, 1.15], vertical_alignment="bottom")
        rng = c1.date_input("Khoảng thời gian", value=(d0, d1), min_value=d0, max_value=d1, format="DD/MM/YYYY",
                            key=f"f_date_{rid}")
        lv = c2.pills("Mức rủi ro", LEVELS, selection_mode="multi", default=LEVELS, key=f"f_lv_{rid}",
                      format_func=lambda l: f"{l} · {LEVEL_INFO[l][0]}")
        ch = c3.selectbox("Kênh thanh toán", ["Tất cả kênh"] + ch_opts, key=f"f_ch_{rid}")
    if isinstance(rng, (list, tuple)):
        start, end = (rng[0], rng[-1]) if len(rng) else (d0, d1)
    else:
        start = end = rng
    return start, end, list(lv or LEVELS), (ch_opts if ch == "Tất cả kênh" else [ch])


def tier_tiles(dff, allst, lv):
    n = max(1, len(dff))
    open_ids = set(allst.loc[allst.status.isin(["Chưa xử lý", "Đang xác minh"]), "transaction_id"])
    tiles = []
    for l in LEVELS:
        sub = dff[dff.risk_level == l]
        extra = ""
        if l in ("CRITICAL", "HIGH"):
            k = int(sub.transaction_id.isin(open_ids).sum())
            extra = f'<div class="open">{n0(k)} chưa xử lý</div>'
        elif l == "MEDIUM":
            extra = '<div class="open">trong danh sách theo dõi</div>'
        tiles.append(f'<div class="fs-tier {"" if l in lv else "off"}" style="--c:{LEVEL_COLOR[l]}"><div class="top">'
                     f'<span class="lv">{l}</span><span class="vn">{LEVEL_INFO[l][0]}</span></div>'
                     f'<div class="num">{n0(len(sub))}</div><div class="pc">{pct(len(sub) / n)} số giao dịch</div>'
                     f'{extra}<div class="act">{LEVEL_INFO[l][1]}</div></div>')
    H('<div class="fs fs-tiers">' + "".join(tiles) + "</div>")


def tab_dashboard(res):
    df, rep, rid = res["df"], res["report"], res["rid"]
    start, end, lv, ch = dash_filters(res)
    dmask = (df.datetime.dt.date >= start) & (df.datetime.dt.date <= end) & df.payment_channel.astype(str).isin(ch)
    dff = df[dmask]
    allst = status_frame(res)
    allst = allst[allst.transaction_id.isin(dff.transaction_id)]
    a = allst[allst.risk_level.isin(lv)]
    n_a = len(a)
    cnt = a.status.value_counts().to_dict()
    n_done = sum(cnt.get(k, 0) for k in DONE)
    n_pend, n_ver = cnt.get("Chưa xử lý", 0), cnt.get("Đang xác minh", 0)
    n_fraud, n_fa = cnt.get("Xác nhận gian lận", 0), cnt.get("Báo động giả", 0)
    open_mask = a.status.isin(["Chưa xử lý", "Đang xác minh"])

    title("Phân tầng rủi ro", "Mọi giao dịch đều được chấm điểm và xếp vào một trong 4 mức. CRITICAL và HIGH vượt ngưỡng "
          f"cảnh báo {d(rep['threshold'], 4)} nên vào hàng đợi xử lý; MEDIUM vào danh sách theo dõi; LOW là bình thường.")
    tier_tiles(dff, allst, lv)

    title("Xử lý cảnh báo", "Đã xử lý = đã có kết luận: nhãn đã xác minh trong Data Pack hoặc quyết định của chuyên viên "
          "ghi trong ứng dụng.")
    if n_a == 0:
        st.info("Bộ lọc hiện tại không có giao dịch CRITICAL/HIGH nào cần xử lý.", icon=":material/info:")
    H('<div class="fs fs-kpis six">' + "".join([
        kpi("Cần xử lý", n0(n_a), f"CRITICAL + HIGH · {pct(n_a / max(1, len(dff)))} số giao dịch"),
        kpi("Đã xử lý", n0(n_done), f"{pct(n_done / max(1, n_a), 0)} số cần xử lý", "good"),
        kpi("Chưa xử lý", n0(n_pend + n_ver), f"{n0(n_ver)} đang xác minh", "accent"),
        kpi("Xác nhận gian lận", n0(n_fraud), f"{pct(n_fraud / max(1, n_done), 0)} số đã xử lý"),
        kpi("Báo động giả", n0(n_fa), f"{pct(n_fa / max(1, n_done), 0)} số đã xử lý"),
        kpi("Tiền chờ kết luận", money_short(a.loc[open_mask, "amount"].sum()),
            f"trên tổng {money_short(a.amount.sum())} đáng ngờ", "warn")]) + "</div>")
    with card("d_prog"):
        title("Tiến độ xử lý")
        H(status_legend(cnt, n_a))

    c1, c2 = st.columns([1.6, 1], gap="medium")
    with c1, card("d_week"):
        title("Cảnh báo theo tuần và trạng thái xử lý")
        weeks = sorted(dff.datetime.dt.to_period("W").dt.start_time.unique())
        t = by_status(a.assign(week=a.datetime.dt.to_period("W").dt.start_time), "week", weeks)
        t.index = [pd.Timestamp(w).strftime("%d/%m") for w in t.index]
        stacked(t, h=300, key="d_wk")
    with c2, card("d_donut"):
        title("Cơ cấu trạng thái")
        fig = go.Figure(go.Pie(labels=STATUSES, values=[cnt.get(k, 0) for k in STATUSES], hole=0.64, sort=False,
                               direction="clockwise", textinfo="none",
                               marker=dict(colors=[STATUS_COLOR[k] for k in STATUSES], line=dict(color="#fff", width=2)),
                               hovertemplate="%{label}: %{value} (%{percent})<extra></extra>"))
        fig.add_annotation(text=f"<b>{n0(n_a)}</b><br>cần xử lý", showarrow=False, font=dict(size=15, color=INK))
        style(fig, 300, legend=True)
        fig.update_layout(legend=dict(orientation="h", y=-0.05, yanchor="top", x=0.5, xanchor="center"))
        chart(fig, "d_pie")

    c1, c2 = st.columns(2, gap="medium")
    with c1, card("d_types"):
        title("Gian lận đã xác nhận theo thủ đoạn", "Thủ đoạn lấy từ nhãn của Data Pack; ca do chuyên viên xác nhận "
              "trong ứng dụng ghi là chưa phân loại")
        conf = a[a.status == "Xác nhận gian lận"]
        lab = res.get("lab_view")
        tmap = dict(zip(lab.transaction_id, lab.fraud_type)) if lab is not None else {}
        ty = conf.transaction_id.map(tmap).fillna("Chưa phân loại").replace("NORMAL", "Chưa phân loại")
        if len(ty):
            hbar(ty.value_counts(), "d_types", h=280, color=LEVEL_COLOR["CRITICAL"], unit="giao dịch gian lận")
        else:
            st.caption("Chưa có giao dịch nào được xác nhận gian lận trong bộ lọc này.")
    with c2, card("d_channel"):
        title("Cảnh báo theo kênh thanh toán", "Màu theo trạng thái xử lý như thanh tiến độ")
        order = a.payment_channel.astype(str).value_counts().index.tolist()
        stacked(by_status(a.assign(payment_channel=a.payment_channel.astype(str)), "payment_channel", order),
                horizontal=True, h=280, key="d_ch", legend=False)

    c1, c2 = st.columns(2, gap="medium")
    with c1, card("d_rules"):
        title("Dấu hiệu xuất hiện nhiều nhất", "Một giao dịch có thể có nhiều dấu hiệu")
        rh = pd.Series({F.RULE_NAMES[k]: int(a[k].sum()) for k in F.RULE_W})
        hbar(rh[rh > 0], "d_rules", h=320)
    with c2, card("d_geo"):
        title("10 tỉnh/thành có nhiều cảnh báo nhất", "Theo nơi phát sinh giao dịch")
        order = a.location.astype(str).value_counts().head(10).index.tolist()
        stacked(by_status(a.assign(location=a.location.astype(str)), "location", order), horizontal=True, h=320,
                key="d_geo", legend=False)

    with card("d_lists"):
        title("Danh sách giao dịch", "Chọn kết luận ngay trong bảng rồi bấm Lưu – các số liệu phía trên cập nhật theo.")
        n_med = int((dff.risk_level == "MEDIUM").sum())
        names = {"Chưa xử lý": f"Chưa xử lý ({n0(n_pend)})", "Đang xác minh": f"Đang xác minh ({n0(n_ver)})",
                 "Đã xử lý": f"Đã xử lý ({n0(n_done)})", "Theo dõi": f"Theo dõi – MEDIUM ({n0(n_med)})"}
        which = st.segmented_control("Danh sách", list(names), default="Chưa xử lý", format_func=lambda x: names[x],
                                     key=f"d_pick_{rid}", label_visibility="collapsed") or "Chưa xử lý"
        if which == "Theo dõi":
            v = dff[dff.risk_level == "MEDIUM"].sort_values("priority_rank")
            grid(pd.DataFrame({"Hạng": v.priority_rank, "Giao dịch": v.transaction_id,
                               "Thời gian GD": v.datetime.dt.strftime("%d/%m/%Y %H:%M"), "Khách hàng": v.customer_id,
                               "Số tiền": v.amount.map(money), "Kênh": v.payment_channel, "Điểm": v.risk_score,
                               "Lý do": v.alert_reason}), height=420,
                 column_config={"Điểm": st.column_config.ProgressColumn(min_value=0, max_value=1, format="%.4f"),
                                "Lý do": st.column_config.TextColumn(width="large")})
            st.caption("MEDIUM chưa đủ ngưỡng cảnh báo nên không chặn; danh sách để chuyên viên rà khi rảnh hoặc khi "
                       "khách có thêm giao dịch đáng ngờ.")
        elif which == "Đã xử lý":
            v = a[a.status.isin(DONE)]
            v = v.assign(_t=pd.to_datetime(v.handled_at, format="%d/%m/%Y %H:%M:%S", errors="coerce")) \
                 .sort_values(["_t", "priority_rank"], ascending=[False, True], na_position="last")
            grid(pd.DataFrame({"Giao dịch": v.transaction_id, "Mức": v.risk_level, "Kết luận": v.status,
                               "Nguồn kết luận": v.source, "Người xử lý": v.handler.replace("", "–"),
                               "Thời điểm xử lý": v.handled_at.replace("", "–"),
                               "Thời gian GD": v.datetime.dt.strftime("%d/%m/%Y %H:%M"), "Số tiền": v.amount.map(money),
                               "Khách hàng": v.customer_id, "Case": v.case_id.fillna("–"), "Ghi chú": v.note}),
                 height=420)
        else:
            v = a[a.status == which].sort_values("priority_rank")
            if v.empty:
                st.success("Không còn giao dịch nào trong danh sách này.", icon=":material/check_circle:")
            else:
                ver = st.session_state.get("ed_ver", 0)
                ed = pd.DataFrame({"Kết luận": pd.Series([None] * len(v), dtype="object"),
                                   "Ghi chú": [""] * len(v), "Hạng": v.priority_rank.values,
                                   "Giao dịch": v.transaction_id.values, "Mức": v.risk_level.values,
                                   "Thời gian GD": v.datetime.dt.strftime("%d/%m/%Y %H:%M").values,
                                   "Khách hàng": v.customer_id.values, "Số tiền": v.amount.map(money).values,
                                   "Điểm": v.risk_score.values, "Case": v.case_id.fillna("–").values,
                                   "Lý do": v.alert_reason.values})
                with st.form(f"proc_{rid}_{which}_{ver}", border=False):
                    who = st.text_input("Người xử lý", placeholder="Tên chuyên viên", key=f"who_{rid}_{which}_{ver}")
                    edited = st.data_editor(
                        ed, hide_index=True, width="stretch", height=420, num_rows="fixed",
                        key=f"ed_{rid}_{which}_{ver}",
                        disabled=[c for c in ed.columns if c not in ("Kết luận", "Ghi chú")],
                        column_config={
                            "Kết luận": st.column_config.SelectboxColumn("Kết luận", options=DECISIONS, width="medium"),
                            "Ghi chú": st.column_config.TextColumn("Ghi chú", width="medium"),
                            "Điểm": st.column_config.ProgressColumn(min_value=0, max_value=1, format="%.4f"),
                            "Lý do": st.column_config.TextColumn(width="large")})
                    saved = st.form_submit_button("Lưu các kết luận đã chọn", type="primary", icon=":material/save:")
                if saved:
                    ch_rows = edited[edited["Kết luận"].isin(DECISIONS)]
                    if ch_rows.empty:
                        st.warning("Chưa chọn kết luận cho giao dịch nào.")
                    else:
                        tm = now_vn()
                        save_decisions(res["ds"], [{"transaction_id": r["Giao dịch"],
                                                    "case_id": "" if r["Case"] == "–" else r["Case"],
                                                    "decision": r["Kết luận"], "note": str(r["Ghi chú"] or "")[:500],
                                                    "user": (who or "chuyên viên")[:60], "time": tm,
                                                    "threshold": rep["threshold"]} for _, r in ch_rows.iterrows()])
                        st.session_state.ed_ver = ver + 1
                        st.session_state.flash = f"Đã lưu kết luận cho {len(ch_rows)} giao dịch"
                        st.rerun()
        b = io.BytesIO()
        a[["transaction_id", "case_id", "datetime", "customer_id", "amount", "risk_score", "risk_level",
           "payment_channel", "location", "status", "source", "handler", "handled_at", "note", "alert_reason"]] \
            .to_csv(b, index=False, encoding="utf-8-sig")
        st.download_button("Tải danh sách cảnh báo kèm trạng thái (.csv)", b.getvalue(),
                           "canh_bao_trang_thai.csv", "text/csv", icon=":material/download:", key=f"dl_dash_{rid}")


def data_menu(res):
    with st.popover("Dữ liệu & quản trị", icon=":material/tune:", width="stretch"):
        st.caption(f"Đang xem: {res['source']} · {n0(len(res['df']))} giao dịch · chạy lúc {res['started']}")
        ups = st.file_uploader("Kiểm tra Data Pack khác (.zip hoặc .csv)", type=["zip", "csv"],
                               accept_multiple_files=True, key="uploader")
        if st.button("Kiểm tra & chạy", icon=":material/upload_file:", disabled=not ups, width="stretch",
                     key="btn_upload"):
            files, notes = F.collect_files([(u.name, u.getvalue()) for u in ups])
            src = ", ".join(u.name for u in ups) if len(ups) <= 2 else f"{len(ups)} tệp CSV tải lên"
            start_run(files, notes, src[:160])
        if (SAMPLE_DIR / "transactions.csv").exists() and not res["source"].startswith("Data Pack BTC"):
            if st.button("Quay lại Data Pack BTC", icon=":material/restart_alt:", width="stretch", key="btn_sample"):
                start_run(sample_files(), [], "Data Pack BTC (có sẵn trong ứng dụng)")
        st.divider()
        st.caption("Kết luận của chuyên viên lưu trên máy chủ đến khi ứng dụng khởi động lại.")
        sure = st.checkbox("Xoá mọi kết luận đã ghi cho bộ dữ liệu này", key="chk_clear")
        if st.button("Xoá kết luận đã ghi", disabled=not sure, width="stretch", key="btn_clear"):
            clear_decisions(res["ds"])
            st.session_state.ed_ver = st.session_state.get("ed_ver", 0) + 1
            st.rerun()


def tab_model(res):
    H('<div class="fs-sec">Mô hình và đánh giá</div><div class="fs-sub">Chất lượng mô hình chấm điểm trên nhãn của '
      'Data Pack – phần này dành cho giám khảo và đội kỹ thuật, không phải màn hình vận hành.</div>')
    model_summary(res)
    tab_eval(res)


def profile_of(df, cust, exclude_ids):
    hist = df[(df.customer_id == cust) & (~df.transaction_id.isin(exclude_ids))]
    top = lambda c: hist[c].value_counts().head(3)  # noqa: E731
    return hist, {k: top(k) for k in ["device_id", "location", "ip_country", "payment_channel"]}


def network(tx, cust):
    devs = tx.groupby("device_id").agg(n=("transaction_id", "size"), reg=("registered_customer_id", "first"),
                                        risk=("device_risk_level", "first")).reset_index()
    mers = tx.groupby("merchant_id").agg(n=("transaction_id", "size"), cat=("merchant_category", "first"),
                                          risk=("merchant_risk_level", "first")).reset_index()
    ys = lambda k: np.linspace(0, 1, k + 2)[1:-1] if k > 1 else np.array([0.5])  # noqa: E731
    pos = {("c", cust): (0, 0.5)}
    for y, r in zip(ys(len(devs)), devs.itertuples()):
        pos[("d", r.device_id)] = (1, y)
    for y, r in zip(ys(len(mers)), mers.itertuples()):
        pos[("m", r.merchant_id)] = (2, y)
    fig = go.Figure()
    ex, ey = [], []
    for r in tx.itertuples():
        for a, b in [(("c", cust), ("d", r.device_id)), (("d", r.device_id), ("m", r.merchant_id))]:
            ex += [pos[a][0], pos[b][0], None]; ey += [pos[a][1], pos[b][1], None]
    fig.add_scatter(x=ex, y=ey, mode="lines", line=dict(color="#C5CEDC", width=1.6), hoverinfo="skip")
    nodes = [(pos[("c", cust)], cust, NAVY, f"Khách hàng {cust}")]
    for r in devs.itertuples():
        own = str(r.reg) == str(cust)
        col = "#2B8A3E" if own else LEVEL_COLOR["CRITICAL"]
        nodes.append((pos[("d", r.device_id)], r.device_id, col,
                      f"Thiết bị {r.device_id}<br>{'Đăng ký cho khách này' if own else 'Không đăng ký cho khách này'}"
                      f"<br>Rủi ro nền {r.risk} · {r.n} giao dịch"))
    for r in mers.itertuples():
        nodes.append((pos[("m", r.merchant_id)], r.merchant_id, LEVEL_COLOR.get(r.risk, "#2557D6") if r.risk != "LOW" else BLUE,
                      f"Merchant {r.merchant_id}<br>{r.cat} · rủi ro {r.risk}<br>{r.n} giao dịch"))
    fig.add_scatter(x=[p[0][0] for p in nodes], y=[p[0][1] for p in nodes], mode="markers+text",
                    marker=dict(size=[30] + [22] * (len(nodes) - 1), color=[p[2] for p in nodes],
                                line=dict(color="#fff", width=2)),
                    text=[p[1] for p in nodes], textposition="bottom center", textfont=dict(size=11, color="#334155"),
                    hovertext=[p[3] for p in nodes], hovertemplate="%{hovertext}<extra></extra>")
    style(fig, 260, legend=False)
    fig.update_xaxes(visible=False, range=[-0.35, 2.35])
    fig.update_yaxes(visible=False, range=[-0.15, 1.1])
    for x, t in [(0, "Khách hàng"), (1, "Thiết bị"), (2, "Merchant")]:
        fig.add_annotation(x=x, y=1.08, text=f"<b>{t}</b>", showarrow=False, font=dict(size=11.5, color=MUTED))
    return fig


def tab_cases(res):
    df, cases = res["df"], res["cases"]
    if cases.empty:
        st.info("Không có cảnh báo nào ở ngưỡng hiện tại.")
        return
    rid = res["rid"]
    H(f'<div class="fs-sec">Hàng đợi case</div><div class="fs-sub">{n0(len(cases))} case từ {n0(res["report"]["n_alerts"])} '
      f'cảnh báo, sắp theo điểm cao nhất trong case. Bấm vào một dòng để mở chi tiết.</div>')
    box = card("queue")
    f1, f2, f3 = box.columns([1.4, 1.2, 1])
    lv = f1.pills("Mức", ["CRITICAL", "HIGH"], selection_mode="multi", default=["CRITICAL", "HIGH"], key=f"lv_{rid}")
    q = f2.text_input("Tìm khách hàng / case / giao dịch", placeholder="VD: CUST0001, CASE0003, TXN001465",
                      key=f"q_{rid}").strip().upper()
    multi = f3.toggle("Chỉ case chưa xử lý xong", value=False, key=f"multi_{rid}")
    stt = status_frame(res).set_index("transaction_id")["status"]

    def case_status(ids):
        ss = [stt.get(t, "Chưa xử lý") for t in ids.split(";")]
        if all(x in DONE for x in ss):
            return "Đã xử lý"
        if any(x == "Đang xác minh" for x in ss):
            return "Đang xác minh"
        if any(x in DONE for x in ss):
            return "Xử lý một phần"
        return "Chưa xử lý"
    cases = cases.assign(case_status=cases.transaction_ids.map(case_status))
    view = cases[cases.case_level.isin(lv or ["CRITICAL", "HIGH"])]
    if q:
        view = view[view.customer_id.str.upper().str.contains(q, regex=False) | view.case_id.str.contains(q, regex=False)
                    | view.transaction_ids.str.upper().str.contains(q, regex=False)]
    if multi:
        view = view[view.case_status != "Đã xử lý"]
    view = view.reset_index(drop=True)
    if view.empty:
        box.warning("Không có case nào khớp bộ lọc.")
        return
    with box:
        ev = grid(pd.DataFrame({"Case": view.case_id, "Mức": view.case_level, "Trạng thái": view.case_status,
                                "Khách hàng": view.customer_id,
                            "Số GD": view.n_transactions, "Bắt đầu": view.start_time.dt.strftime("%d/%m/%Y %H:%M"),
                            "Tổng tiền": view.total_amount.map(money), "Điểm cao nhất": view.max_risk_score,
                            "Lý do chính": view.top_reason}),
                  height=330, on_select="rerun", selection_mode="single-row", key=f"cases_tbl_{rid}",
                  column_config={"Điểm cao nhất": st.column_config.ProgressColumn(min_value=0, max_value=1,
                                                                                  format="%.4f"),
                                 "Lý do chính": st.column_config.TextColumn(width="large")})
    try:
        rows = ev.selection.rows
    except Exception:  # noqa
        rows = []
    i = rows[0] if rows and rows[0] < len(view) else 0
    c = view.iloc[i]
    case_detail(res, c)


def case_detail(res, c):
    df = res["df"]
    ids = c.transaction_ids.split(";")
    tx = df[df.transaction_id.isin(ids)].sort_values("datetime")
    cust, first = c.customer_id, tx.iloc[0]
    with card("case_head"):
        H(f"""<div class="fs"><div style="display:flex;align-items:center;gap:10px;flex-wrap:wrap;margin-bottom:12px">
          <span style="font-size:19px;font-weight:800;color:{INK}">{e(c.case_id)}</span>{badge(c.case_level)}
          <span class="fs-sub" style="margin:0">Hạng ưu tiên #{int(c.case_rank)}</span></div>
          <div class="fs-case">
            <div class="item"><div class="k">Khách hàng</div><div class="v">{e(cust)}</div></div>
            <div class="item"><div class="k">Phân khúc · nơi ở</div><div class="v">{e(first.customer_segment)} · {e(first.home_location)}</div></div>
            <div class="item"><div class="k">Thời gian</div><div class="v">{dt(c.start_time)} → {pd.Timestamp(c.end_time):%H:%M}</div></div>
            <div class="item"><div class="k">Số giao dịch</div><div class="v">{int(c.n_transactions)}</div></div>
            <div class="item"><div class="k">Tổng tiền</div><div class="v">{money(c.total_amount)}</div></div>
            <div class="item"><div class="k">Điểm cao nhất</div><div class="v">{d(c.max_risk_score, 4)}</div></div>
          </div><div class="fs-reason"><b>Lý do chính:</b> {e(c.top_reason)}</div></div>""")
        on = [(F.RULE_NAMES[k], int(tx[k].sum())) for k in F.RULE_W if tx[k].sum() > 0]
        if on:
            H('<div class="fs" style="margin-top:12px">' + "".join(
                f'<span class="fs-flag on">{e(nm)} · {k} GD</span>' for nm, k in on) + "</div>")

    c1, c2 = st.columns([1, 1.05], gap="medium")
    with c1, card("case_prof"):
        hist, prof = profile_of(df, cust, ids)
        title("Thói quen của khách so với case này",
              f"Thói quen tính từ {n0(len(hist))} giao dịch khác của {e(cust)} (không tính các giao dịch trong case)")
        rows = []
        for col, lab in [("device_id", "Thiết bị"), ("location", "Tỉnh/thành"), ("ip_country", "Quốc gia IP"),
                         ("payment_channel", "Kênh")]:
            usual = ", ".join(f"{e(k)} ({v})" for k, v in prof[col].items()) or "–"
            now = []
            for v in sorted(tx[col].astype(str).unique()):
                now.append(f'<span class="new">{e(v)} · mới</span>' if v not in set(hist[col].astype(str)) else e(v))
            rows.append({"k": lab, "usual": usual, "now": ", ".join(now)})
        med = hist.amount.median() if len(hist) else np.nan
        rows.append({"k": "Số tiền", "usual": f"trung vị {money(med)}" if med == med else "–",
                     "now": f"cao nhất {money(tx.amount.max())} (gấp {d(tx.amt_ratio.max(), 1)} lần mức nền)"})
        table(rows, [("k", "Chiều"), ("usual", "Thường dùng"), ("now", "Trong case")], raw_cols=("usual", "now"))
    with c2, card("case_net"):
        title("Liên kết khách hàng – thiết bị – merchant",
              "Xanh lá: thiết bị đăng ký cho khách · Đỏ: thiết bị lạ · Màu merchant theo mức rủi ro nền")
        chart(network(tx, cust), f"net_{c.case_id}")

    with card("case_tx"):
        title(f"Các giao dịch trong case ({len(tx)})")
        grid(pd.DataFrame({"Thời gian": tx.datetime.dt.strftime("%d/%m %H:%M:%S"), "Giao dịch": tx.transaction_id,
                           "Số tiền": tx.amount.map(money), "Trạng thái": tx.transaction_status,
                           "Kênh": tx.payment_channel, "Thiết bị": tx.device_id, "Tỉnh": tx.location,
                           "IP": tx.ip_country, "Merchant": tx.merchant_id + " · " + tx.merchant_risk_level.astype(str),
                           "Điểm": tx.risk_score, "Mức": tx.risk_level, "Lý do": tx.alert_reason}),
             column_config={"Điểm": st.column_config.ProgressColumn(min_value=0, max_value=1, format="%.4f"),
                            "Lý do": st.column_config.TextColumn(width="large")})

    with card("case_decide"):
        title("Quyết định của chuyên viên", "Áp dụng cho mọi giao dịch trong case; lưu vào nhật ký kiểm toán (kèm phiên bản mô hình và ngưỡng) và cập nhật Dashboard")
        with st.form(f"f_{res['rid']}_{c.case_id}", clear_on_submit=True, border=False):
            a, b = st.columns([1.3, 1])
            dec = a.radio("Kết luận", DECISIONS, horizontal=True)
            who = b.text_input("Người xử lý", placeholder="Tên chuyên viên")
            note = st.text_area("Ghi chú", placeholder="VD: đã gọi xác minh, khách xác nhận không thực hiện", height=80)
            if st.form_submit_button("Ghi quyết định", type="primary", icon=":material/task_alt:"):
                tm = now_vn()
                save_decisions(res["ds"], [{"transaction_id": t, "case_id": c.case_id, "decision": dec, "note": note[:500],
                                            "user": (who or "chuyên viên")[:60], "time": tm,
                                            "threshold": res["report"]["threshold"]} for t in ids])
                st.session_state.flash = f"Đã ghi quyết định cho {c.case_id} ({len(ids)} giao dịch)"
                st.rerun()
        lg = load_log(res["ds"])
        lg = lg[lg.case_id == c.case_id].drop_duplicates(["time", "decision", "user", "note"])
        log = lg.to_dict("records")
        if log:
            table(log[::-1], [("time", "Thời điểm"), ("user", "Người xử lý"), ("decision", "Kết luận"), ("note", "Ghi chú")])
        else:
            st.caption("Chưa có quyết định nào cho case này.")


FIELD_LABELS = {"customer_id": "Khách hàng", "customer_segment": "Phân khúc", "home_location": "Nơi ở",
                "datetime": "Thời gian", "amount": "Số tiền", "avg_transaction_amount": "Mức nền của khách",
                "amt_ratio": "Gấp mức nền", "payment_channel": "Kênh", "transaction_status": "Trạng thái",
                "authentication_method": "Xác thực", "device_id": "Thiết bị", "device_type": "Loại thiết bị",
                "registered_customer_id": "Thiết bị đăng ký cho", "dev_age_days": "Tuổi thiết bị (ngày)",
                "location": "Tỉnh/thành giao dịch", "ip_country": "Quốc gia IP", "merchant_id": "Merchant",
                "merchant_category": "Ngành hàng", "merchant_risk_level": "Rủi ro nền merchant",
                "confidence": "Độ tin cậy", "case_id": "Case"}


def fmt_field(k, v):
    if v is None or (isinstance(v, float) and v != v) or (isinstance(v, str) and v in ("", "nan", "None")):
        return "–"
    if k in ("amount", "avg_transaction_amount"):
        return money(v)
    if k == "amt_ratio":
        return d(v, 2) + "×"
    if k == "datetime":
        return pd.Timestamp(v).strftime("%d/%m/%Y %H:%M:%S")
    if k == "dev_age_days":
        return n0(v)
    if k == "confidence":
        return pct(v)
    return str(v)


def tab_lookup(res):
    df, rep = res["df"], res["report"]
    thr = rep["threshold"]
    H('<div class="fs-sec">Tra cứu giao dịch</div><div class="fs-sub">Gõ mã giao dịch để lọc; danh sách xếp theo '
      'thứ tự ưu tiên. Mỗi giao dịch có điểm từng tầng, các rule bật và độ lệch so với hồ sơ của chính khách hàng.</div>')
    ids = df.transaction_id.tolist()
    lab = dict(zip(df.transaction_id, [f"{t} · {l} · {d(s, 4)}" for t, l, s in
                                       zip(df.transaction_id, df.risk_level, df.risk_score)]))
    tid = st.selectbox("Mã giao dịch", ids, index=0, format_func=lambda t: lab.get(t, t), key=f"txn_{res['rid']}")
    r = df[df.transaction_id == tid].iloc[0]

    with card("tx_head"):
        H(f"""<div class="fs"><div style="display:flex;align-items:center;gap:10px;flex-wrap:wrap;margin-bottom:12px">
          <span style="font-size:19px;font-weight:800;color:{INK}">{e(r.transaction_id)}</span>{badge(r.risk_level)}
          <span class="fs-sub" style="margin:0">Hạng ưu tiên #{int(r.priority_rank)} / {n0(len(df))}</span></div>
          <div class="fs-case">
            <div class="item"><div class="k">Điểm rủi ro</div><div class="v">{d(r.risk_score, 4)}</div></div>
            <div class="item"><div class="k">Ngưỡng cảnh báo</div><div class="v">{d(thr, 4)}</div></div>
            <div class="item"><div class="k">Cảnh báo</div><div class="v">{'Có' if r.predicted_fraud else 'Không'}</div></div>
            <div class="item"><div class="k">Số tiền</div><div class="v">{money(r.amount)}</div></div>
            <div class="item"><div class="k">Case</div><div class="v">{e(r.case_id) if isinstance(r.case_id, str) else '–'}</div></div>
          </div><div class="fs-reason"><b>Lý do:</b> {e(r.alert_reason)}</div></div>""")

    c1, c2 = st.columns([1.1, 1], gap="medium")
    with c1, card("tx_score"):
        w = F.W_FUSION
        title("Điểm được tạo thành thế nào", f"risk = 0,6 × ML + 0,3 × Rule + 0,1 × Bất thường = {d(r.risk_score, 4)}")
        parts = [("ML (Logistic Regression)", w[0] * r.ml_score, r.ml_score, BLUE),
                 ("Rule (noisy-OR)", w[1] * r.rule_score, r.rule_score, LEVEL_COLOR["HIGH"]),
                 ("Bất thường (Isolation Forest)", w[2] * r.anomaly_score, r.anomaly_score, "#7C5CD6")]
        fig = go.Figure()
        for nm, v, raw, col in parts:
            fig.add_bar(y=["Điểm"], x=[v], name=f"{nm}: {d(raw, 3)}", orientation="h", marker_color=col,
                        hovertemplate=f"{nm}<br>điểm gốc {d(raw, 4)} · đóng góp {d(v, 4)}<extra></extra>")
        fig.add_vline(x=thr, line=dict(color=NAVY, dash="dash", width=1.5))
        fig.add_annotation(x=thr, y=1.0, yref="paper", text=f"Ngưỡng {d(thr, 4)}", showarrow=False, yanchor="bottom",
                           font=dict(size=11, color=NAVY))
        style(fig, 190, barmode="stack", legend=True)
        fig.update_layout(legend=dict(y=-0.25, yanchor="top", traceorder="normal"), margin=dict(t=24, b=6))
        fig.update_xaxes(range=[0, 1], dtick=0.1)
        fig.update_yaxes(visible=False)
        chart(fig, f"sc_{tid}")
        H('<div class="fs" style="margin-top:4px">' + "".join(
            f'<span class="fs-flag {"on" if r[k] else "off"}">{"●" if r[k] else "○"} {e(F.RULE_NAMES[k])}</span>'
            for k in F.RULE_W) + "</div>")
    with c2, card("tx_prof"):
        title("Lệch so với hồ sơ hành vi của khách", "Hồ sơ dựng từ các giao dịch khác của chính khách hàng (leave-one-out)")
        yn = lambda v: '<span class="new">Có</span>' if v else "Không"  # noqa: E731
        rows = [{"k": "Số tiền lệch (z-score so với chính khách)", "v": d(r.prof_amt_z, 2)},
                {"k": "Lần đầu dùng thiết bị này", "v": yn(r.prof_new_device)},
                {"k": "Lần đầu giao dịch ở tỉnh này", "v": yn(r.prof_new_location)},
                {"k": "Lần đầu dùng kênh này", "v": yn(r.prof_new_channel)},
                {"k": "Lần đầu dùng IP nước này", "v": yn(r.prof_new_ip)},
                {"k": "Số chiều lệch hồ sơ", "v": str(int(r.profile_dev_count))},
                {"k": "Giao dịch của khách trong ±5 phút", "v": str(int(r.n_txn_5m))},
                {"k": "Giao dịch thất bại trong ±30 phút", "v": str(int(r.n_fail_30m))}]
        table(rows, [("k", "Chỉ báo"), ("v", "Giá trị")], num_cols=("v",), raw_cols=("v",))

    with card("tx_fields"):
        title("Thông tin giao dịch")
        H('<div class="fs fs-kv">' + "".join(
            f"<div><span>{e(lab_)}</span><b>{e(fmt_field(k, r[k]))}</b></div>"
            for k, lab_ in FIELD_LABELS.items() if k in r.index) + "</div>")


def tab_eval(res):
    rep, df = res["report"], res["df"]
    if res["mode"] != "train":
        st.info("Đánh giá mô hình cần nhãn. Data Pack vừa chạy không có fraud_labels_sample.csv nên ứng dụng chỉ "
                "chấm điểm bằng mô hình học sẵn – hãy tải kèm tệp nhãn để xem F1, Recall, baseline và độ bền.",
                icon=":material/info:")
        return
    lv = res["lab_view"]
    H(f'<div class="fs-sec">Đánh giá mô hình</div><div class="fs-sub">{n0(len(lv))} giao dịch có nhãn '
      f'({n0(lv.y.sum())} gian lận) · 5-fold phân tầng · ngưỡng chọn lồng trong từng fold · seed {rep["seed"]}</div>')

    with card("cmp"):
        title("So sánh với các baseline", "Cùng 5 fold, cùng cách chọn ngưỡng – chỉ khác cấu hình mô hình")
        comp = rep["comparison_nested_cv"]
        names = list(comp)
        rows = [{"name": k, "p": d(v["precision"]), "r": d(v["recall"]), "f1": d(v["f1"]), "auc": d(v["pr_auc"]),
                 "fpr": pct(v["fpr"], 2), "fp": v["fp"], "fn": v["fn"]} for k, v in comp.items()]
        best = {c: int(np.argmax([comp[k][m] for k in names])) for c, m in
                [("p", "precision"), ("r", "recall"), ("f1", "f1"), ("auc", "pr_auc")]}
        table(rows, [("name", "Cấu hình"), ("p", "Precision"), ("r", "Recall"), ("f1", "F1"), ("auc", "PR-AUC"),
                     ("fpr", "FPR"), ("fp", "Báo nhầm"), ("fn", "Bỏ sót")],
              num_cols=("p", "r", "f1", "auc", "fpr", "fp", "fn"), hl_last=True, best=best)

    c1, c2 = st.columns([1.15, 1], gap="medium")
    with c1, card("pertype"):
        title("Recall theo từng mẫu gian lận", "So sánh không có hồ sơ hành vi (A) và có hồ sơ (B, chính thức) · vạch: mục tiêu 0,80")
        cnt = {k: v for k, v in rep["labeled_counts"].items() if k != "NORMAL"}
        types = sorted(cnt, key=lambda t: rep["detected_by_type"]["Gộp 3 tầng – có hồ sơ (B, chính thức)"].get(t, 0) / cnt[t])
        A = rep["detected_by_type"]["Gộp 3 tầng – không hồ sơ (A)"]
        B = rep["detected_by_type"]["Gộp 3 tầng – có hồ sơ (B, chính thức)"]
        fig = go.Figure()
        fig.add_bar(y=types, x=[A.get(t, 0) / cnt[t] for t in types], orientation="h", name="Không hồ sơ (A)",
                    marker_color=BLUE_SOFT, text=[f"{A.get(t, 0)}/{cnt[t]}" for t in types], textposition="inside",
                    insidetextanchor="end", textfont=dict(size=10.5, color="#1E3A8A"),
                    hovertemplate="%{y}<br>A: %{x:.0%}<extra></extra>")
        fig.add_bar(y=types, x=[B.get(t, 0) / cnt[t] for t in types], orientation="h", name="Có hồ sơ (B, chính thức)",
                    marker_color=BLUE, text=[f"{B.get(t, 0)}/{cnt[t]}" for t in types], textposition="inside",
                    insidetextanchor="end", textfont=dict(size=10.5, color="#fff"),
                    hovertemplate="%{y}<br>B: %{x:.0%}<extra></extra>")
        fig.add_vline(x=0.8, line=dict(color=NAVY, dash="dash", width=1.5))
        style(fig, 360, barmode="group", bargap=0.25, bargroupgap=0.08)
        fig.update_xaxes(range=[0, 1.02], tickformat=".0%")
        chart(fig, "pertype")
    with c2, card("robust"):
        title("Độ bền theo thời gian", "Học trên các tháng đầu, kiểm tra trên các tháng sau – mô phỏng triển khai thật")
        ts = rep.get("time_split")
        if ts:
            H(f'<div class="fs-sub" style="margin-bottom:6px"><b>Học</b> {e(ts["train_months"])} ({n0(ts["n_train"])} GD, '
              f'{ts["fraud_train"]} gian lận) → <b>kiểm tra</b> {e(ts["test_months"])} ({n0(ts["n_test"])} GD, '
              f'{ts["fraud_test"]} gian lận)</div>')
            kpis([kpi("Precision", d(ts["precision"])), kpi("Recall", d(ts["recall"])), kpi("F1", d(ts["f1"]))], small=True)
            st.caption(f"{ts['fn']} bỏ sót, {ts['fp']} báo nhầm trên tập tương lai · PR-AUC {d(ts['pr_auc'])}")
        else:
            st.caption("Dữ liệu quá ít tháng để tách theo thời gian.")
        title("Danh sách ưu tiên (top-k)", "Nếu chỉ đủ người xem k giao dịch điểm cao nhất")
        table([{"k": n0(t["k"]), "f": t["fraud"], "p": d(t["precision"], 2), "r": d(t["recall"])} for t in rep["topk"]],
              [("k", "Xem k giao dịch đầu"), ("f", "Gian lận thật"), ("p", "Precision@k"), ("r", "Recall@k")],
              num_cols=("k", "f", "p", "r"))

    with card("thr"):
        title("Thử thay đổi ngưỡng cảnh báo",
              "Kéo thanh trượt để thấy đánh đổi giữa bắt được nhiều gian lận và báo nhầm. Điểm dùng ở đây là điểm "
              "out-of-fold tại một ngưỡng cố định nên có thể chênh nhẹ so với bảng so sánh (ngưỡng lồng).")
        key = f"thr_{res['rid']}"
        cost = rep["cost_threshold"]["threshold"]
        if key not in st.session_state:
            st.session_state[key] = float(rep["threshold"])
        b1, b2, _ = st.columns([1, 1.25, 2])
        b1.button(f"Cân bằng F1 · {d(rep['threshold'], 4)}", width="stretch", key=f"pf_{res['rid']}",
                  on_click=lambda: st.session_state.update({key: float(rep["threshold"])}))
        b2.button(f"Theo chi phí (bỏ sót = 10 × báo nhầm) · {d(cost, 4)}", width="stretch", key=f"pc_{res['rid']}",
                  on_click=lambda: st.session_state.update({key: float(cost)}))
        t = st.slider("Ngưỡng", 0.05, 0.99, step=0.0005, format="%.4f", key=key)
        y, s = lv.y.values, lv.score.values
        pred = s >= t
        tp, fp, fn = int((pred & (y == 1)).sum()), int((pred & (y == 0)).sum()), int((~pred & (y == 1)).sum())
        P = tp / max(1, tp + fp); R = tp / max(1, tp + fn); F1 = 2 * P * R / max(1e-9, P + R)
        kpis([kpi("Precision", d(P)), kpi("Recall", d(R)), kpi("F1", d(F1)), kpi("Bỏ sót / báo nhầm", f"{fn} / {fp}"),
              kpi("Cảnh báo trên toàn bộ", n0((df.risk_score >= t).sum()), f"trên {n0(len(df))} giao dịch")])
        pr, rc, th = precision_recall_curve(y, s)
        fig = go.Figure()
        fig.add_scatter(x=rc, y=pr, mode="lines", line=dict(color=BLUE, width=2.5, shape="hv"), name="Đường PR",
                        hovertemplate="Recall %{x:.3f}<br>Precision %{y:.3f}<extra></extra>")
        fig.add_scatter(x=[R], y=[P], mode="markers", marker=dict(size=13, color=LEVEL_COLOR["HIGH"],
                                                                  line=dict(color="#fff", width=2)),
                        name=f"Ngưỡng {d(t, 4)}", hovertemplate="Recall %{x:.3f}<br>Precision %{y:.3f}<extra></extra>")
        style(fig, 330, xaxis=dict(title="Recall", range=[0, 1.02]), yaxis=dict(title="Precision", range=[0, 1.04]))
        chart(fig, "pr")

    c1, c2 = st.columns(2, gap="medium")
    with c1, card("fair"):
        title("Công bằng theo phân khúc khách hàng", "Phân khúc và nơi ở không dùng làm feature; bảng này để theo dõi thiên lệch")
        seg = pd.Series(rep["alert_rate_by_segment"]); fpr = pd.Series(rep["fpr_by_segment_labeled"]).reindex(seg.index)
        fig = go.Figure()
        fig.add_bar(y=seg.index, x=seg.values, orientation="h", name="Tỉ lệ cảnh báo (toàn bộ)", marker_color=BLUE,
                    hovertemplate="%{y}<br>%{x:.1%}<extra></extra>")
        fig.add_bar(y=fpr.index, x=fpr.values, orientation="h", name="Tỉ lệ báo nhầm FPR (có nhãn)",
                    marker_color=LEVEL_COLOR["HIGH"], hovertemplate="%{y}<br>%{x:.2%}<extra></extra>")
        style(fig, 300, barmode="group")
        fig.update_xaxes(tickformat=".1%")
        chart(fig, "fair")
    with c2, card("geo"):
        title("Tỉ lệ cảnh báo theo tỉnh/thành nơi ở của khách")
        geo = pd.Series(rep["alert_rate_by_home_location"]).sort_values()
        fig = go.Figure(go.Bar(y=geo.index, x=geo.values, orientation="h", marker_color=BLUE, text=[pct(v) for v in geo.values],
                               textposition="outside", cliponaxis=False, hovertemplate="%{y}<br>%{x:.1%}<extra></extra>"))
        style(fig, 300, legend=False)
        fig.update_xaxes(tickformat=".0%", range=[0, geo.max() * 1.22])
        chart(fig, "geo")

    ak = rep.get("answer_key")
    if ak:
        with card("ak"):
            title(f"Đối chiếu answer_key_sample.csv của BTC ({ak['n']} mẫu)")
            kpis([kpi("Khớp có/không cảnh báo", f"{ak['agree_alert']}/{ak['n']}"),
                  kpi("Mẫu gian lận được cảnh báo", f"{ak['fraud_alerted']}/{ak['n_fraud']}"),
                  kpi("Khớp mức HIGH/CRITICAL", f"{ak['level_match_on_alerted']}/{ak['fraud_alerted']}"),
                  kpi("Mẫu bình thường bị cảnh báo", f"{ak['normal_alerted']}/{ak['n_normal']}")])

    c1, c2 = st.columns(2, gap="medium")
    cols = {"transaction_id": "Giao dịch", "fraud_type": "Mẫu", "amt_ratio": "Gấp mức nền", "location": "Tỉnh",
            "home_location": "Nơi ở", "ip_country": "IP", "merchant_risk_level": "Merchant", "risk_oof": "Điểm"}
    for col, key, ttl, sub in [(c1, "false_negatives", "Gian lận bị bỏ sót", "Điểm dưới ngưỡng – chủ yếu merchant rủi ro "
                                "với số tiền không quá lệch"),
                               (c2, "false_positives", "Báo động giả", "Giao dịch bình thường nhưng số tiền rất lệch mức nền")]:
        with col, card(key):
            items = rep.get(key, [])
            title(f"{ttl} ({len(items)})", sub)
            if items:
                x = pd.DataFrame(items)[list(cols)].rename(columns=cols)
                x["Mẫu"] = x["Mẫu"].fillna("NORMAL")
                grid(x, column_config={"Gấp mức nền": st.column_config.NumberColumn(format="%.2f×"),
                                       "Điểm": st.column_config.NumberColumn(format="%.3f")})


def tab_checks(res):
    ch = res["checks"]
    cnt = pd.Series([c["status"] for c in ch]).value_counts()
    H('<div class="fs-sec">Kiểm tra dữ liệu</div><div class="fs-sub">Nguồn: ' + e(res["source"]) + "</div>")
    kpis([kpi("Đạt", n0(cnt.get("ĐẠT", 0)), "hạng mục"), kpi("Cảnh báo", n0(cnt.get("CẢNH BÁO", 0)), "vẫn chạy được"),
          kpi("Lỗi", n0(cnt.get("LỖI", 0)), "chặn việc chạy mô hình"),
          kpi("Thiếu (tuỳ chọn)", n0(cnt.get("THIẾU", 0)), "không bắt buộc")])
    with card("checks"):
        checks_table(ch)
    df = res["df"]
    with card("dataset"):
        title("Tóm tắt Data Pack")
        H('<div class="fs fs-kv">' + "".join(f"<div><span>{k}</span><b>{v}</b></div>" for k, v in [
            ("Giao dịch", n0(len(df))), ("Khách hàng", n0(df.customer_id.nunique())),
            ("Merchant", n0(df.merchant_id.nunique())), ("Thiết bị", n0(df.device_id.nunique())),
            ("Từ ngày", res["report"]["period"][0]), ("Đến ngày", res["report"]["period"][1]),
            ("Chế độ", "Huấn luyện + đánh giá" if res["mode"] == "train" else "Chỉ chấm điểm (mô hình học sẵn)"),
            ("Mô hình", e(F.VERSION))]) + "</div>")


@st.cache_data(show_spinner=False, max_entries=8)
def build_files(rid, _res):
    out = {}
    b = io.BytesIO(); _res["submission"].to_csv(b, index=False, encoding="utf-8-sig"); out["submission_output.csv"] = b.getvalue()
    b = io.BytesIO(); _res["cases"].to_csv(b, index=False, encoding="utf-8-sig"); out["cases.csv"] = b.getvalue()
    b = io.BytesIO(); _res["df"].drop(columns=["ts"], errors="ignore").to_csv(b, index=False, encoding="utf-8-sig")
    out["scored_transactions_detail.csv"] = b.getvalue()
    out["kpi_report.json"] = json.dumps(_res["report"], ensure_ascii=False, indent=2, default=str).encode("utf-8")
    b = io.BytesIO(); pd.DataFrame(_res["checks"]).to_csv(b, index=False, encoding="utf-8-sig"); out["data_checks.csv"] = b.getvalue()
    return out


def tab_downloads(res):
    files = dict(build_files(res["rid"], res))
    b = io.BytesIO()
    audit = load_log(res["ds"])
    audit.to_csv(b, index=False, encoding="utf-8-sig")
    files["audit_log.csv"] = b.getvalue()
    H('<div class="fs-sec">Tải kết quả</div><div class="fs-sub">Các tệp được tạo từ lượt chạy này; '
      'submission_output.csv theo đúng cột và thứ tự dòng của template BTC.</div>')
    z = io.BytesIO()
    with zipfile.ZipFile(z, "w", zipfile.ZIP_DEFLATED) as zf:
        for k, v in files.items():
            zf.writestr(k, v)
    st.download_button("Tải tất cả (.zip)", z.getvalue(), "FraudShield_ketqua.zip", "application/zip", type="primary",
                       icon=":material/download:", key=f"dl_all_{res['rid']}")
    desc = [("submission_output.csv", "Tệp nộp theo template BTC: risk_score, risk_level, predicted_fraud, alert_reason, "
             "priority_rank, confidence, model_name, processing_time_ms", "text/csv"),
            ("cases.csv", "Danh sách case: khách hàng, khung thời gian, tổng tiền, mức cao nhất, lý do chính", "text/csv"),
            ("scored_transactions_detail.csv", "Toàn bộ giao dịch kèm feature, cờ rule và điểm từng tầng", "text/csv"),
            ("kpi_report.json", "Báo cáo chỉ số: so sánh baseline, theo mẫu, top-k, chi phí, công bằng, answer key",
             "application/json"),
            ("data_checks.csv", "Kết quả kiểm tra dữ liệu và đối chiếu manifest", "text/csv"),
            ("audit_log.csv", f"Nhật ký kết luận của chuyên viên ({len(audit)} dòng, mỗi dòng một giao dịch)",
             "text/csv")]
    cols = st.columns(2, gap="medium")
    for i, (nm, ds, mime) in enumerate(desc):
        with cols[i % 2], card(f"dl{i}"):
            title(nm, f"{ds} · {n0(max(1, len(files[nm]) / 1024))} KB")
            st.download_button(f"Tải {nm}", files[nm], nm, mime, icon=":material/download:", key=f"dl_{i}_{res['rid']}")


# ----------------------------------------------------------------------------- khung kết quả
def results(res):
    rep = res["report"]
    if st.session_state.get("flash"):
        st.toast(st.session_state.pop("flash"), icon=":material/check_circle:")
    mode = "huấn luyện + đánh giá" if res["mode"] == "train" else "chấm điểm bằng mô hình học sẵn"
    df = res["df"]
    a, b = st.columns([4.2, 1], vertical_alignment="center")
    with a:
        H(f"""<div class="fs fs-head"><div class="fs-logo">FS</div><div>
          <div class="t">Fraud Shield · Giám sát gian lận thanh toán</div>
          <div class="m">{e(res['source'])} · {n0(len(df))} giao dịch {df.datetime.min():%d/%m/%Y} – {df.datetime.max():%d/%m/%Y}
          · {mode} · cập nhật {datetime.now(VN_TZ):%H:%M %d/%m/%Y}</div></div></div>""")
    with b:
        data_menu(res)
    if "failed" in st.session_state:
        st.error("Data Pack vừa tải lên chưa đạt các kiểm tra bắt buộc nên chưa thay dữ liệu đang xem – chi tiết bên dưới.")
        checks_table(st.session_state.failed["checks"])
    tabs = st.tabs(["Dashboard", "Hàng đợi case", "Tra cứu giao dịch", "Mô hình & đánh giá", "Kiểm tra dữ liệu",
                    "Tải kết quả"])
    for tab, fn in zip(tabs, [tab_dashboard, tab_cases, tab_lookup, tab_model, tab_checks, tab_downloads]):
        with tab:
            fn(res)


# ----------------------------------------------------------------------------- main
if "res" not in st.session_state:
    auto_load_sample()
if "res" in st.session_state:
    results(st.session_state.res)
else:
    landing()
H(f"""<div class="fs fs-foot"><span>* Cảnh báo không đồng nghĩa với kết luận gian lận – quyết định cuối cùng thuộc
  chuyên viên. Dữ liệu tải lên chỉ xử lý trong phiên; kết luận xử lý lưu trên máy chủ đến khi ứng dụng khởi động lại.</span><span>{e(F.VERSION)} · Festival Chuyển đổi
  số 2026 · Bảng B</span></div>""")
