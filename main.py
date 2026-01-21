import re
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple

import pandas as pd
import requests
import streamlit as st

from bs4 import BeautifulSoup
import tldextract
import plotly.express as px


# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="섹션별 Top 5 + 성향 분포 (직전 24시간)",
    page_icon="🗞️",
    layout="wide",
)

st.markdown(
    """
<style>
.block-container { padding-top: 1.1rem; padding-bottom: 2rem; }
small.muted { color: rgba(49,51,63,.65); }
.card {
  border: 1px solid rgba(49,51,63,.15);
  border-radius: 14px;
  padding: 14px 16px;
  margin-bottom: 10px;
  background: rgba(255,255,255,.02);
}
.card h4 { margin: 0 0 8px 0; }
.kv { display: flex; gap: 12px; flex-wrap: wrap; margin: 6px 0 10px 0; }
.kv span { font-size: 0.92rem; color: rgba(49,51,63,.72); }
.badge { display:inline-block; padding:2px 8px; border-radius: 999px; border:1px solid rgba(49,51,63,.18); font-size:.82rem;}
hr.soft { border: none; border-top: 1px solid rgba(49,51,63,.10); margin: 14px 0; }
ul.tight { margin: 0.2rem 0 0.2rem 1.2rem; }
</style>
""",
    unsafe_allow_html=True,
)

# -----------------------------
# Constants
# -----------------------------
KST = timezone(timedelta(hours=9))
UTC = timezone.utc

GDELT_DOC_ENDPOINT = "https://api.gdeltproject.org/api/v2/doc/doc"
USER_AGENT = "Mozilla/5.0 (compatible; StreamlitSectionTop5/2.2; +https://streamlit.io)"
REQUEST_TIMEOUT = 10  # seconds

BIAS_ORDER = ["보수", "중도", "진보", "미분류"]


# -----------------------------
# Models / Config
# -----------------------------
@dataclass(frozen=True)
class SectionQuery:
    section: str
    domestic_query: str
    overseas_query: str


SECTIONS: List[SectionQuery] = [
    SectionQuery(
        section="정치",
        domestic_query='(정치 OR 정부 OR 국회 OR 대통령 OR 여당 OR 야당 OR 총선 OR 대선 OR 선거 OR 공천)',
        overseas_query='(politics OR government OR parliament OR congress OR president OR election OR campaign)',
    ),
    SectionQuery(
        section="경제",
        domestic_query='(경제 OR 증시 OR 주식 OR 코스피 OR 코스닥 OR 환율 OR 금리 OR 물가 OR 인플레이션 OR 기업 OR 산업 OR 반도체)',
        overseas_query='(economy OR markets OR stocks OR inflation OR interest rates OR central bank OR currency OR business OR industry OR semiconductor)',
    ),
    SectionQuery(
        section="사회",
        domestic_query='(사회 OR 사건 OR 사고 OR 범죄 OR 재난 OR 교육 OR 의료 OR 복지 OR 노동 OR 파업 OR 법원 OR 검찰 OR 경찰)',
        overseas_query='(society OR crime OR accident OR disaster OR education OR health OR welfare OR labor OR strike OR court OR police)',
    ),
    SectionQuery(
        section="국제",
        domestic_query='(국제 OR 외교 OR 정상회담 OR UN OR 유엔 OR 미국 OR 중국 OR 일본 OR 러시아 OR 우크라이나 OR 중동 OR 가자)',
        overseas_query='(world OR international OR diplomacy OR summit OR UN OR Ukraine OR Russia OR China OR Japan OR Middle East OR Gaza)',
    ),
    SectionQuery(
        section="스포츠",
        domestic_query='(스포츠 OR 축구 OR 야구 OR 농구 OR 배구 OR 골프 OR e스포츠 OR 올림픽 OR 월드컵 OR KBO OR K리그)',
        overseas_query='(sports OR football OR soccer OR baseball OR basketball OR Olympics OR World Cup OR NBA OR MLB OR NHL)',
    ),
]


# -----------------------------
# Bias mapping (starter; user-editable)
# -----------------------------
def default_bias_mapping_df() -> pd.DataFrame:
    data = [
        # Korea (illustrative)
        ("chosun.com", "보수"),
        ("donga.com", "보수"),
        ("joongang.co.kr", "중도"),
        ("mk.co.kr", "중도"),
        ("yonhapnews.co.kr", "중도"),
        ("hani.co.kr", "진보"),
        ("khan.co.kr", "진보"),
        # Global (illustrative)
        ("reuters.com", "중도"),
        ("apnews.com", "중도"),
        ("bbc.co.uk", "중도"),
        ("economist.com", "중도"),
        ("foxnews.com", "보수"),
        ("wsj.com", "보수"),
        ("nytimes.com", "진보"),
        ("washingtonpost.com", "진보"),
        ("cnn.com", "진보"),
    ]
    return pd.DataFrame(data, columns=["domain", "bias"])


# -----------------------------
# Helpers: text / parsing
# -----------------------------
def clean_text(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip()


def escape_html(s: str) -> str:
    s = s or ""
    return (
        s.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#39;")
    )


def normalize_domain(url: str) -> Optional[str]:
    if not url or not isinstance(url, str):
        return None
    try:
        ext = tldextract.extract(url)
        return ext.registered_domain.lower() if ext.registered_domain else None
    except Exception:
        return None


def parse_seendate_utc(s: str) -> Optional[datetime]:
    if not s or not isinstance(s, str):
        return None
    try:
        return datetime.strptime(s, "%Y%m%d%H%M%S").replace(tzinfo=UTC)
    except Exception:
        return None


def rolling_24h_range_utc() -> Tuple[datetime, datetime]:
    """
    검색 실행 시점 기준 직전 24시간 범위(UTC).
    """
    end_utc = datetime.now(UTC)
    start_utc = end_utc - timedelta(hours=24)
    return start_utc, end_utc


def build_section_query(region: str, section_cfg: SectionQuery, extra_keyword: str) -> str:
    """
    국내: language:kor + 섹션 키워드 (+ optional extra keyword)
    해외: language:eng -sourceCountry:KOR + 섹션 키워드 (+ optional extra keyword)
    """
    extra = clean_text(extra_keyword)
    extra_part = f'("{extra}")' if extra else ""

    if region == "국내":
        base = "language:kor"
        sec = section_cfg.domestic_query
        return f"{base} {sec} {extra_part}".strip()

    base = "language:eng -sourceCountry:KOR"
    sec = section_cfg.overseas_query
    return f"{base} {sec} {extra_part}".strip()


# -----------------------------
# GDELT fetch
# -----------------------------
@st.cache_data(ttl=60 * 10, show_spinner=False)
def fetch_gdelt_articles(
    query: str,
    start_dt_utc: datetime,
    end_dt_utc: datetime,
    max_records: int,
) -> pd.DataFrame:
    def fmt(dt: datetime) -> str:
        return dt.astimezone(UTC).strftime("%Y%m%d%H%M%S")

    params = {
        "query": query,
        "mode": "ArtList",
        "format": "json",
        "maxrecords": int(max_records),
        "sort": "HybridRel",
        "startdatetime": fmt(start_dt_utc),
        "enddatetime": fmt(end_dt_utc),
    }

    headers = {"User-Agent": USER_AGENT}
    r = requests.get(GDELT_DOC_ENDPOINT, params=params, headers=headers, timeout=REQUEST_TIMEOUT)
    r.raise_for_status()
    data = r.json()

    rows = []
    for a in (data.get("articles", []) or []):
        url = a.get("url")
        rows.append(
            {
                "title": clean_text(a.get("title") or ""),
                "url": url,
                "seendate": a.get("seendate"),
                "published_utc": parse_seendate_utc(a.get("seendate")),
                "sourceCountry": a.get("sourceCountry"),
                "language": a.get("language"),
                "domain": normalize_domain(url) or "unknown",
            }
        )

    if not rows:
        return pd.DataFrame(columns=["title", "url", "seendate", "published_utc", "sourceCountry", "language", "domain"])

    df = pd.DataFrame(rows)
    df["published_utc"] = pd.to_datetime(df["published_utc"], utc=True, errors="coerce")
    df = df.dropna(subset=["published_utc"])
    df = df[df["title"] != ""]
    df = df.drop_duplicates(subset=["url"], keep="first")
    return df


# -----------------------------
# Summarization: 3 bullets + fallback meta
# -----------------------------
@st.cache_data(ttl=60 * 60, show_spinner=False)
def fetch_page_text_and_meta(url: str) -> Tuple[str, str]:
    if not url:
        return "", ""
    headers = {
        "User-Agent": USER_AGENT,
        "Accept-Language": "ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7",
    }
    try:
        r = requests.get(url, headers=headers, timeout=REQUEST_TIMEOUT)
        if r.status_code >= 400:
            return "", ""
        soup = BeautifulSoup(r.text, "lxml")

        desc = ""
        og = soup.find("meta", property="og:description")
        if og and og.get("content"):
            desc = clean_text(og.get("content"))

        if not desc:
            meta = soup.find("meta", attrs={"name": "description"})
            if meta and meta.get("content"):
                desc = clean_text(meta.get("content"))

        paras = soup.find_all("p")
        texts = []
        for p in paras[:10]:
            t = clean_text(p.get_text(" ", strip=True))
            if len(t) >= 40:
                texts.append(t)

        body = " ".join(texts)
        body = body[:2000]
        return body, desc
    except Exception:
        return "", ""


def split_sentences(text: str) -> List[str]:
    text = clean_text(text)
    if not text:
        return []
    parts = re.split(r"(?<=[\.\!\?])\s+|(?<=\n)\s*", text)
    out: List[str] = []
    for p in parts:
        p = clean_text(p)
        if 25 <= len(p) <= 220:
            out.append(p)

    seen = set()
    uniq = []
    for s in out:
        key = re.sub(r"[^0-9A-Za-z가-힣]+", "", s).lower()
        if key and key not in seen:
            seen.add(key)
            uniq.append(s)
    return uniq


def summarize_3_bullets(page_text: str, meta_desc: str) -> List[str]:
    sents = split_sentences(page_text)
    bullets: List[str] = []

    for s in sents:
        if len(bullets) >= 3:
            break
        low = s.lower()
        if any(k in low for k in ["cookies", "subscribe", "sign up", "광고", "저작권", "무단", "구독"]):
            continue
        bullets.append(s)

    if len(bullets) < 3 and meta_desc:
        md = clean_text(meta_desc)
        chunks = re.split(r"[•\-\|/]\s*", md)
        for c in chunks:
            c = clean_text(c)
            if 25 <= len(c) <= 220 and c not in bullets:
                bullets.append(c)
            if len(bullets) >= 3:
                break

    return bullets[:3]


# -----------------------------
# Dedup clustering (token Jaccard) - SAFE
# -----------------------------
STOPWORDS_KO = set(
    "그리고 그러나 또한 때문에 통해 관련 대한 따르면 경우 이번 오늘 내일 어제 기자 단독 속보 "
    "영상 사진 발표 밝혔다 말했다 예정 진행 가능 확대 감소 증가 정부 국회 대통령 "
    .split()
)
STOPWORDS_EN = set(
    "the a an and or but if then than this that those these to of in on for with without "
    "as from by at is are was were be been being it its into about after before over under "
    "says said say will would could should may might "
    .split()
)


def title_tokens(title: str) -> List[str]:
    t = clean_text(title).lower()
    t = re.sub(r"https?://\S+", " ", t)
    t = re.sub(r"[^0-9a-z가-힣\s]", " ", t)
    toks = [x for x in t.split() if len(x) >= 2]
    filtered: List[str] = []
    for x in toks:
        if re.fullmatch(r"\d+", x):
            filtered.append(x)
            continue
        if x in STOPWORDS_EN or x in STOPWORDS_KO:
            continue
        filtered.append(x)
    return filtered[:32]


def jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def dedup_by_title_cluster(df: pd.DataFrame, sim_threshold: float = 0.62) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    if "title" not in df.columns or "published_utc" not in df.columns:
        return df

    dfx = df.copy().sort_values("published_utc", ascending=False)

    kept_idx: List[int] = []
    cluster_reps: List[set] = []

    for idx, row in dfx.iterrows():
        toks = set(title_tokens(row.get("title", "")))
        if not toks:
            continue

        dup = False
        for rep in cluster_reps:
            if jaccard(toks, rep) >= sim_threshold:
                dup = True
                break

        if not dup:
            kept_idx.append(idx)
            cluster_reps.append(toks)

    if not kept_idx:
        return dfx.head(0)

    return dfx.loc[kept_idx].copy()


# -----------------------------
# Bias mapping apply + distribution
# -----------------------------
def apply_bias_mapping(df: pd.DataFrame, mapping: Dict[str, str]) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    out = df.copy()
    out["bias"] = out["domain"].map(lambda d: mapping.get((d or "").lower(), "미분류"))
    out["bias"] = out["bias"].where(out["bias"].isin(["보수", "중도", "진보"]), "미분류")
    return out


def distribution(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty or "bias" not in df.columns:
        return pd.DataFrame(columns=["bias", "count", "share"])
    dist = df.groupby("bias").size().reset_index(name="count")
    total = dist["count"].sum()
    dist["share"] = dist["count"] / total if total else 0
    dist["bias"] = pd.Categorical(dist["bias"], categories=BIAS_ORDER, ordered=True)
    return dist.sort_values("bias")


# -----------------------------
# Rendering
# -----------------------------
def render_top_list(section_name: str, top_df: pd.DataFrame, enable_summary: bool):
    st.subheader(f"{section_name} · Top {len(top_df)}")
    if top_df is None or top_df.empty:
        st.warning("해당 섹션에서 기사 후보를 찾지 못했습니다. (기간/키워드/범위 조정 필요)")
        return

    for idx, row in top_df.reset_index(drop=True).iterrows():
        title = row.get("title") or "(제목 없음)"
        url = row.get("url") or ""
        domain = row.get("domain") or "unknown"
        bias = row.get("bias") or "미분류"

        pub_str = ""
        try:
            pub_kst = pd.to_datetime(row.get("published_utc"), utc=True).tz_convert(KST)
            pub_str = pub_kst.strftime("%Y-%m-%d %H:%M (KST)")
        except Exception:
            pass

        bullets: List[str] = row.get("bullets") or []
        meta_desc = row.get("meta_desc") or ""

        if enable_summary:
            if bullets:
                summary_html = "<ul class='tight'>" + "".join([f"<li>{escape_html(b)}</li>" for b in bullets]) + "</ul>"
            elif meta_desc:
                summary_html = f"<small class='muted'>{escape_html(meta_desc)}</small>"
            else:
                summary_html = "<small class='muted'>요약을 불러오지 못했습니다(사이트 차단/메타정보/본문 부재 가능).</small>"
        else:
            summary_html = "<small class='muted'>요약 기능이 꺼져 있습니다.</small>"

        st.markdown(
            f"""
<div class="card">
  <div class="kv">
    <span class="badge">#{idx+1}</span>
    <span>성향: <b>{escape_html(bias)}</b></span>
    <span>도메인: <b>{escape_html(domain)}</b></span>
    <span>발행: <b>{escape_html(pub_str)}</b></span>
  </div>
  <h4>{escape_html(title)}</h4>
  <div>
    <a href="{url}" target="_blank" rel="noopener noreferrer">{url}</a>
  </div>
  <hr class="soft"/>
  <div>
    <b>핵심 요약 (3 bullets)</b><br/>
    {summary_html}
  </div>
</div>
""",
            unsafe_allow_html=True,
        )


# -----------------------------
# UI
# -----------------------------
st.title("섹션별 주요 뉴스 Top 5 + 성향 분포")
st.caption("국내/해외 선택 후 섹션별 Top 5를 ‘중복 제거 + 3줄 요약’으로 개선하고, 섹션별 성향 분포를 함께 보여줍니다. (데이터: GDELT)")

with st.sidebar:
    st.header("1) 범위 선택")
    region = st.radio("국내/해외", options=["국내", "해외"], horizontal=True)

    st.divider()
    st.header("2) 섹션 선택")
    section_names = [s.section for s in SECTIONS]
    selected_sections = st.multiselect(
        "분석할 섹션(복수 선택 가능)",
        options=section_names,
        default=section_names,
    )

    st.divider()
    st.header("3) Top 뉴스 구성")
    extra_keyword = st.text_input(
        "추가 키워드(선택)",
        value="",
        help="예: ‘탄소세’, ‘철강’, ‘원전’ 등을 넣으면 해당 이슈 중심으로 섹션별 Top 5가 구성됩니다.",
    )

    top_n = st.number_input("섹션별 Top N", min_value=3, max_value=10, value=5, step=1)

    candidate_pool = st.number_input(
        "섹션별 후보 기사 수(수집량)",
        min_value=60,
        max_value=500,
        value=220,
        step=10,
        help="각 섹션에서 Top N을 뽑기 전 GDELT에서 가져오는 후보 기사 수입니다.",
    )

    st.divider()
    st.header("품질 옵션")
    enable_summary = st.toggle("3줄 핵심 bullet 요약", value=True)
    sim_threshold = st.slider(
        "중복 제거 유사도 임계값(Jaccard)",
        min_value=0.45,
        max_value=0.80,
        value=0.62,
        step=0.01,
        help="값이 높을수록 ‘거의 같은 제목’만 중복으로 제거합니다. 0.60~0.70 권장.",
    )

    st.divider()
    st.header("성향 매핑")
    uploaded = st.file_uploader("매핑 CSV 업로드 (domain,bias)", type=["csv"])
    if uploaded is not None:
        try:
            map_df = pd.read_csv(uploaded)[["domain", "bias"]].dropna()
        except Exception:
            st.warning("CSV를 읽지 못했습니다. columns: domain,bias 형태인지 확인하세요.")
            map_df = default_bias_mapping_df()
    else:
        map_df = default_bias_mapping_df()

    edited_map_df = st.data_editor(
        map_df,
        num_rows="dynamic",
        use_container_width=True,
        column_config={
            "domain": st.column_config.TextColumn("domain"),
            "bias": st.column_config.SelectboxColumn("bias", options=["보수", "중도", "진보"]),
        },
    )

    mapping_dict = {
        str(r["domain"]).strip().lower(): str(r["bias"]).strip()
        for _, r in edited_map_df.dropna().iterrows()
        if str(r.get("domain", "")).strip() and str(r.get("bias", "")).strip()
    }

    st.divider()
    debug = st.toggle("디버그 표시(쿼리/건수)", value=False)

    run = st.button("직전 24시간 섹션별 Top 뉴스 생성", type="primary", use_container_width=True)

if not run:
    st.info("좌측에서 범위/섹션/옵션을 선택한 뒤 실행하세요.")
    st.stop()

if not selected_sections:
    st.warning("최소 1개 섹션을 선택해야 합니다.")
    st.stop()

start_utc, end_utc = rolling_24h_range_utc()
start_kst = start_utc.astimezone(KST)
end_kst = end_utc.astimezone(KST)

st.markdown(f"### {region} · 섹션별 Top {int(top_n)}")
st.caption(f"수집 기간: {start_kst.strftime('%Y-%m-%d %H:%M')} ~ {end_kst.strftime('%Y-%m-%d %H:%M')} (KST, 직전 24시간)")

section_cfg_map: Dict[str, SectionQuery] = {s.section: s for s in SECTIONS}
results: Dict[str, Dict[str, pd.DataFrame]] = {}

with st.spinner("섹션별 기사 후보를 수집/정제 중입니다..."):
    for sec_name in selected_sections:
        cfg = section_cfg_map[sec_name]
        q = build_section_query(region, cfg, extra_keyword)

        try:
            df = fetch_gdelt_articles(
                query=q,
                start_dt_utc=start_utc,
                end_dt_utc=end_utc,
                max_records=int(candidate_pool),
            )
        except Exception:
            df = pd.DataFrame(columns=["title", "url", "seendate", "published_utc", "sourceCountry", "language", "domain"])

        if debug:
            st.write(f"[DEBUG] {sec_name} query = {q}")
            st.write(f"[DEBUG] {sec_name} fetched rows = {len(df)}")

        # 성향 매핑
        df = apply_bias_mapping(df, mapping_dict)

        # 중복 제거(컬럼 보존 안전)
        df_dedup = dedup_by_title_cluster(df, sim_threshold=float(sim_threshold))

        # 방어: published_utc 없으면 빈 DF로
        if df_dedup is None or "published_utc" not in df_dedup.columns:
            df_dedup = df.head(0).copy()

        if not df_dedup.empty:
            df_dedup = df_dedup.sort_values("published_utc", ascending=False)

        top_df = df_dedup.head(int(top_n)).copy()

        # 3줄 요약
        if enable_summary and not top_df.empty:
            top_df["bullets"] = None
            top_df["meta_desc"] = ""
            for i in range(len(top_df)):
                url = top_df.iloc[i].get("url")
                time.sleep(0.12)
                page_text, meta_desc = fetch_page_text_and_meta(url)
                bullets = summarize_3_bullets(page_text, meta_desc)
                top_df.iat[i, top_df.columns.get_loc("bullets")] = bullets
                top_df.iat[i, top_df.columns.get_loc("meta_desc")] = meta_desc or ""

        dist_df = distribution(df_dedup)

        results[sec_name] = {
            "candidates": df_dedup,
            "top": top_df,
            "dist": dist_df,
            "query": pd.DataFrame([{"query": q}]),
        }

# Render tabs
tabs = st.tabs(selected_sections)
for tab, sec_name in zip(tabs, selected_sections):
    with tab:
        q = results[sec_name]["query"].iloc[0]["query"]
        st.markdown(f"<small class='muted'>사용 쿼리: {escape_html(clean_text(q))}</small>", unsafe_allow_html=True)

        cands = results[sec_name]["candidates"]
        dist_df = results[sec_name]["dist"]

        c1, c2, c3 = st.columns(3)
        c1.metric("후보 기사(중복 제거 후)", f"{len(cands):,}" if cands is not None else "0")
        unknown_share = dist_df.loc[dist_df["bias"] == "미분류", "share"].sum() if dist_df is not None and not dist_df.empty else 0
        c2.metric("미분류 비율", f"{unknown_share*100:.1f}%")
        c3.metric("고유 도메인", f"{cands['domain'].nunique(dropna=True):,}" if cands is not None and not cands.empty and "domain" in cands.columns else "0")

        if dist_df is not None and not dist_df.empty:
            fig = px.bar(dist_df, x="bias", y="count", text=dist_df["share"].map(lambda x: f"{x*100:.1f}%"))
            fig.update_layout(xaxis_title="성향", yaxis_title="기사 수", showlegend=False, height=320)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("성향 분포를 만들 데이터가 없습니다.")

        st.divider()
        render_top_list(sec_name, results[sec_name]["top"], enable_summary)

        with st.expander("진단: 후보 기사(중복 제거 후) 미리보기", expanded=False):
            if cands is None or cands.empty:
                st.write("후보 기사가 없습니다.")
            else:
                cols = [c for c in ["published_utc", "bias", "domain", "title", "url", "language", "sourceCountry"] if c in cands.columns]
                st.dataframe(cands[cols].head(60), use_container_width=True, height=420)

st.caption(
    "주의: (1) 섹션 분류는 섹션별 대표 키워드 기반이며, (2) 요약은 웹페이지 접근 가능 범위에서만 생성됩니다. "
    "정확도를 더 높이려면 ‘언론사별 RSS/섹션 URL’ 기반 수집으로 확장하는 것을 권장합니다."
)
