import re
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple, Any

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
    page_title="섹션별 Top 5 + 성향 분포 (전날 기준 / Parse Fix)",
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
code.small { font-size: 0.85rem; }
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
USER_AGENT = "Mozilla/5.0 (compatible; StreamlitSectionTop5/2.6; +https://streamlit.io)"
REQUEST_TIMEOUT = 15  # seconds

BIAS_ORDER = ["보수", "중도", "진보", "미분류"]
EMPTY_COLUMNS = ["title", "url", "seendate", "published_raw", "published_utc", "sourceCountry", "language", "domain"]


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
        domestic_query="(politics OR government OR parliament OR national assembly OR president OR election OR ruling party OR opposition)",
        overseas_query="(politics OR government OR parliament OR congress OR president OR election OR campaign)",
    ),
    SectionQuery(
        section="경제",
        domestic_query="(economy OR markets OR stocks OR exchange rate OR interest rates OR inflation OR prices OR companies OR industry OR semiconductor)",
        overseas_query="(economy OR markets OR stocks OR inflation OR interest rates OR central bank OR currency OR business OR industry OR semiconductor)",
    ),
    SectionQuery(
        section="사회",
        domestic_query="(crime OR accident OR disaster OR education OR health OR welfare OR labor OR strike OR court OR prosecutors OR police)",
        overseas_query="(society OR crime OR accident OR disaster OR education OR health OR welfare OR labor OR strike OR court OR police)",
    ),
    SectionQuery(
        section="국제",
        domestic_query="(diplomacy OR summit OR UN OR United Nations OR United States OR China OR Japan OR Russia OR Ukraine OR Middle East OR Gaza)",
        overseas_query="(world OR international OR diplomacy OR summit OR UN OR Ukraine OR Russia OR China OR Japan OR Middle East OR Gaza)",
    ),
    SectionQuery(
        section="스포츠",
        domestic_query="(sports OR soccer OR football OR baseball OR basketball OR volleyball OR golf OR esports OR Olympics OR World Cup)",
        overseas_query="(sports OR football OR soccer OR baseball OR basketball OR Olympics OR World Cup OR NBA OR MLB OR NHL)",
    ),
]


# -----------------------------
# Bias mapping
# -----------------------------
def default_bias_mapping_df() -> pd.DataFrame:
    data = [
        ("chosun.com", "보수"),
        ("donga.com", "보수"),
        ("joongang.co.kr", "중도"),
        ("mk.co.kr", "중도"),
        ("yonhapnews.co.kr", "중도"),
        ("hani.co.kr", "진보"),
        ("khan.co.kr", "진보"),
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
# Helpers
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


def yesterday_kst_range_utc() -> Tuple[datetime, datetime]:
    now_kst = datetime.now(KST)
    today_start_kst = now_kst.replace(hour=0, minute=0, second=0, microsecond=0)
    start_kst = today_start_kst - timedelta(days=1)
    end_kst = today_start_kst
    return start_kst.astimezone(UTC), end_kst.astimezone(UTC)


# -----------------------------
# Query builder
# -----------------------------
def build_section_query_candidates(region: str, section_cfg: SectionQuery, extra_keyword: str) -> List[str]:
    extra = clean_text(extra_keyword)

    extra_parts = []
    if extra:
        extra_parts.append(f'("{extra}")')
    extra_parts.append("")  # fallback extra 제거

    queries: List[str] = []

    if region == "국내":
        lang_candidates = ["sourcelang:kor", "sourcelang:korean", "sourcelang:Korean"]
        for lang in lang_candidates:
            for extra_part in extra_parts:
                queries.append(f"{lang} {section_cfg.domestic_query} {extra_part}".strip())
        # 최후 폴백: 언어 제한 제거
        for extra_part in extra_parts:
            queries.append(f"{section_cfg.domestic_query} {extra_part}".strip())
        return queries

    lang_candidates = ["sourcelang:eng", "sourcelang:english", "sourcelang:English"]
    for lang in lang_candidates:
        for extra_part in extra_parts:
            queries.append(f"{lang} {section_cfg.overseas_query} {extra_part}".strip())
    for extra_part in extra_parts:
        queries.append(f"{section_cfg.overseas_query} {extra_part}".strip())
    return queries


# -----------------------------
# Robust datetime parsing
# -----------------------------
def pick_published_raw(article: Dict[str, Any]) -> Optional[str]:
    """
    DOC 응답에서 시간 필드가 seendate 외에도 다른 이름으로 올 수 있어 후보를 순차 시도.
    """
    for key in ["seendate", "seenDate", "datetime", "date", "published", "publicationDate"]:
        v = article.get(key)
        if v:
            return str(v)
    return article.get("seendate")


def parse_published_utc(raw: Optional[str]) -> Optional[pd.Timestamp]:
    """
    - 14자리 숫자(YYYYMMDDHHMMSS)도 처리
    - ISO8601 등은 pandas.to_datetime으로 처리
    """
    if not raw:
        return None
    s = str(raw).strip()
    # 14자리 숫자 형태
    if re.fullmatch(r"\d{14}", s):
        ts = pd.to_datetime(s, format="%Y%m%d%H%M%S", utc=True, errors="coerce")
        return ts if pd.notna(ts) else None
    # 그 외(ISO8601 등)
    ts = pd.to_datetime(s, utc=True, errors="coerce")
    return ts if pd.notna(ts) else None


# -----------------------------
# GDELT fetch (핵심 수정: published 파싱)
# -----------------------------
@st.cache_data(ttl=60 * 10, show_spinner=False)
def fetch_gdelt_articles(
    query: str,
    start_dt_utc: datetime,
    end_dt_utc: datetime,
    max_records: int,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    def fmt(dt: datetime) -> str:
        return dt.astimezone(UTC).strftime("%Y%m%d%H%M%S")

    params = {
        "query": query,
        "mode": "artlist",
        "format": "json",
        "maxrecords": int(max_records),
        "sort": "datedesc",
        "startdatetime": fmt(start_dt_utc),
        "enddatetime": fmt(end_dt_utc),
    }

    headers = {"User-Agent": USER_AGENT}

    dbg: Dict[str, Any] = {
        "status_code": None,
        "final_url": None,
        "top_keys": None,
        "error": None,
        "message": None,
        "articles_count": None,
        "sample_article_keys": None,
        "sample_published_raw": None,
    }

    r = requests.get(GDELT_DOC_ENDPOINT, params=params, headers=headers, timeout=REQUEST_TIMEOUT)
    dbg["status_code"] = r.status_code
    dbg["final_url"] = r.url
    r.raise_for_status()

    data = r.json()
    dbg["top_keys"] = sorted(list(data.keys())) if isinstance(data, dict) else [type(data).__name__]
    dbg["error"] = data.get("error") if isinstance(data, dict) else None
    dbg["message"] = data.get("message") if isinstance(data, dict) else None

    articles = (data.get("articles", []) or []) if isinstance(data, dict) else []
    dbg["articles_count"] = len(articles)

    if articles:
        dbg["sample_article_keys"] = sorted(list(articles[0].keys()))
        raw0 = pick_published_raw(articles[0]) or ""
        dbg["sample_published_raw"] = raw0

    rows = []
    for a in articles:
        url = a.get("url")
        title = clean_text(a.get("title") or "")
        published_raw = pick_published_raw(a)
        published_ts = parse_published_utc(published_raw)

        rows.append(
            {
                "title": title,
                "url": url,
                "seendate": a.get("seendate"),
                "published_raw": published_raw,
                "published_utc": published_ts,
                "sourceCountry": a.get("sourceCountry"),
                "language": a.get("language"),
                "domain": normalize_domain(url) or "unknown",
            }
        )

    if not rows:
        return pd.DataFrame(columns=EMPTY_COLUMNS), dbg

    df = pd.DataFrame(rows)

    # IMPORTANT: 여기서부터 "필터로 전부 날려버리는" 문제를 막기 위해 단계적 정제
    # 1) title 없는 것만 제거
    df = df[df["title"].fillna("").str.len() > 0]

    # 2) published_utc 파싱 실패가 많으면 여기서 전부 날아갈 수 있으므로
    #    우선 dropna를 하지 말고, 정렬 가능한 것만 우선 사용
    df["published_utc"] = pd.to_datetime(df["published_utc"], utc=True, errors="coerce")

    # 3) url 중복 제거는 유지
    df = df.drop_duplicates(subset=["url"], keep="first")

    return df, dbg


# -----------------------------
# Summarization
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

        body = " ".join(texts)[:2000]
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
# Dedup clustering (SAFE)
# -----------------------------
STOPWORDS_KO = set(
    "그리고 그러나 또한 때문에 통해 관련 대한 따르면 경우 이번 오늘 내일 어제 기자 단독 속보 "
    "영상 사진 발표 밝혔다 말했다 예정 진행 가능 확대 감소 증가 정부 국회 대통령 ".split()
)
STOPWORDS_EN = set(
    "the a an and or but if then than this that those these to of in on for with without "
    "as from by at is are was were be been being it its into about after before over under "
    "says said say will would could should may might ".split()
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
    if "title" not in df.columns:
        return df

    # published_utc가 NaT일 수 있으니, 정렬키를 보정
    dfx = df.copy()
    if "published_utc" in dfx.columns:
        dfx["_sort_key"] = dfx["published_utc"].fillna(pd.Timestamp.min.tz_localize("UTC"))
        dfx = dfx.sort_values("_sort_key", ascending=False).drop(columns=["_sort_key"])
    else:
        dfx = dfx

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
# Bias mapping + distribution
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
        st.warning("해당 섹션에서 기사 후보를 찾지 못했습니다.")
        return

    for idx, row in top_df.reset_index(drop=True).iterrows():
        title = row.get("title") or "(제목 없음)"
        url = row.get("url") or ""
        domain = row.get("domain") or "unknown"
        bias = row.get("bias") or "미분류"

        pub_str = ""
        try:
            if pd.notna(row.get("published_utc")):
                pub_kst = pd.to_datetime(row.get("published_utc"), utc=True).tz_convert(KST)
                pub_str = pub_kst.strftime("%Y-%m-%d %H:%M (KST)")
            else:
                pub_str = "(시간 정보 없음)"
        except Exception:
            pub_str = "(시간 파싱 실패)"

        bullets: List[str] = row.get("bullets") or []
        meta_desc = row.get("meta_desc") or ""

        if enable_summary:
            if bullets:
                summary_html = "<ul class='tight'>" + "".join([f"<li>{escape_html(b)}</li>" for b in bullets]) + "</ul>"
            elif meta_desc:
                summary_html = f"<small class='muted'>{escape_html(meta_desc)}</small>"
            else:
                summary_html = "<small class='muted'>요약을 불러오지 못했습니다(차단/본문 부재 가능).</small>"
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
st.title("섹션별 주요 뉴스 Top 5 + 성향 분포 (전날 기준)")
st.caption("현재 문제는 API 응답은 오는데, 우리 쪽 파싱/필터에서 전부 제거되는 현상으로 보입니다. 이를 해결하기 위해 published_utc 파싱을 강화했습니다.")

with st.sidebar:
    region = st.radio("국내/해외", options=["국내", "해외"], horizontal=True)

    section_names = [s.section for s in SECTIONS]
    selected_sections = st.multiselect("분석할 섹션", options=section_names, default=section_names)

    extra_keyword = st.text_input("추가 키워드(선택)", value="")

    top_n = st.number_input("섹션별 Top N", min_value=3, max_value=10, value=5, step=1)
    candidate_pool = st.number_input("후보 기사 수", min_value=60, max_value=500, value=250, step=10)

    enable_summary = st.toggle("3줄 요약", value=True)
    sim_threshold = st.slider("중복 제거 임계값", 0.45, 0.80, 0.62, 0.01)

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

    debug = st.toggle("디버그(응답 구조/샘플 키)", value=True)
    run = st.button("전날 뉴스 생성", type="primary", use_container_width=True)

if not run:
    st.stop()

start_utc, end_utc = yesterday_kst_range_utc()
start_kst = start_utc.astimezone(KST)
end_kst = end_utc.astimezone(KST)
st.caption(f"수집 기간: {start_kst.strftime('%Y-%m-%d %H:%M')} ~ {end_kst.strftime('%Y-%m-%d %H:%M')} (KST)")

# 연결 테스트
with st.expander("진단: 연결 테스트", expanded=True):
    try:
        df_t, dbg_t = fetch_gdelt_articles('"Korea"', start_utc, end_utc, 5)
        st.write("status_code:", dbg_t.get("status_code"))
        st.write("articles_count:", dbg_t.get("articles_count"))
        st.write("top_keys:", dbg_t.get("top_keys"))
        st.write("sample_article_keys:", dbg_t.get("sample_article_keys"))
        st.write("sample_published_raw:", dbg_t.get("sample_published_raw"))
        st.write("final_url:", dbg_t.get("final_url"))
        st.write("df rows (after parsing):", len(df_t))
        if not df_t.empty:
            st.dataframe(df_t[["published_raw", "published_utc", "domain", "title", "url"]], use_container_width=True)
        else:
            st.warning("DataFrame이 비어 있습니다. sample_published_raw가 어떤 형태인지 확인하세요.")
    except Exception as e:
        st.error(f"연결 테스트 실패: {repr(e)}")

# 섹션 처리
section_cfg_map: Dict[str, SectionQuery] = {s.section: s for s in SECTIONS}
results: Dict[str, Dict[str, Any]] = {}

with st.spinner("섹션별 수집/정제 중..."):
    for sec_name in selected_sections:
        cfg = section_cfg_map[sec_name]
        cands = build_section_query_candidates(region, cfg, extra_keyword)

        df = pd.DataFrame(columns=EMPTY_COLUMNS)
        used_q = cands[0]
        used_dbg = {}
        last_exc = None

        for q in cands:
            used_q = q
            try:
                df_try, dbg_try = fetch_gdelt_articles(q, start_utc, end_utc, int(candidate_pool))
                used_dbg = dbg_try
                last_exc = None
            except Exception as e:
                df_try = pd.DataFrame(columns=EMPTY_COLUMNS)
                dbg_try = {}
                used_dbg = dbg_try
                last_exc = repr(e)

            if debug:
                st.write(f"[DEBUG] {sec_name} query = {q}")
                if used_dbg:
                    st.write(
                        {
                            "status_code": used_dbg.get("status_code"),
                            "articles_count": used_dbg.get("articles_count"),
                            "sample_published_raw": used_dbg.get("sample_published_raw"),
                        }
                    )
                    st.write("final_url:", used_dbg.get("final_url"))
                if last_exc:
                    st.write("[DEBUG] exception:", last_exc)
                st.write(f"[DEBUG] df_try rows(after parsing)={len(df_try)}")

            if not df_try.empty:
                df = df_try
                break

        df = apply_bias_mapping(df, mapping_dict)

        df_dedup = dedup_by_title_cluster(df, float(sim_threshold))
        top_df = df_dedup.head(int(top_n)).copy()

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
            "query": used_q,
            "dbg": used_dbg,
            "candidates": df_dedup,
            "top": top_df,
            "dist": dist_df,
        }

tabs = st.tabs(selected_sections)
for tab, sec_name in zip(tabs, selected_sections):
    with tab:
        st.markdown(f"<small class='muted'>최종 사용 쿼리: {escape_html(results[sec_name]['query'])}</small>", unsafe_allow_html=True)

        cands = results[sec_name]["candidates"]
        dist_df = results[sec_name]["dist"]

        c1, c2, c3 = st.columns(3)
        c1.metric("후보(중복 제거 후)", f"{len(cands):,}")
        unknown_share = dist_df.loc[dist_df["bias"] == "미분류", "share"].sum() if not dist_df.empty else 0
        c2.metric("미분류 비율", f"{unknown_share*100:.1f}%")
        c3.metric("고유 도메인", f"{cands['domain'].nunique(dropna=True):,}" if not cands.empty else "0")

        if not dist_df.empty:
            fig = px.bar(dist_df, x="bias", y="count", text=dist_df["share"].map(lambda x: f"{x*100:.1f}%"))
            fig.update_layout(xaxis_title="성향", yaxis_title="기사 수", showlegend=False, height=320)
            st.plotly_chart(fig, use_container_width=True)

        st.divider()
        render_top_list(sec_name, results[sec_name]["top"], enable_summary)
