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


# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="오늘 섹션별 Top 5 (국내/해외)",
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
USER_AGENT = "Mozilla/5.0 (compatible; StreamlitSectionTop5/1.0; +https://streamlit.io)"
REQUEST_TIMEOUT = 10  # seconds


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
# Helpers
# -----------------------------
def clean_text(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip()


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


def kst_today_range_utc() -> Tuple[datetime, datetime]:
    now_kst = datetime.now(KST)
    start_kst = now_kst.replace(hour=0, minute=0, second=0, microsecond=0)
    return start_kst.astimezone(UTC), now_kst.astimezone(UTC)


@st.cache_data(ttl=60 * 10, show_spinner=False)  # 10 minutes
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

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    df["published_utc"] = pd.to_datetime(df["published_utc"], utc=True, errors="coerce")
    df = df.dropna(subset=["published_utc"])
    df = df[df["title"] != ""]
    df = df.drop_duplicates(subset=["url"], keep="first")
    return df


@st.cache_data(ttl=60 * 60, show_spinner=False)  # 1 hour
def fetch_meta_description(url: str) -> Optional[str]:
    if not url:
        return None
    headers = {
        "User-Agent": USER_AGENT,
        "Accept-Language": "ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7",
    }
    try:
        r = requests.get(url, headers=headers, timeout=REQUEST_TIMEOUT)
        if r.status_code >= 400:
            return None
        soup = BeautifulSoup(r.text, "lxml")

        og = soup.find("meta", property="og:description")
        if og and og.get("content"):
            return clean_text(og.get("content"))

        meta = soup.find("meta", attrs={"name": "description"})
        if meta and meta.get("content"):
            return clean_text(meta.get("content"))

        p = soup.find("p")
        if p and p.get_text(strip=True):
            return clean_text(p.get_text(strip=True))[:260]

        return None
    except Exception:
        return None


def build_section_query(region: str, section_cfg: SectionQuery, extra_keyword: str) -> str:
    """
    - 국내: sourceCountry:KOR + 한국어 섹션 키워드 (+ optional extra keyword)
    - 해외: language:eng -sourceCountry:KOR + 영어 섹션 키워드 (+ optional extra keyword)
    """
    extra = clean_text(extra_keyword)
    extra_part = f'("{extra}")' if extra else ""

    if region == "국내":
        base = "sourceCountry:KOR"
        sec = section_cfg.domestic_query
        return f"{base} {sec} {extra_part}".strip()

    base = "language:eng -sourceCountry:KOR"
    sec = section_cfg.overseas_query
    return f"{base} {sec} {extra_part}".strip()


def rank_and_pick_top(df: pd.DataFrame, top_n: int) -> pd.DataFrame:
    """
    GDELT HybridRel 기반 반환을 받되, 화면에서는 최신성을 조금 더 반영.
    """
    if df.empty:
        return df
    df = df.copy()
    df = df.sort_values("published_utc", ascending=False)
    return df.head(top_n)


def render_top_list(section_name: str, top_df: pd.DataFrame, enable_summary: bool):
    st.subheader(f"{section_name} · Top {len(top_df)}")
    if top_df.empty:
        st.warning("해당 섹션에서 오늘 기사 후보를 찾지 못했습니다. (키워드/범위 조정 필요)")
        return

    for idx, row in top_df.reset_index(drop=True).iterrows():
        title = row.get("title") or "(제목 없음)"
        url = row.get("url") or ""
        domain = row.get("domain") or "unknown"
        pub_kst = row.get("published_utc").tz_convert(KST) if hasattr(row.get("published_utc"), "tz_convert") else None
        pub_str = pub_kst.strftime("%H:%M (KST)") if pub_kst is not None else ""

        summary = row.get("summary") or ""

        st.markdown(
            f"""
<div class="card">
  <div class="kv">
    <span class="badge">#{idx+1}</span>
    <span>도메인: <b>{domain}</b></span>
    <span>발행: <b>{pub_str}</b></span>
  </div>
  <h4>{title}</h4>
  <div>
    <a href="{url}" target="_blank" rel="noopener noreferrer">{url}</a>
  </div>
  <hr class="soft"/>
  <div>
    <b>핵심 요약</b><br/>
    {"<small class='muted'>요약 기능이 꺼져 있습니다.</small>" if not enable_summary else (
        "<small class='muted'>요약을 불러오지 못했습니다(사이트 차단/메타정보 부재 가능).</small>" if not summary else summary
    )}
  </div>
</div>
""",
            unsafe_allow_html=True,
        )


# -----------------------------
# UI
# -----------------------------
st.title("오늘 섹션별 주요 뉴스 Top 5")
st.caption("국내/해외 선택 후, 섹션별(정치·경제·사회·국제·스포츠)로 오늘 Top 5를 요약 정리합니다. (데이터: GDELT)")

with st.sidebar:
    st.header("1) 범위 선택")
    region = st.radio("국내/해외", options=["국내", "해외"], horizontal=True)

    st.divider()
    st.header("2) 섹션 선택")
    section_names = [s.section for s in SECTIONS]
    selected_sections = st.multiselect(
        "분석할 섹션(복수 선택 가능)",
        options=section_names,
        default=section_names,  # 기본: 전체 섹션
    )

    st.markdown(
        '<small class="muted">섹션은 GDELT에 “편집국 섹션”이 직접 제공되지 않으므로, 섹션별 대표 키워드 쿼리로 구성합니다.</small>',
        unsafe_allow_html=True,
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
        min_value=50,
        max_value=400,
        value=180,
        step=10,
        help="각 섹션에서 Top N을 뽑기 전 GDELT에서 가져오는 후보 기사 수입니다.",
    )

    enable_summary = st.toggle(
        "기사 핵심요약(메타디스크립션) 가져오기",
        value=True,
        help="사이트 차단/속도 저하가 있을 수 있습니다. 끄면 타이틀 중심으로만 표시합니다.",
    )

    run = st.button("오늘 섹션별 Top 뉴스 생성", type="primary", use_container_width=True)

if not run:
    st.info("좌측에서 범위와 섹션을 선택한 뒤, ‘오늘 섹션별 Top 뉴스 생성’을 눌러 주세요.")
    st.stop()

if not selected_sections:
    st.warning("최소 1개 섹션을 선택해야 합니다.")
    st.stop()

start_utc, end_utc = kst_today_range_utc()
today_kst = datetime.now(KST).strftime("%Y-%m-%d")

st.markdown(f"### {today_kst} · {region} · 섹션별 Top {int(top_n)}")
st.caption("수집 기간: 오늘 00:00 ~ 현재 (KST)")

# Build a quick lookup for section configs
section_cfg_map: Dict[str, SectionQuery] = {s.section: s for s in SECTIONS}

results: Dict[str, pd.DataFrame] = {}

with st.spinner("섹션별 기사 후보를 수집 중입니다..."):
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
        except requests.HTTPError as e:
            st.error(f"[{sec_name}] GDELT 요청 실패(HTTPError): {e}")
            df = pd.DataFrame()
        except Exception as e:
            st.error(f"[{sec_name}] GDELT 요청 실패: {e}")
            df = pd.DataFrame()

        # (국내) 안전장치: sourceCountry=KOR만 유지
        if region == "국내" and not df.empty:
            df = df[df["sourceCountry"].fillna("").str.upper() == "KOR"]

        top_df = rank_and_pick_top(df, int(top_n))

        # Summaries (optional) - only for selected top rows
        if enable_summary and not top_df.empty:
            top_df = top_df.copy()
            top_df["summary"] = ""
            for i in range(len(top_df)):
                url = top_df.iloc[i]["url"]
                time.sleep(0.12)  # polite delay
                top_df.iat[i, top_df.columns.get_loc("summary")] = fetch_meta_description(url) or ""

        results[sec_name] = top_df

# Render: tabs per section
tabs = st.tabs(selected_sections)
for tab, sec_name in zip(tabs, selected_sections):
    with tab:
        # show query diagnostics
        cfg = section_cfg_map[sec_name]
        q = build_section_query(region, cfg, extra_keyword)
        st.markdown(f"<small class='muted'>사용 쿼리: {clean_text(q)}</small>", unsafe_allow_html=True)

        render_top_list(sec_name, results.get(sec_name, pd.DataFrame()), enable_summary)

st.caption(
    "주의: ‘Top’은 GDELT 수집/정렬(HybridRel)과 최신성 기준의 휴리스틱으로 선정된 대표 기사입니다. "
    "포털/편집국의 ‘메인 Top’과 완전히 동일하지 않을 수 있습니다."
)
