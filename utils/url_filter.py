import re
from typing import List, Tuple
from urllib.parse import urlparse


BLACKLIST_DOMAINS = {
    "patreon.com", "twitter.com", "x.com", "instagram.com",
    "discord.gg", "discord.com", "amzn.to", "bit.ly",
    "linkedin.com", "ko-fi.com", "gumroad.com", "udemy.com",
    "coursera.com", "tiktok.com", "facebook.com", "youtube.com",
    "youtu.be", "t.co",
}

URL_RE = re.compile(r"https?://[^\s\)\]>\"']+")


def extract_urls_from_description(description: str) -> List[str]:
    return URL_RE.findall(description)


def _domain(url: str) -> str:
    try:
        return urlparse(url).netloc.lstrip("www.")
    except Exception:
        return ""


def blacklist_filter(urls: List[str]) -> List[str]:
    return [u for u in urls if _domain(u) not in BLACKLIST_DOMAINS]


def llm_classify_urls(
    urls_with_context: List[Tuple[str, str]],
    client,
) -> List[str]:
    if not urls_with_context:
        return []
    items = [{"url": u, "context": c} for u, c in urls_with_context]
    system = (
        "You classify URLs from YouTube video descriptions. "
        "For each URL and its surrounding context, output label: "
        "'educational_reference' (papers, docs, repos, blog posts) or 'promotional' (courses, merch, social, donations). "
        "Return JSON: {\"classifications\": [{\"url\": \"...\", \"label\": \"...\"}]}"
    )
    user = f"Classify these URLs:\n{items}"
    result = client.complete_json(system=system, user=user)
    return [
        item["url"]
        for item in result.get("classifications", [])
        if item.get("label") == "educational_reference"
    ]


def _extract_context(description: str, url: str) -> str:
    idx = description.find(url)
    start = max(0, idx - 80)
    end = min(len(description), idx + len(url) + 80)
    return description[start:end]


def filter_description_urls(description: str, client) -> List[str]:
    all_urls = extract_urls_from_description(description)
    after_blacklist = blacklist_filter(all_urls)
    urls_with_context = [
        (url, _extract_context(description, url)) for url in after_blacklist
    ]
    return llm_classify_urls(urls_with_context, client)
