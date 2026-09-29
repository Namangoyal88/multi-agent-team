"""arXiv, Semantic Scholar, PubMed, web search (Tavily), GitHub, Kaggle adapters."""
from __future__ import annotations

import re
import xml.etree.ElementTree as ET

from .base import Adapter

_NS = {"a": "http://www.w3.org/2005/Atom"}


class ArxivAdapter(Adapter):
    name = "arxiv"

    def _search(self, query, limit):
        r = self._get("https://export.arxiv.org/api/query",
                      params={"search_query": f"all:{query}", "max_results": limit,
                              "sortBy": "relevance"})
        root = ET.fromstring(r.text)
        out = []
        for e in root.findall("a:entry", _NS):
            url = (e.findtext("a:id", "", _NS) or "").strip()
            title = re.sub(r"\s+", " ", e.findtext("a:title", "", _NS)).strip()
            if not title or "Error" == title:
                continue
            out.append({"source": "arxiv", "title": title, "url": url,
                        "arxiv_id": url.rsplit("/abs/", 1)[-1],
                        "abstract": re.sub(r"\s+", " ", e.findtext("a:summary", "", _NS)).strip(),
                        "authors": [a.findtext("a:name", "", _NS) for a in e.findall("a:author", _NS)],
                        "year": (e.findtext("a:published", "", _NS) or "")[:4]})
        return out


class SemanticScholarAdapter(Adapter):
    name = "semantic_scholar"

    def _search(self, query, limit):
        headers = {"x-api-key": self.keys["SEMANTIC_SCHOLAR_API_KEY"]} if self.keys.get(
            "SEMANTIC_SCHOLAR_API_KEY") else {}
        r = self._get("https://api.semanticscholar.org/graph/v1/paper/search", headers=headers,
                      params={"query": query, "limit": limit,
                              "fields": "title,abstract,year,authors,externalIds,url,citationCount,venue"})
        out = []
        for p in r.json().get("data", []):
            ext = p.get("externalIds") or {}
            out.append({"source": "semantic_scholar", "title": p.get("title", ""), "url": p.get("url", ""),
                        "abstract": p.get("abstract") or "", "year": str(p.get("year") or ""),
                        "authors": [a.get("name", "") for a in p.get("authors", [])],
                        "doi": ext.get("DOI"), "arxiv_id": ext.get("ArXiv"),
                        "citations": p.get("citationCount"), "venue": p.get("venue")})
        return out


class PubMedAdapter(Adapter):
    name = "pubmed"
    base = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"

    def _search(self, query, limit):
        key = {"api_key": self.keys["NCBI_API_KEY"]} if self.keys.get("NCBI_API_KEY") else {}
        ids = self._get(self.base + "esearch.fcgi", params={"db": "pubmed", "term": query,
                        "retmax": limit, "retmode": "json", **key}).json()["esearchresult"]["idlist"]
        if not ids:
            return []
        root = ET.fromstring(self._get(self.base + "efetch.fcgi", params={
            "db": "pubmed", "id": ",".join(ids), "retmode": "xml", **key}).text)
        out = []
        for art in root.findall(".//PubmedArticle"):
            pmid = art.findtext(".//PMID", "")
            out.append({"source": "pubmed", "title": "".join(art.find(".//ArticleTitle").itertext()),
                        "abstract": " ".join("".join(t.itertext()) for t in art.findall(".//AbstractText")),
                        "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/", "pmid": pmid,
                        "year": art.findtext(".//PubDate/Year", ""),
                        "authors": [f"{a.findtext('ForeName', '')} {a.findtext('LastName', '')}".strip()
                                    for a in art.findall(".//Author")]})
        return out


class WebSearchAdapter(Adapter):
    name = "web_search"
    requires_key = "TAVILY_API_KEY"

    def _search(self, query, limit):
        r = self._post("https://api.tavily.com/search", json={
            "api_key": self.keys["TAVILY_API_KEY"], "query": query, "max_results": limit})
        return [{"source": "web", "title": x.get("title", ""), "url": x.get("url", ""),
                 "abstract": x.get("content", "")} for x in r.json().get("results", [])]


class GitHubAdapter(Adapter):
    name = "github"

    def _search(self, query, limit):
        h = {"Accept": "application/vnd.github+json"}
        if self.keys.get("GITHUB_TOKEN"):
            h["Authorization"] = f"Bearer {self.keys['GITHUB_TOKEN']}"
        r = self._get("https://api.github.com/search/repositories", headers=h,
                      params={"q": query, "per_page": limit, "sort": "stars"})
        return [{"source": "github", "title": x["full_name"], "url": x["html_url"],
                 "abstract": x.get("description") or "", "stars": x.get("stargazers_count"),
                 "year": (x.get("pushed_at") or "")[:4]} for x in r.json().get("items", [])]


class KaggleAdapter(Adapter):
    name = "kaggle"
    requires_key = "KAGGLE_KEY"

    def available(self):
        return bool(self.keys.get("KAGGLE_KEY") and self.keys.get("KAGGLE_USERNAME"))

    def _search(self, query, limit):
        r = self._get("https://www.kaggle.com/api/v1/datasets/list", params={"search": query},
                      auth=(self.keys["KAGGLE_USERNAME"], self.keys["KAGGLE_KEY"]))
        return [{"source": "kaggle", "title": d.get("title", ""), "url": d.get("url", ""),
                 "abstract": d.get("subtitle", "")} for d in r.json()[:limit]]
