from common.schemas.paper import PaperMetadata


def deduplicate_papers(papers: list[PaperMetadata]) -> list[PaperMetadata]:
    seen_dois: set[str] = set()
    seen_titles: set[str] = set()
    unique: list[PaperMetadata] = []

    for paper in papers:
        norm_doi = paper.doi.strip().lower() if paper.doi else None
        norm_title = "".join(c.lower() for c in paper.title if c.isalnum())
        if norm_doi and norm_doi in seen_dois:
            continue
        if norm_title in seen_titles:
            continue
        if norm_doi:
            seen_dois.add(norm_doi)
        seen_titles.add(norm_title)
        unique.append(paper)
    return unique
