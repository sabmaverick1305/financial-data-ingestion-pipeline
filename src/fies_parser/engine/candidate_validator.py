"""Structural sanity checks on a `ParserCandidate`.

Deliberately narrow: uniqueness of ids, resolvable cross-references, and
well-formed geometry. This is NOT financial validation — value correctness,
unit checks, and business rules belong to a later layer, never here.
"""

from __future__ import annotations

from fies_parser.canonical.candidate_models import CandidateElement, CandidatePage, CandidateTable, ParserCandidate


class CandidateValidator:
    """Runs a fixed set of structural checks and returns human-readable issues.

    An empty list means the candidate is structurally sound. Callers decide
    what to do with issues (`ParserEngine` raises `InvalidCandidateError`).
    """

    def validate(self, candidate: ParserCandidate) -> list[str]:
        issues: list[str] = []
        issues.extend(self._check_unique_element_ids(candidate.elements))
        issues.extend(self._check_unique_table_ids(candidate.tables))
        issues.extend(self._check_unique_page_numbers(candidate.pages))
        issues.extend(self._check_page_cross_references(candidate))
        issues.extend(self._check_table_page_ranges(candidate.tables))
        issues.extend(self._check_bounding_boxes(candidate))
        return issues

    def _check_unique_element_ids(self, elements: list[CandidateElement]) -> list[str]:
        seen: set[str] = set()
        issues: list[str] = []
        for element in elements:
            if element.element_id in seen:
                issues.append(f"duplicate element_id: {element.element_id!r}")
            seen.add(element.element_id)
        return issues

    def _check_unique_table_ids(self, tables: list[CandidateTable]) -> list[str]:
        seen: set[str] = set()
        issues: list[str] = []
        for table in tables:
            if table.table_id in seen:
                issues.append(f"duplicate table_id: {table.table_id!r}")
            seen.add(table.table_id)
        return issues

    def _check_unique_page_numbers(self, pages: list[CandidatePage]) -> list[str]:
        seen: set[int] = set()
        issues: list[str] = []
        for page in pages:
            if page.page_number in seen:
                issues.append(f"duplicate page_number: {page.page_number}")
            seen.add(page.page_number)
        return issues

    def _check_page_cross_references(self, candidate: ParserCandidate) -> list[str]:
        issues: list[str] = []
        elements_by_id = {element.element_id: element for element in candidate.elements}
        table_ids = {table.table_id for table in candidate.tables}

        for page in candidate.pages:
            for element_id in page.element_ids:
                element = elements_by_id.get(element_id)
                if element is None:
                    issues.append(f"page {page.page_number} references unknown element_id {element_id!r}")
                elif element.page_number != page.page_number:
                    issues.append(
                        f"page {page.page_number} lists element_id {element_id!r} which belongs to page {element.page_number}"
                    )
            for table_id in page.table_ids:
                if table_id not in table_ids:
                    issues.append(f"page {page.page_number} references unknown table_id {table_id!r}")
        return issues

    def _check_table_page_ranges(self, tables: list[CandidateTable]) -> list[str]:
        issues: list[str] = []
        for table in tables:
            if table.page_start > table.page_end:
                issues.append(f"table {table.table_id!r} has page_start > page_end ({table.page_start} > {table.page_end})")
        return issues

    def _check_bounding_boxes(self, candidate: ParserCandidate) -> list[str]:
        issues: list[str] = []
        for element in candidate.elements:
            if element.bbox and (element.bbox.x1 < element.bbox.x0 or element.bbox.y1 < element.bbox.y0):
                issues.append(f"element {element.element_id!r} has a degenerate bounding box")
        for table in candidate.tables:
            if table.bbox and (table.bbox.x1 < table.bbox.x0 or table.bbox.y1 < table.bbox.y0):
                issues.append(f"table {table.table_id!r} has a degenerate bounding box")
        return issues
