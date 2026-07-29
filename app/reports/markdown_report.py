from app.matching.matcher import MatchResult


def build_markdown_report(results: list[MatchResult]) -> str:
    lines = [
        "# Отчет по подбору складских позиций",
        "",
    ]

    if not results:
        lines.append("Подходящие позиции не найдены.")
        return "\n".join(lines)

    for result in results:
        lines.extend(
            [
                f"## {result.spec_item.name}",
                "",
                f"- Дистрибьютор: {result.offer.distributor}",
                f"- SKU: {result.offer.sku}",
                f"- Позиция: {result.offer.name}",
                f"- Оценка совпадения: {result.score:.2f}",
                f"- Причина: {result.reason}",
                "",
            ],
        )

    return "\n".join(lines)
