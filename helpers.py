""" Small, dependency-free helper functions used across the project."""
import re 


def sanitize_filename(name: str) -> str:
    """make an uploaded filename for safe to store disk (no path tricks, no add chars)."""
    name = name.replace("\\", "/").split("/")[-1]
    name = re.sub(r"[^A-Za-z0-9._ -]", "_", name)
    return name.strip() or "document.pdf"


def truncate(text: str, limit: int = 400) -> str:
    """shorten long text fot display, adding an ellipsis, when something we cut."""
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit -1].rstrip() + "…"


def evenly_sample(items: list, max_items: int) -> list:
    """pick at max_items elements spread evenly across the list (keeps order)."""
    if len(items) <= max_items:
        return items
    step = len(items) / max_items
    return [items[int(i * step)] for i in range(max_items)]