from pathlib import Path
import markdown
from jinja2 import Environment, FileSystemLoader
from markupsafe import Markup

# this file lives at questionagent/tools/template.py, so its own package's
# templates folder is two levels up: questionagent/templates/
TEMPLATE_DIR = Path(__file__).resolve().parent.parent / "templates"
env = Environment(loader=FileSystemLoader(str(TEMPLATE_DIR)))


def _markdown_to_html(text: str) -> str:
    """Convert the LLM's markdown answer into real HTML."""
    return markdown.markdown(text, extensions=["fenced_code", "tables", "nl2br"])


def fill_template(
    question: str,
    answer: str,
    sources: list[dict],
    output_path: str = "output/answer.html",
) -> str:
    """
    Render the final answer into the HTML template and save it.

    Args:
        question: the original question, used as the page title/heading.
        answer: the LLM's final answer text.
        sources: list of {title, url, ...} dicts, e.g. from web_access results.
        output_path: where to save the finished HTML file.

    Returns:
        The path the file was saved to.
    """
    template = env.get_template("answer_template.html")

    html = template.render(
        question=question,
        answer_html=Markup(_markdown_to_html(answer)),
        sources=sources,
    )

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)

    return output_path
