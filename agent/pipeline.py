"""
Research-mode pipeline: plan -> search per goal -> shrink -> notes.

Your code runs the order. The model never decides what step comes next.
Mark mode does not use this file.
"""

import json
import logging
from pydantic import BaseModel

from .tools.web_access import web_access as raw_web_access
from .tools.scrape import scrape as raw_scrape

LOGGER_NAME = "questionagent"

MAX_GOALS = 5
RESULTS_PER_GOAL = 4
SCRAPE_CHARS = 12000  # shrink step handles long pages now, so this can be bigger than 5000

PLAN_PROMPT = """You are helping a student research a question. Your job right now is only to plan, not to answer.

The student's question:

{question}

First, separate two different things the question might contain:
- The DELIVERABLE - what format the student has to hand in (e.g. "make a poster", "write an essay", "create a presentation", "draw a diagram"). This describes the shape of their homework, not something to research. It is never its own clause or goal - searching for "poster" or "essay" itself would only turn up shopping results and templates, not actual information.
- The CONTENT - the actual facts, explanations, or topics the deliverable needs to contain. This is the only thing you plan around.

Work through this in order. Do not skip to the goal list before doing step 2 properly.

STEP 1 - Find every distinct content clause being asked for, before summarising anything. Ignore the deliverable entirely here.
Read the question slowly, clause by clause. A clause can be an explicitly numbered or listed part, but it can just as easily be a small phrase buried in the middle of a sentence, after a comma, or tucked onto the end - those are the ones that get missed by a fast read. Write down every one you find, in the question's own words, even ones that feel minor or secondary. Do not merge two distinct things into one just because they're related. Do not skip something because it seems like it might get covered as a side effect of answering another part.

STEP 2 - Restate the question in one or two sentences, in your own words, so the student can check you understood it.

STEP 3 - Turn each clause from step 1 into one short, searchable goal phrase (not a full sentence). One goal per clause, same count, same order, each one clearly traceable back to the clause it came from. Maximum {max_goals} - if the question genuinely has more than that, keep the {max_goals} most important and drop the rest, but do not silently merge distinct clauses together just to fit the limit.

Do not answer the question or state any facts anywhere in this response - only the clauses, the restatement, and the goals.

---

EXAMPLE 1 - a question that lists its own parts:

Question: "What are vpns for. Explain what vpns were first designed for, what are some common good use cases and some bad, eg student uses it to play games at school etc"

clauses:
- what VPNs were first designed for
- common good/legitimate use cases for VPNs
- common bad uses, e.g. a student using one to play games at school

restatement: "The question asks what VPNs are for: their original purpose, common legitimate uses, and misuse such as a student bypassing school restrictions."

goals:
- original purpose VPNs were designed for
- common legitimate uses of VPNs
- VPN misuse, e.g. bypassing school restrictions

---

EXAMPLE 2 - a vague question with no listed parts, where the parts have to be worked out:

Question: "why did the roman empire fall"

clauses:
- what internal/economic problems weakened the empire
- what external pressures (invasions, migrations) contributed
- whether there was one specific turning point or a long decline

restatement: "The question asks what caused the fall of the Roman Empire, covering internal weaknesses and outside pressures."

goals:
- internal economic and political problems in the late Roman Empire
- barbarian invasions and migrations affecting Rome
- timeline of Rome's decline - gradual collapse vs a specific turning point

---

EXAMPLE 3 - a question with a clause easy to miss, buried mid-sentence:

Question: "Discuss the causes of World War One, focusing especially on the alliance system that dragged multiple countries into a conflict that started as a regional dispute"

A fast read might only catch "causes of WWI" and miss that the question specifically calls out the alliance system and the regional-to-global escalation, not just causes in general.

clauses:
- general causes of World War One
- the alliance system specifically, and how it pulled in multiple countries
- how a regional dispute escalated into a wider conflict

restatement: "The question asks about the causes of WWI, with particular focus on how the alliance system turned a regional dispute into a multi-country war."

goals:
- causes of World War One
- WWI alliance system and how it drew in multiple countries
- escalation from regional dispute to wider war in 1914

---

EXAMPLE 4 - a question wrapped in a deliverable/task instruction:

Question: "Make a poster about the water cycle. It needs to show the main stages, explain why the water cycle matters for life on earth, and include one real example of the water cycle being disrupted (e.g. a drought)."

"Make a poster" is the deliverable - ignore it completely, it is not a clause. The actual content being asked for is the three things listed after it.

clauses:
- the main stages of the water cycle
- why the water cycle matters for life on Earth
- a real example of the water cycle being disrupted, e.g. a drought

restatement: "The question asks for a poster on the water cycle covering its main stages, why it matters for life on Earth, and one real example of it being disrupted."

goals:
- main stages of the water cycle
- why the water cycle matters for life on Earth
- real example of the water cycle being disrupted, e.g. drought

(Note there is no goal for "poster" itself - that would only return shopping listings for posters to buy, not water cycle information.)

---

Now do the same for the student's actual question above.

Respond with ONLY a single JSON object, no other text before or after it, no markdown code fences. It must have exactly these three keys:
{{"clauses": ["...", "..."], "restatement": "...", "goals": ["...", "..."]}}"""

SHRINK_PROMPT = """I am trying to find out this one thing:

{goal}

Below is a web page from {url}.

{text}

First, check whether this page is actually the kind of thing worth reading, regardless of how on-topic it sounds.

Reply with exactly: nothing relevant
if any of these are true:
- the page is spam, advertising, or incoherent
- the website looks like a content farm, a spam site, or a lookalike of a real organisation
- it's a store, marketplace, or product listing page (prices, "buy now", shipping/delivery info, star ratings, "X bought in the past month", size/colour options) - reject this EVEN IF the product itself is literally about the topic, e.g. a page selling "water cycle posters" is still a shopping page, not a source of water cycle information
- the page has nothing useful for that one thing

If it passes that check, ask a second question before pulling anything out: would this specific sentence actually help a student understand or complete this part, or does it just mention the topic in passing? Dumping a sentence because it contains the right keywords is not the bar - it has to carry real, usable information.

Pull out AT MOST 4 points that clear that bar - pick the best, clearest ones rather than including everything. Be careful about HOW you split it into points:
- If several consecutive sentences are just one continuous explanation of a single idea (e.g. a process described step by step: it evaporates, then it rises, then it condenses), that is ONE point, not one point per sentence. Splitting a single explanation across several bullets just because it has several sentences makes it read like a chopped-up paragraph, not real notes.
- Only give something its own separate point if it's a genuinely distinct fact or idea - a different figure, a different cause, a different example - not just the next sentence in the same explanation.
Each point should be one short sentence, rarely two. Rephrase down to the useful core rather than copying a long passage word for word - you're taking a note, not quoting a chunk of the page. Do not comment on anything or add opinions of your own."""


class Plan(BaseModel):
    # Field order matters: the model fills these in order, so this order forces
    # it to work out the clauses (step 1) before writing goals (step 3), rather
    # than jumping straight to a compressed answer.
    clauses: list[str]  # every distinct thing found in the question
    restatement: str
    goals: list[str]


def _log():
    return logging.getLogger(LOGGER_NAME)


def _extract_json(text: str) -> str:
    """Models sometimes wrap JSON in ```json fences or add a stray sentence
    around it despite being told not to. Pull out just the {...} block."""
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("no JSON object found in the plan response")
    return text[start:end + 1]


def make_plan(llm, question: str):
    """
    Returns (plan, ai_messages). ai_messages are kept so the audit can count
    tokens.

    Deliberately NOT using with_structured_output()/forced tool-calling here:
    some providers' "thinking" models reject forced tool_choice outright
    (DeepSeek does - see the 2026-10-01 log). Plain JSON-in-the-prompt, parsed
    by hand, works everywhere regardless of thinking mode or tool-calling
    support.
    """
    prompt = PLAN_PROMPT.format(question=question, max_goals=MAX_GOALS)
    try:
        msg = llm.invoke(prompt)
        messages = [msg]
        data = json.loads(_extract_json(msg.content))
        plan = Plan(**data)
        if not plan.goals:
            raise ValueError("plan had no goals")
    except Exception as e:
        # Fall back to searching the question as one goal, rather than crashing the run.
        _log().warning("Plan step failed (%s) - falling back to a single goal.", e)
        return Plan(clauses=[question], restatement=question, goals=[question]), []

    plan.goals = [g.strip() for g in plan.goals if g.strip()][:MAX_GOALS]
    _log().info("PLAN clauses=%r restatement=%r goals=%r", plan.clauses, plan.restatement, plan.goals)
    return plan, messages


MAX_EXTRACT_CHARS = 700  # hard cap - a prompt asking for "at most 4 short points"
# isn't always obeyed (seen firsthand: near-full pages came back regardless),
# so this backstops it in code rather than trusting wording alone.


def shrink(llm, goal: str, url: str, text: str):
    """Returns (extract, ai_message). extract is '' if the page wasn't useful."""
    msg = llm.invoke(SHRINK_PROMPT.format(goal=goal, url=url, text=text))
    extract = (msg.content or "").strip()
    if not extract or "nothing relevant" in extract.lower()[:40]:
        return "", msg
    if len(extract) > MAX_EXTRACT_CHARS:
        _log().warning(
            "SHRINK extract for goal=%r url=%r was %d chars - truncated to %d (model ignored the length limit)",
            goal, url, len(extract), MAX_EXTRACT_CHARS,
        )
        extract = extract[:MAX_EXTRACT_CHARS].rsplit(" ", 1)[0] + " [...]"
    return extract, msg


def research(llm, question: str, plan: Plan = None, plan_messages: list = None):
    """
    Returns (plan, notes, ai_messages, tool_counts).
    notes: list of {goal, title, url, extract}

    If plan is already made (e.g. main.py called make_plan() itself first, to
    print it before searching starts), pass it in here along with its
    messages - research() won't call make_plan() a second time.
    """
    log = _log()
    if plan is None:
        plan, ai_messages = make_plan(llm, question)
    else:
        ai_messages = list(plan_messages or [])
    notes = []
    tool_counts = {"web_access": 0, "scrape": 0}
    seen_urls = set()

    for goal in plan.goals:
        log.info("CALL web_access(query=%r)", goal)
        results = raw_web_access(goal, max_results=RESULTS_PER_GOAL)
        tool_counts["web_access"] += 1
        log.info("RESULT web_access -> %d result(s)", len(results))

        for r in results:
            url = r["url"]
            if url in seen_urls:
                continue
            seen_urls.add(url)

            log.info("CALL scrape(url=%r)", url)
            page = raw_scrape(url, max_chars=SCRAPE_CHARS)
            tool_counts["scrape"] += 1
            if not page:
                page = r["content"]  # fall back to Tavily's short preview

            extract, msg = shrink(llm, goal, url, page)
            ai_messages.append(msg)
            log.info("SHRINK goal=%r url=%r kept=%d chars", goal, url, len(extract))

            if extract:
                notes.append({"goal": goal, "title": r["title"], "url": url, "extract": extract})

    return plan, notes, ai_messages, tool_counts


def number_sources(notes: list[dict]) -> dict[str, int]:
    """Assigns each unique URL a 1-based number, in the order it first
    appears. This is the single source of truth for numbering - format_notes,
    format_references, and notes_to_sources all use it, so a bullet's [n]
    always matches the same URL in the reference list."""
    numbers = {}
    for n in notes:
        if n["url"] not in numbers:
            numbers[n["url"]] = len(numbers) + 1
    return numbers


def format_notes(plan: Plan, notes: list[dict]) -> str:
    """Builds the text block the writing step receives, grouped by goal.
    Each note is tagged with a [n] number instead of a full URL - the writer
    copies the number as-is, it never has to handle (or risk mangling) an
    actual URL. The real reference list gets built separately, by code, from
    number_sources()."""
    numbers = number_sources(notes)
    blocks = []
    for goal in plan.goals:
        lines = [f"Goal: {goal}"]
        goal_notes = [n for n in notes if n["goal"] == goal]
        if not goal_notes:
            lines.append("- nothing found")
        for n in goal_notes:
            lines.append(f"- [{numbers[n['url']]}] {n['extract']}")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def format_references(notes: list[dict]) -> str:
    """The actual reference list, built entirely from code - not something
    the writer produces, so a URL can never be typo'd, mangled, or invented."""
    numbers = number_sources(notes)
    if not numbers:
        return ""
    titles = {n["url"]: n["title"] for n in notes}
    lines = ["References:"]
    for url, i in sorted(numbers.items(), key=lambda kv: kv[1]):
        lines.append(f"{i}. {titles.get(url, url)} - {url}")
    return "\n".join(lines)


def notes_to_sources(notes: list[dict]) -> list[dict]:
    """Same shape the HTML template already expects: {title, url}."""
    seen, sources = set(), []
    for n in notes:
        if n["url"] not in seen:
            seen.add(n["url"])
            sources.append({"title": n["title"], "url": n["url"]})
    return sources
