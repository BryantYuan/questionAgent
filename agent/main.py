"""
Orchestrator - reads the question from input/questions.txt, gives the
chosen model both tools, and prints the final output.
"""

import os
import re
import ast
import argparse
import datetime
import random
import webbrowser
import logging
from pathlib import Path
from dotenv import load_dotenv
from langgraph.errors import GraphRecursionError
from langchain_core.tools import tool
from langchain.agents import create_agent
from langchain.agents.middleware import ToolCallLimitMiddleware

from .tools.web_access import web_access as raw_web_access
from .tools.scrape import scrape as raw_scrape
from .tools.template import fill_template
from .audit import record_run
from .pipeline import make_plan, research, format_notes, format_references

PROJECT_ROOT = Path.cwd()
load_dotenv(PROJECT_ROOT / ".env")

# Resolved from the current working directory, not this file's own
# location - once installed as a command, main.py's actual file lives
# inside site-packages, nowhere near your input/output folders (and,
# as it turns out, nowhere near your .env either). Run 'questionAgent'
# from inside the questionAgent/ project folder, same rule as running
# 'git status' only working inside a git repo.
INPUT_DIR = PROJECT_ROOT / "input"
OUTPUT_DIR = PROJECT_ROOT / "output"
LOGS_DIR = PROJECT_ROOT / "logs"
SUBMISSIONS_DIR = PROJECT_ROOT / "submissions"
RUBRIC_DIR = PROJECT_ROOT / "rubrics"

LOGGER_NAME = "questionagent"

MISSING_FILE_JOKES = [
    "Bold of you to ask me to find '{filename}' when even YOU don't remember where you put it. It's not in {folder}.",
    "'{filename}' isn't in {folder}. Did you write it down somewhere, or is it living in your head rent-free along with everything else you forgot to save?",
    "Searched {folder} for '{filename}' and came up empty. Even your own file doesn't recognise your handwriting apparently.",
    "No '{filename}' in {folder}. At this point I'm starting to think you make up filenames for fun.",
    "'{filename}' is missing from {folder}. Groundbreaking discovery: you can't run a file that doesn't exist. Try that again.",
]


def missing_file_message(filename: str, folder: Path) -> str:
    joke = random.choice(MISSING_FILE_JOKES)
    return joke.format(filename=filename, folder=folder)


# --- Wrap the plain functions as LangChain tools ---

@tool
def web_access(query: str) -> list[dict]:
    """Search trusted sources for information relevant to a question."""
    logger = logging.getLogger(LOGGER_NAME)
    logger.info("CALL web_access(query=%r)", query)
    results = raw_web_access(query)
    logger.info("RESULT web_access -> %d result(s): %s", len(results), str(results)[:2000])
    return results


@tool
def scrape(url: str, max_chars: int = 5000) -> str:
    """Fetch a specific page and return its clean article text, truncated to max_chars."""
    logger = logging.getLogger(LOGGER_NAME)
    logger.info("CALL scrape(url=%r, max_chars=%d)", url, max_chars)
    text = raw_scrape(url, max_chars=max_chars)
    logger.info("RESULT scrape -> %d character(s): %s", len(text), text[:2000])
    return text


# --- Model selection ---

MODEL_NAMES = {
    "deepseek": "deepseek-v4-flash",
    "gemini": "gemini-2.5-flash",
}


def get_llm(provider: str):
    if provider == "deepseek":
        from langchain_deepseek import ChatDeepSeek
        return ChatDeepSeek(model=MODEL_NAMES["deepseek"], api_key=os.getenv("DEEPSEEK_API_KEY"))
    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(model=MODEL_NAMES["gemini"], api_key=os.getenv("GEMINI_API_KEY"))
    raise ValueError(f"Unknown provider: {provider!r}. Use 'deepseek' or 'gemini'.")


SYSTEM_PROMPT = (
    "You are a research notes finder. You do not answer the student's "
    "question, explain it, or help them write anything - your only job is "
    "to hand back what was found, organised and sourced.\n\n"
    "IMPORTANT: this is a single, one-time response. There is no "
    "follow-up turn - the student cannot reply or ask you to continue "
    "after this, so give the complete set of notes now.\n\n"
    "You will be given the student's question and RESEARCH NOTES already "
    "gathered for it, grouped by what needed to be found out. You do not "
    "search for anything yourself - just organise what's given to you.\n\n"
    "RULES:\n"
    "1. Only use facts that appear in the research notes. Never add "
    "anything from your own knowledge, and never write connecting prose, "
    "an explanation, or a summary around them.\n"
    "2. Put one heading per goal from the research notes, using the "
    "goal's own wording (or the question's own listed parts, if the "
    "question spelled them out itself).\n"
    "3. Under each heading, list the facts as short, separate bullet "
    "points, close to their original wording. Do not merge them into a "
    "paragraph.\n"
    "4. Every note in the research notes already has a [n] number in "
    "front of it. Put that same number next to the matching bullet in your "
    "output, e.g. 'Water covers 70% of Earth's surface. [3]'. Copy the "
    "number exactly as given - never write out a URL yourself, never "
    "invent a number, and never renumber them. A separate reference list "
    "matching each number to its real source is added automatically after "
    "your response, so you don't need to produce one.\n"
    "5. Multiple notes under the same goal often come from different "
    "sources but say the same thing in different words - that's expected, "
    "since the same fact can turn up on several pages. Do not give the "
    "same fact a separate bullet for every source that mentioned it. "
    "Write it once, and put all of the numbers that support it next to "
    "it, e.g. 'Evaporation is the first stage of the water cycle. "
    "[5][6][8]'. A bullet should represent one distinct fact, however "
    "many sources back it up.\n"
    "6. Include as many genuinely distinct facts as the notes support - "
    "do not compress different facts down into one, and do not drop a "
    "real one just to keep the list short. Distinct means a different "
    "fact, not a different source repeating the same one.\n"
    "7. If a goal says 'nothing found', say so plainly under that "
    "heading. Do not guess, and do not add ideas, angles, or opinions of "
    "your own.\n"
    "8. Never answer the question, never give an opinion or verdict, "
    "never write a conclusion. Notes only."
)

BASE_RUBRICS = {
    "FACTUAL": (
        "1. Accuracy - every factual claim made is correct and supported by real evidence.\n"
        "2. Completeness - the main facts genuinely needed to answer the question are all present, not just a partial subset.\n"
        "3. Relevance - every point included actually relates to the question asked, no padding with unrelated information.\n"
        "4. Clarity - the explanation is understandable on its own, without requiring outside context to make sense."
    ),
    "COMPARATIVE": (
        "1. Both sides addressed - each thing being compared is actually discussed, not just one side described in detail while the other is glossed over.\n"
        "2. Direct comparison made - the answer actually compares the two (similarities, differences, trade-offs), not just two separate descriptions placed next to each other.\n"
        "3. Accuracy - the facts stated about each side are correct.\n"
        "4. Fair treatment - neither side is misrepresented or strawmanned to make the comparison easier."
    ),
    "OPINION": (
        "1. Clear position - the answer actually takes and states a position, rather than sitting on the fence throughout.\n"
        "2. Reasoning, not just assertion - each claim is backed by a reason or evidence, not simply stated as fact.\n"
        "3. Addresses a counterpoint - the answer acknowledges at least one reasonable objection to its position, rather than ignoring that the other side exists.\n"
        "4. Internal consistency - the argument doesn't contradict itself, and the conclusion actually follows from what was argued.\n"
        "NOTE: there is no single correct position for this type of question. Mark the quality of the argument for whichever side the student chose - never mark down a student for reaching a different conclusion than your own independent answer."
    ),
    "STRUCTURED": (
        "1. Every required element named in the question is genuinely covered, not skipped or merged into a vague general statement.\n"
        "2. Each element is addressed with real, specific content - not just restating the element's name as if that counts as covering it.\n"
        "3. Accuracy within each element - the facts or reasoning given for each required part are correct.\n"
        "4. Overall coherence - the required parts are presented as a coherent answer, not just an unconnected list."
    ),
}

MARK_SYSTEM_PROMPT = (
    "You are marking a student's finished answer purely against a marking "
    "rubric. Nothing else factors into your assessment - not your own "
    "opinion of the topic, not what you'd personally include, just "
    "whether the student's answer meets the rubric's criteria.\n\n"
    "CRITICAL INSTRUCTION - READ THIS FIRST: your job includes catching "
    "mistakes, not just praising effort. Do not default to agreement. If "
    "a claim in the student's answer is wrong, unsupported, vague, or "
    "off-topic, say so plainly and kindly - do not soften a real problem "
    "into something that sounds like a compliment. Do not assume 'the "
    "student probably meant something reasonable' - judge what they "
    "actually wrote, not what they might have intended. Being liked is "
    "not the goal here; being accurate is.\n\n"
    "You will be given a question and the student's own answer to it. "
    "You may also be given a SUPPLIED RUBRIC. Do this in sequence:\n\n"
    "1. MARKING CRITERIA - if a SUPPLIED RUBRIC was given below, use it "
    "exactly - do not invent your own on top of it or ignore parts of it.\n"
    "If no rubric was supplied, pick the ONE fixed markscheme below that "
    "best fits this question's actual shape, and use its criteria "
    "EXACTLY as written - do not reword, add to, or invent new criteria. "
    "These are fixed on purpose so marking stays consistent question to "
    "question:\n\n"
    f"FACTUAL (use for questions with one correct set of facts):\n{BASE_RUBRICS['FACTUAL']}\n\n"
    f"COMPARATIVE (use when the question asks to compare two or more things):\n{BASE_RUBRICS['COMPARATIVE']}\n\n"
    f"OPINION (use for subjective/evaluative questions with no single correct stance):\n{BASE_RUBRICS['OPINION']}\n\n"
    f"STRUCTURED (use when the question itself lists explicit required elements):\n{BASE_RUBRICS['STRUCTURED']}\n\n"
    "If the question genuinely mixes two shapes (e.g. a structured "
    "question where one required element is itself an opinion "
    "judgement), say so and apply both relevant markschemes together - "
    "but this should be the exception, not the default.\n"
    "State under a MARKING CRITERIA heading which fixed markscheme you "
    "used (or that a supplied rubric was used instead), and why it fits "
    "this question.\n\n"
    "2. Go through the student's answer against ONLY those criteria - "
    "nothing else factors into your marking. Do NOT write a separate "
    "answer of your own, do NOT research the topic broadly, and do NOT "
    "list what a 'complete' answer would look like beyond what's needed "
    "to justify a specific criterion's verdict.\n"
    "For each criterion:\n"
    "- If it concerns factual accuracy, use web_access and scrape to "
    "check the SPECIFIC factual claims the student actually made - look "
    "up whether those exact claims are true, don't research the topic "
    "generally. Apply the same two-layer trust filter as always: does "
    "the URL look like a real, credible source, and separately, does the "
    "preview/content actually read as coherent and on-topic? Discard "
    "anything that fails either check.\n"
    "- If it concerns structure, reasoning, or coverage rather than "
    "facts, judge that directly from the text itself - no research "
    "needed for those.\n"
    "- State plainly whether the student's answer meets, partially "
    "meets, or misses that criterion, quoting or pointing to the "
    "specific part of their answer that justifies your judgement, and "
    "citing what your fact-check found if research was involved. Do not "
    "be vague - 'partially meets this' needs a real reason, not just the "
    "label.\n"
    "HARD LIMIT: at most 4 calls to web_access and 4 calls to scrape, "
    "total, spent on checking the student's actual claims - not on "
    "building a broader picture of the topic. If exceeded, that call is "
    "blocked automatically - don't keep retrying it.\n\n"
    "3. Structure your output under these headings:\n\n"
    "MARKING CRITERIA\n"
    "From step 1.\n\n"
    "CRITERION-BY-CRITERION ASSESSMENT\n"
    "One entry per criterion, in order, as described in step 2.\n\n"
    "OVERALL IMPRESSION\n"
    "A short, honest, descriptive summary of how well the answer met the "
    "criteria overall. Do not give a numeric score or grade unless one "
    "is explicitly requested elsewhere.\n\n"
    "Do NOT rewrite the student's answer for them or tell them the exact "
    "wording to change to. Describe the gap or the issue; let them decide "
    "what to do about it. This is still a single, one-time response - "
    "there is no follow-up turn, so give the complete assessment now."
)


def build_mark_agent(provider: str):
    llm = get_llm(provider)
    mark_tool_limits = [
        ToolCallLimitMiddleware(tool_name="web_access", run_limit=4, exit_behavior="continue"),
        ToolCallLimitMiddleware(tool_name="scrape", run_limit=4, exit_behavior="continue"),
    ]
    return create_agent(
        model=llm,
        tools=[web_access, scrape],
        system_prompt=MARK_SYSTEM_PROMPT,
        middleware=mark_tool_limits,
    )


def read_question(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


def slugify(text: str, max_words: int = 6, max_len: int = 40) -> str:
    """Turns a typed-in question into a safe-ish output filename stem,
    e.g. 'What are vpns for?' -> 'what_are_vpns_for'."""
    words = re.findall(r"[a-zA-Z0-9]+", text.lower())[:max_words]
    return ("_".join(words) or "question")[:max_len]


def resolve_question(arg: str) -> tuple[str, str] | tuple[None, None]:
    """
    The first command-line argument can be either a filename inside input/
    (the original way) or the question typed straight into the command
    (e.g. `uv run questionAgent "What are vpns for"`).

    Returns (question_text, stem_for_output_filename), or (None, None) if
    it looks like a filename that just doesn't exist (so the caller can show
    the normal missing-file message instead of treating a typo as a question).
    """
    candidate = INPUT_DIR / arg
    if candidate.exists():
        return read_question(candidate), candidate.stem

    looks_like_a_filename = " " not in arg and arg.lower().endswith(".txt")
    if looks_like_a_filename:
        return None, None

    return arg.strip(), slugify(arg)


def extract_sources(messages) -> list[dict]:
    sources = []
    seen_urls = set()
    for msg in messages:
        if getattr(msg, "name", None) != "web_access":
            continue
        try:
            results = ast.literal_eval(msg.content)
        except (ValueError, SyntaxError):
            continue
        for r in results:
            url = r.get("url")
            if url and url not in seen_urls:
                sources.append(r)
                seen_urls.add(url)
    return sources


def setup_run_logger(logs_dir: Path) -> tuple[logging.Logger, Path]:
    """Configure this run's logger, writing to its own timestamped file
    in logs/ via Python's standard logging module. Tool calls log
    themselves in real time as they happen, rather than the log being
    reconstructed afterward from the message history."""
    logs_dir.mkdir(exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_path = logs_dir / f"{timestamp}.log"

    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()  # avoid duplicate handlers if main() ever runs twice in one process

    handler = logging.FileHandler(log_path, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(handler)

    return logger, log_path


def main():
    parser = argparse.ArgumentParser(prog="questionAgent")
    parser.add_argument(
        "question",
        help="Either a question file inside the input folder (e.g. example.txt), "
             "or the question itself typed directly, in quotes, e.g. \"What are vpns for\"",
    )
    parser.add_argument(
        "answer_filename",
        nargs="?",
        default=None,
        help="Name of your answer file inside the submissions folder. Required when using --mark.",
    )
    parser.add_argument("--provider", choices=["deepseek", "gemini"], default="deepseek")
    parser.add_argument(
        "--mark",
        action="store_true",
        help="Compare your own submitted answer against an independently researched one.",
    )
    parser.add_argument(
        "--rubric",
        default=None,
        help="Optional: name of a rubric file inside the rubrics folder. Only used with --mark. If omitted, the model picks the best-fitting fixed markscheme itself.",
    )
    args = parser.parse_args()

    question, stem = resolve_question(args.question)
    if question is None:
        print(missing_file_message(args.question, INPUT_DIR))
        return

    if args.mark:
        if not args.answer_filename:
            print(
                "Marking mode needs your answer file too, e.g.:\n"
                f"  questionAgent --mark {args.question!r} myresult.txt\n"
                f"(myresult.txt should be inside {SUBMISSIONS_DIR})"
            )
            return

        SUBMISSIONS_DIR.mkdir(exist_ok=True)
        answer_path = SUBMISSIONS_DIR / args.answer_filename
        if not answer_path.exists():
            print(missing_file_message(args.answer_filename, SUBMISSIONS_DIR))
            return

        student_answer = read_question(answer_path)
        print(f"Marking mode - Question: {question}\n")

        rubric_text = None
        if args.rubric:
            rubric_path = RUBRIC_DIR / args.rubric
            if not rubric_path.exists():
                print(missing_file_message(args.rubric, RUBRIC_DIR))
                return
            rubric_text = read_question(rubric_path)
            print(f"Using supplied rubric: {args.rubric}\n")

        agent = build_mark_agent(args.provider)
        user_message = f"QUESTION:\n{question}\n\n"
        if rubric_text:
            user_message += f"SUPPLIED RUBRIC:\n{rubric_text}\n\n"
        user_message += f"STUDENT'S ANSWER TO COMPARE AGAINST:\n{student_answer}"
    else:
        print(f"Question: {question}\n")

        agent = None
        user_message = question

    logger, log_path = setup_run_logger(LOGS_DIR)
    logger.info("Provider: %s", args.provider)
    logger.info("Mode: %s", "mark" if args.mark else "research")
    logger.info("Question: %s", question)

    tool_counts = None  # None = audit counts them from the messages (mark mode)

    if args.mark:
        try:
            result = agent.invoke(
                {"messages": [{"role": "user", "content": user_message}]},
                config={"recursion_limit": 20},
            )
            final_answer = result["messages"][-1].content
            messages_for_log = result["messages"]

        except GraphRecursionError:
            logger.warning("Hit the tool-call/recursion limit - falling back to a best-effort answer without tools.")
            print("Hit the tool-call limit before finishing research - falling back to a best-effort answer.\n")

            llm = get_llm(args.provider)
            fallback = llm.invoke([
                {
                    "role": "system",
                    "content": (
                        "You ran out of research budget searching for trustworthy "
                        "sources on this question and could not finish properly. "
                        "Answer as well as you honestly can from your own general "
                        "knowledge, but start your response with a clear, obvious "
                        "warning that this answer was NOT verified against "
                        "trusted sources due to running out of search attempts, "
                        "and that the student should treat it with extra caution "
                        "and ideally verify it themselves before relying on it."
                    ),
                },
                {"role": "user", "content": user_message},
            ])
            final_answer = fallback.content
            messages_for_log = [{"role": "user", "content": user_message}, fallback]

        logger.info("FINAL ANSWER:\n%s", final_answer)
        sources = extract_sources(messages_for_log)

    else:
        llm = get_llm(args.provider)

        print("Planning...\n")
        plan, plan_messages = make_plan(llm, question)
        print(f"My understanding: {plan.restatement}\n")
        print("Going to search for:")
        for i, goal in enumerate(plan.goals, start=1):
            print(f"  {i}. {goal}")
        print()

        print("Researching...\n")
        plan, notes, research_messages, tool_counts = research(llm, question, plan=plan, plan_messages=plan_messages)

        writer_input = (
            f"The student's question:\n\n{question}\n\n"
            f"My understanding of the question:\n\n{plan.restatement}\n\n"
            "RESEARCH NOTES, grouped by what needed to be found out. Each "
            "note shows where it came from:\n\n"
            f"{format_notes(plan, notes)}"
        )
        logger.info("WRITER INPUT:\n%s", writer_input)

        reply = llm.invoke([
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": writer_input},
        ])

        # The reference list is built entirely from the notes themselves, not
        # generated by the writer - a URL can't be mangled, renumbered, or
        # invented if the AI never has to type one out.
        references = format_references(notes)
        final_answer = (
            f"**My understanding of your question:** {plan.restatement}\n\n---\n\n"
            f"{reply.content}"
            + (f"\n\n---\n\n{references}" if references else "")
        )
        messages_for_log = research_messages + [reply]
        logger.info("FINAL ANSWER:\n%s", final_answer)
        # References are already embedded in final_answer (via format_references,
        # numbered to match the [n] tags in the text) - don't also render the
        # template's separate Sources section, or every source shows up twice.
        sources = []

    OUTPUT_DIR.mkdir(exist_ok=True)
    suffix = "_marked.html" if args.mark else "_output.html"
    output_filename = f"{stem}{suffix}"
    output_path = fill_template(
        question=question,
        answer=final_answer,
        sources=sources,
        output_path=str(OUTPUT_DIR / output_filename),
    )
    print(f"Saved to {output_path}")
    webbrowser.open(f"file://{Path(output_path).resolve()}")

    print(f"Log saved to {log_path}")

    LOGS_DIR.mkdir(exist_ok=True)
    all_time = record_run(LOGS_DIR, MODEL_NAMES[args.provider], messages_for_log, tool_counts)
    print(
        f"All-time usage: {all_time['total_tokens']} tokens across "
        f"{all_time['runs']} runs (~${all_time['cost_usd']:.4f} estimated)"
    )


if __name__ == "__main__":
    main()
