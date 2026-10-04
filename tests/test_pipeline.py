"""
Unit tests for pipeline.py. These use a fake LLM and fake web_access/scrape,
so they cost nothing and run instantly - they check the CODE around the
prompts (looping, filtering, trimming, formatting), not whether the prompt
wording itself is good. Wording quality needs a real run against a test
question, read by eye - see README's "Adjusting the prompts" note.

Run with: python -m unittest tests/test_pipeline.py
"""

import unittest
from unittest.mock import patch

from agent import pipeline as p


class FakeMessage:
    def __init__(self, content):
        self.content = content
        self.usage_metadata = {"input_tokens": 10, "output_tokens": 5}


class FakeLLM:
    """
    A stand-in AI with one plain .invoke(prompt) method, same as the real
    model - no with_structured_output(), since make_plan() no longer uses it
    (DeepSeek's thinking mode rejects forced tool_choice - see the
    2026-10-01 bug). Plan calls and shrink calls both come through here, so
    this tells them apart by a phrase unique to SHRINK_PROMPT.

    plan_json: the raw text the AI "replies" with for a plan call - pass
    real JSON to test a working plan, or garbage text to test the fallback
    path. plan=... is a shortcut that builds the JSON from a Plan for you.
    shrink_response: called for every shrink() call - override per test.
    """
    def __init__(self, plan=None, plan_json=None, shrink_response=lambda prompt: "a relevant point"):
        if plan is not None:
            plan_json = plan.model_dump_json()
        self._plan_json = plan_json
        self._shrink_response = shrink_response

    def invoke(self, prompt):
        if "I am trying to find out this one thing" in prompt:  # shrink call
            return FakeMessage(self._shrink_response(prompt))
        # plan call
        if self._plan_json is None:
            return FakeMessage("sorry, I can't help with that today")  # not JSON -> triggers fallback
        return FakeMessage(self._plan_json)


class TestMakePlan(unittest.TestCase):
    def test_returns_the_plan_and_trims_to_max_goals(self):
        """More than MAX_GOALS goals from the AI should be cut down, not passed through."""
        too_many = p.Plan(clauses=[f"clause{i}" for i in range(10)], restatement="r", goals=[f"goal{i}" for i in range(10)])
        llm = FakeLLM(plan=too_many)

        plan, messages = p.make_plan(llm, "some question")

        self.assertEqual(len(plan.goals), p.MAX_GOALS)
        self.assertEqual(plan.restatement, "r")

    def test_drops_blank_goals(self):
        messy = p.Plan(clauses=["c1", "c2", "c3"], restatement="r", goals=["real goal", "  ", ""])
        llm = FakeLLM(plan=messy)

        plan, _ = p.make_plan(llm, "q")

        self.assertEqual(plan.goals, ["real goal"])

    def test_clauses_are_carried_through(self):
        """clauses is the step-1 chain-of-thought field - make sure it actually
        survives make_plan() rather than being dropped anywhere."""
        plan_with_clauses = p.Plan(
            clauses=["what VPNs were designed for", "good uses", "bad uses"],
            restatement="r",
            goals=["original purpose of VPNs", "good uses of VPNs", "bad uses of VPNs"],
        )
        llm = FakeLLM(plan=plan_with_clauses)

        plan, _ = p.make_plan(llm, "some question")

        self.assertEqual(plan.clauses, ["what VPNs were designed for", "good uses", "bad uses"])
        self.assertEqual(len(plan.clauses), len(plan.goals))

    def test_falls_back_to_single_goal_when_the_ai_does_not_return_json(self):
        """If the AI replies with plain text instead of JSON (wrong format,
        refusal, provider error, etc.), don't crash - treat the whole
        question as one goal instead."""
        llm = FakeLLM(plan_json=None)

        plan, messages = p.make_plan(llm, "what are vpns for")

        self.assertEqual(plan.goals, ["what are vpns for"])
        self.assertEqual(plan.restatement, "what are vpns for")
        self.assertEqual(messages, [])

    def test_falls_back_when_json_is_missing_a_required_field(self):
        """Valid JSON but missing e.g. 'clauses' should fall back too, not crash."""
        llm = FakeLLM(plan_json='{"restatement": "r", "goals": ["g1"]}')  # no clauses

        plan, messages = p.make_plan(llm, "what are vpns for")

        self.assertEqual(plan.goals, ["what are vpns for"])  # fell back

    def test_handles_json_wrapped_in_markdown_fences(self):
        """Some models wrap JSON in ```json fences despite being told not to -
        this should still parse rather than falling back unnecessarily."""
        fenced = '```json\n{"clauses": ["c1"], "restatement": "r", "goals": ["g1"]}\n```'
        llm = FakeLLM(plan_json=fenced)

        plan, messages = p.make_plan(llm, "question")

        self.assertEqual(plan.goals, ["g1"])
        self.assertEqual(plan.clauses, ["c1"])


class TestShrink(unittest.TestCase):
    def test_keeps_a_useful_extract(self):
        llm = FakeLLM(shrink_response=lambda prompt: "- useful point one\n- useful point two")

        extract, msg = p.shrink(llm, "goal", "https://good.example", "page text")

        self.assertIn("useful point one", extract)

    def test_drops_nothing_relevant_pages(self):
        llm = FakeLLM(shrink_response=lambda prompt: "nothing relevant")

        extract, msg = p.shrink(llm, "goal", "https://spam.example", "buy now!!!")

        self.assertEqual(extract, "")

    def test_drops_empty_replies_too(self):
        llm = FakeLLM(shrink_response=lambda prompt: "   ")

        extract, msg = p.shrink(llm, "goal", "https://x.example", "text")

        self.assertEqual(extract, "")


class TestResearchWithPreMadePlan(unittest.TestCase):
    def test_does_not_call_make_plan_again_when_a_plan_is_passed_in(self):
        """main.py now calls make_plan() itself first (to print the plan to
        the terminal before searching starts), then passes it into research().
        research() must use that plan, not silently make a second one."""
        plan = p.Plan(clauses=["c1"], restatement="r", goals=["goal one"])
        llm = FakeLLM(plan=plan)  # if research() ignored the passed-in plan and
        # called make_plan() again, it would ask this same llm for another
        # plan - fine either way here since it's the same plan, so instead:
        results = [{"title": "A", "url": "https://a.example", "content": "preview"}]

        with patch.object(p, "raw_web_access", return_value=results) as mock_search, \
             patch.object(p, "raw_scrape", return_value="page text"):

            returned_plan, notes, messages, tool_counts = p.research(
                llm, "question", plan=plan, plan_messages=["pretend-plan-message"]
            )

            self.assertIs(returned_plan, plan)
            # the passed-in plan message should be carried through, not dropped
            self.assertIn("pretend-plan-message", messages)
            # only one search - for goal one - proves it didn't regenerate the plan
            self.assertEqual(mock_search.call_count, 1)


class TestResearch(unittest.TestCase):
    def setUp(self):
        self.plan = p.Plan(clauses=["clause one", "clause two"], restatement="r", goals=["goal one", "goal two"])
        self.llm = FakeLLM(plan=self.plan)

    def test_runs_a_search_per_goal_and_a_scrape_per_result(self):
        # Different URLs per goal, so the same-URL dedup (tested separately
        # below) doesn't swallow any of these.
        def results_for(goal, max_results=2):
            return [
                {"title": "A", "url": f"https://{goal}-a.example", "content": "preview a"},
                {"title": "B", "url": f"https://{goal}-b.example", "content": "preview b"},
            ]

        with patch.object(p, "raw_web_access", side_effect=results_for) as mock_search, \
             patch.object(p, "raw_scrape", return_value="full page text") as mock_scrape:

            plan, notes, messages, tool_counts = p.research(self.llm, "question")

            # one search call per goal
            self.assertEqual(mock_search.call_count, len(plan.goals))
            # one scrape call per (goal, result) pair, since every URL here is unique
            self.assertEqual(mock_scrape.call_count, len(plan.goals) * 2)
            self.assertEqual(tool_counts["web_access"], len(plan.goals))

    def test_nothing_relevant_pages_do_not_become_notes(self):
        fake_results = [{"title": "Spam", "url": "https://spam.example", "content": "buy now"}]
        llm = FakeLLM(plan=self.plan, shrink_response=lambda prompt: "nothing relevant")

        with patch.object(p, "raw_web_access", return_value=fake_results), \
             patch.object(p, "raw_scrape", return_value="junk page"):

            plan, notes, messages, tool_counts = p.research(llm, "question")

            self.assertEqual(notes, [])

    def test_same_url_is_not_scraped_twice_across_goals(self):
        """If the same page turns up for two different goals, only scrape it once."""
        same_result = [{"title": "A", "url": "https://a.example", "content": "preview"}]
        with patch.object(p, "raw_web_access", return_value=same_result), \
             patch.object(p, "raw_scrape", return_value="page text") as mock_scrape:

            p.research(self.llm, "question")

            self.assertEqual(mock_scrape.call_count, 1)

    def test_falls_back_to_search_preview_when_scrape_fails(self):
        """scrape() returning empty (e.g. page failed to load) shouldn't lose the note -
        fall back to the short preview text web_access already returned."""
        fake_results = [{"title": "A", "url": "https://a.example", "content": "the preview text"}]
        seen = {}

        def capture_shrink(prompt):
            seen["prompt"] = prompt
            return "kept"

        llm = FakeLLM(plan=self.plan, shrink_response=capture_shrink)

        with patch.object(p, "raw_web_access", return_value=fake_results), \
             patch.object(p, "raw_scrape", return_value=""):

            p.research(llm, "question")

            self.assertIn("the preview text", seen["prompt"])


class TestFormatNotes(unittest.TestCase):
    def test_groups_notes_under_their_goal_in_goal_order(self):
        plan = p.Plan(clauses=["clause a", "clause b"], restatement="r", goals=["goal a", "goal b"])
        notes = [
            {"goal": "goal b", "title": "T", "url": "https://b.example", "extract": "note b"},
            {"goal": "goal a", "title": "T", "url": "https://a.example", "extract": "note a"},
        ]

        text = p.format_notes(plan, notes)

        self.assertLess(text.index("goal a"), text.index("goal b"))
        self.assertIn("note a", text)
        self.assertIn("note b", text)

    def test_empty_goal_says_nothing_found(self):
        plan = p.Plan(clauses=["clause with no results"], restatement="r", goals=["goal with no results"])

        text = p.format_notes(plan, [])

        self.assertIn("nothing found", text)


    def test_notes_are_tagged_with_a_reference_number_not_a_raw_url(self):
        """The writer should never see a raw URL to copy or mangle - only a
        [n] tag, matching format_references()."""
        plan = p.Plan(clauses=["c"], restatement="r", goals=["goal a"])
        notes = [{"goal": "goal a", "title": "T", "url": "https://a.example", "extract": "some fact"}]

        text = p.format_notes(plan, notes)

        self.assertIn("[1]", text)
        self.assertNotIn("https://a.example", text)

    def test_same_url_gets_the_same_number_across_goals(self):
        plan = p.Plan(clauses=["c1", "c2"], restatement="r", goals=["goal a", "goal b"])
        notes = [
            {"goal": "goal a", "title": "T", "url": "https://shared.example", "extract": "fact one"},
            {"goal": "goal b", "title": "T", "url": "https://shared.example", "extract": "fact two"},
        ]

        text = p.format_notes(plan, notes)

        # both facts came from the same URL, so both must carry the same number
        self.assertEqual(text.count("[1]"), 2)


class TestNumberSources(unittest.TestCase):
    def test_numbers_start_at_one_in_first_seen_order(self):
        notes = [
            {"goal": "g1", "title": "A", "url": "https://a.example", "extract": "x"},
            {"goal": "g1", "title": "B", "url": "https://b.example", "extract": "y"},
        ]

        numbers = p.number_sources(notes)

        self.assertEqual(numbers, {"https://a.example": 1, "https://b.example": 2})

    def test_repeated_url_keeps_its_first_number(self):
        notes = [
            {"goal": "g1", "title": "A", "url": "https://a.example", "extract": "x"},
            {"goal": "g2", "title": "A", "url": "https://a.example", "extract": "y"},
        ]

        numbers = p.number_sources(notes)

        self.assertEqual(numbers, {"https://a.example": 1})


class TestFormatReferences(unittest.TestCase):
    def test_lists_each_source_once_with_its_number(self):
        notes = [
            {"goal": "g1", "title": "A Title", "url": "https://a.example", "extract": "x"},
            {"goal": "g2", "title": "A Title", "url": "https://a.example", "extract": "y"},  # dupe, same source
            {"goal": "g2", "title": "B Title", "url": "https://b.example", "extract": "z"},
        ]

        text = p.format_references(notes)

        self.assertEqual(text.count("https://a.example"), 1)  # not once per note that used it
        self.assertIn("1. A Title - https://a.example", text)
        self.assertIn("2. B Title - https://b.example", text)

    def test_empty_notes_gives_empty_string(self):
        self.assertEqual(p.format_references([]), "")


class TestNotesToSources(unittest.TestCase):
    def test_deduplicates_by_url(self):
        notes = [
            {"goal": "g1", "title": "A", "url": "https://a.example", "extract": "x"},
            {"goal": "g2", "title": "A", "url": "https://a.example", "extract": "y"},
        ]

        sources = p.notes_to_sources(notes)

        self.assertEqual(len(sources), 1)


if __name__ == "__main__":
    unittest.main()
