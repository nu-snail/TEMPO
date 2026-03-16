"""TimeSPEC inference-time agent: 5-phase leakage blocking pipeline.

Reimplements the KDD paper's TemporalLLMAgent architecture:
  Phase 1 -- Generator:    search + generate draft prediction
  Phase 2 -- Supervisor 1: extract claims, detect leakage
  Phase 3 -- Regenerator:  (conditional) regenerate with only valid evidence
  Phase 4 -- Supervisor 2: (conditional) re-validate regenerated output
  Phase 5 -- Aggregator:   synthesise final prediction from valid claims only

The final (prediction, rationale) is returned for downstream evaluation via
the standard 4-phase evaluation pipeline (claim extraction -> Shapley ->
leakage detection -> metrics).
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Any

from src.shared.claim_extractor import ClaimExtractor
from src.shared.data_loader import TaskInstance
from src.shared.leakage_detector import LeakageDetector
from src.shared.llm_client import LLMClient
from src.shared.search_client import PersistentSearchCache, SearchClient

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Task-specific prediction format (aligned with GRPO training output format)
# ---------------------------------------------------------------------------

_PREDICTION_DESC: dict[str, str] = {
    "stock": (
        "A JSON array of ticker symbols ranked from BEST (highest expected "
        "return) to WORST (lowest expected return). "
        'Example: ["DVN", "MPC", "CVX", "SLB", "KMI"]'
    ),
    "salary": (
        "Average Annual Value (AAV) in USD as an integer. "
        "Example: 15000000"
    ),
    "legal": (
        "Probability that the PETITIONER wins, as a float from 0.0 to 1.0. "
        "Example: 0.65"
    ),
}

# ---------------------------------------------------------------------------
# Tool factories (task-specific output constraints, cutoff-aware search)
# ---------------------------------------------------------------------------


def _make_search_tool(cutoff_date: str) -> dict:
    return {
        "type": "function",
        "function": {
            "name": "search_information",
            "description": (
                f"Search for information available BEFORE {cutoff_date}. "
                "Results are automatically filtered to pre-cutoff sources only."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": (
                            "Search query targeting pre-cutoff information. "
                            "Include relevant names, tickers, and dates."
                        ),
                    },
                    "purpose": {
                        "type": "string",
                        "description": "Why this search is needed for the prediction.",
                    },
                },
                "required": ["query", "purpose"],
            },
        },
    }


def _make_draft_tool(task: str) -> dict:
    return {
        "type": "function",
        "function": {
            "name": "submit_draft_prediction",
            "description": "Submit draft prediction after gathering sufficient evidence.",
            "parameters": {
                "type": "object",
                "properties": {
                    "prediction": {
                        "type": "string",
                        "description": _PREDICTION_DESC.get(
                            task, "The prediction value."
                        ),
                    },
                    "rationale": {
                        "type": "string",
                        "description": (
                            "Detailed analysis grounded in specific, datable, "
                            "pre-cutoff facts. Minimum 400 characters. "
                            "Each major claim must cite its source and date."
                        ),
                    },
                },
                "required": ["prediction", "rationale"],
            },
        },
    }


def _make_final_tool(task: str) -> dict:
    return {
        "type": "function",
        "function": {
            "name": "submit_final_prediction",
            "description": (
                "Submit final synthesised prediction based on validated evidence only."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "prediction": {
                        "type": "string",
                        "description": _PREDICTION_DESC.get(
                            task, "The final prediction value."
                        ),
                    },
                    "rationale": {
                        "type": "string",
                        "description": (
                            "Final analysis citing ONLY validated, pre-cutoff "
                            "evidence. Minimum 400 characters."
                        ),
                    },
                },
                "required": ["prediction", "rationale"],
            },
        },
    }


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

_TEMPORAL_BLOCK = (
    "CRITICAL TEMPORAL CONSTRAINT:\n"
    "You may ONLY use information available BEFORE {cutoff_date}. "
    "Do NOT use any information from on or after this date.\n"
)

_TASK_INSTRUCTION: dict[str, str] = {
    "stock": (
        "Rank the given companies by their expected stock return over the "
        "specified evaluation period, from BEST (highest expected return) to "
        "WORST (lowest expected return), as a JSON array of ticker symbols."
    ),
    "salary": (
        "Predict the given athlete's Average Annual Value (AAV) in USD for "
        "their next contract. Output an integer (e.g., 15000000)."
    ),
    "legal": (
        "Predict the probability that the PETITIONER wins the given U.S. "
        "Supreme Court case. Output a float from 0.0 to 1.0 (e.g., 0.65)."
    ),
}

_CLAIM_TAXONOMY = (
    "Each validated claim has a category:\n"
    "- A1 (Discrete Event): a specific event at a known point in time.\n"
    "- A2 (State/Measurement): a measurable quantity or state from a specific time.\n"
    "- A3 (Publication): information from a dated document or announcement.\n"
    "- B1 (Background): timeless or well-established facts.\n"
    "- B2 (Definitional): definitions, tautologies, or logical derivations.\n\n"
    "Claims categorized as A4 (Outcome) or A5 (Consequential) have been "
    "removed as temporal violations. Only the above categories remain."
)


def _generator_system(task: str, cutoff_date: str) -> str:
    temporal = _TEMPORAL_BLOCK.format(cutoff_date=cutoff_date)
    instruction = _TASK_INSTRUCTION[task]
    return (
        f"# ROLE\nPrediction analyst gathering evidence for forecasting.\n\n"
        f"# PREDICTION TASK\n{instruction}\n\n"
        f"{temporal}\n"
        "# WORKFLOW\n"
        "1. Search for relevant, verifiable, pre-cutoff evidence.\n"
        "2. Gather specific facts with dates and sources.\n"
        "3. Submit your draft prediction with a detailed, fact-grounded rationale.\n\n"
        "# RATIONALE REQUIREMENTS\n"
        "- Every major claim must cite a specific fact with a date.\n"
        "- Avoid vague statements without supporting data.\n"
        "- Minimum 400 characters.\n\n"
        "# OUTPUT\n"
        "After gathering sufficient evidence, call `submit_draft_prediction`."
    )


def _generator_user(instance: TaskInstance) -> str:
    case_input = _format_case_input(instance)
    return (
        f"Make a prediction for the following case:\n\n"
        f"{case_input}\n\n"
        f"Knowledge cutoff: {instance.cutoff_date}\n\n"
        "Search for relevant evidence, then submit your draft prediction "
        "with a detailed rationale grounded in specific pre-cutoff facts."
    )


def _regenerator_system(task: str, cutoff_date: str) -> str:
    temporal = _TEMPORAL_BLOCK.format(cutoff_date=cutoff_date)
    return (
        "# ROLE\n"
        "Analyst regenerating prediction after temporal violations detected.\n\n"
        f"{temporal}\n"
        "# CONTEXT\n"
        "Your initial prediction contained claims that could not be verified "
        "as pre-cutoff. Those claims have been removed.\n\n"
        "# OBJECTIVES\n"
        "1. REUSE all valid claims and search results provided below.\n"
        "2. Search for NEW, DIVERSE evidence to fill gaps left by removed claims. "
        "Do NOT repeat the previous queries listed below.\n"
        "3. Submit an improved prediction grounded in combined evidence.\n\n"
        "# OUTPUT\n"
        "Call `submit_draft_prediction` with your revised prediction and rationale."
    )


def _regenerator_user(
    instance: TaskInstance,
    valid_claims: list[dict],
    valid_searches: list[dict],
    previous_queries: list[str],
) -> str:
    case_input = _format_case_input(instance)
    claims_text = _format_valid_claims(valid_claims)
    searches_text = _format_valid_searches(valid_searches)
    queries_text = "\n".join(f"- {q}" for q in previous_queries) or "(none)"
    return (
        f"# REGENERATION TASK\n\n"
        f"Case input:\n{case_input}\n\n"
        f"Knowledge cutoff: {instance.cutoff_date}\n\n"
        f"## LAYER 1: VALID CLAIMS (reuse these)\n{claims_text}\n\n"
        f"## LAYER 2: VALID SEARCH RESULTS (reuse these)\n{searches_text}\n\n"
        f"## PREVIOUS QUERIES (do NOT repeat)\n{queries_text}\n\n"
        "Review the valid evidence above. Search new angles to strengthen your "
        "prediction, then submit your revised prediction and rationale."
    )


def _aggregator_system(task: str, cutoff_date: str) -> str:
    temporal = _TEMPORAL_BLOCK.format(cutoff_date=cutoff_date)
    instruction = _TASK_INSTRUCTION[task]
    return (
        "# ROLE\nAnalyst performing final synthesis of validated evidence.\n\n"
        f"# PREDICTION TASK\n{instruction}\n\n"
        f"{temporal}\n"
        "# CRITICAL INSTRUCTION\n"
        "Your rationale must contain ONLY facts from the case input and "
        "the validated claims below. Do NOT introduce any new information.\n\n"
        f"# CLAIM CATEGORIES\n{_CLAIM_TAXONOMY}\n\n"
        "# SYNTHESIS STRATEGY\n"
        "1. Prioritize A1/A2/A3 claims as primary evidence.\n"
        "2. Use B1/B2 claims for contextual grounding.\n"
        "3. Weigh evidence by relevance and recency (within pre-cutoff bounds).\n"
        "4. If claims conflict, note the conflict and explain your resolution.\n\n"
        "# OUTPUT\n"
        "Call `submit_final_prediction` with your synthesised prediction."
    )


def _aggregator_user(instance: TaskInstance, valid_claims: list[dict]) -> str:
    case_input = _format_case_input(instance)
    claims_text = _format_valid_claims(valid_claims)
    return (
        f"# FINAL SYNTHESIS TASK\n\n"
        f"Case input:\n{case_input}\n\n"
        f"Knowledge cutoff: {instance.cutoff_date}\n\n"
        f"## VALIDATED CLAIMS\n{claims_text}\n\n"
        "Synthesise a final prediction using ONLY the case input and validated "
        "claims above. Cite specific facts for each element of your prediction."
    )


# ---------------------------------------------------------------------------
# Case input formatting (aligned with GRPO training user prompts)
# ---------------------------------------------------------------------------


def _format_case_input(instance: TaskInstance) -> str:
    """Format case input to match GRPO training user prompts for fair comparison.

    Each task's format mirrors the corresponding build_user_prompt() in
    src/shared/prompts/{stock,salary,legal}.py.
    """
    raw = instance.raw

    if instance.task == "stock":
        companies = raw.get("companies", [])
        names = raw.get("company_names", {})
        sector = raw.get("sector", "Unknown")
        period = raw.get("past_period", {})
        start = period.get("start_date", period.get("start", ""))
        end = period.get("end_date", period.get("end", ""))
        lines = [
            f"Sector: {sector}",
            f"Evaluation period: {start} to {end}",
            "Companies to rank (best to worst expected return):",
        ]
        for ticker in companies:
            name = names.get(ticker, ticker)
            lines.append(f"  - {ticker} ({name})")
        return "\n".join(lines)

    if instance.task == "salary":
        player = raw.get("Player", "Unknown")
        sport = raw.get("Sport", "NBA")
        year = raw.get("Year", raw.get("Season", ""))
        status = raw.get("Status", "")
        lines = [
            f"Sport: {sport}",
            f"Free-agent year: {year}",
            f"Player: {player}",
        ]
        if status:
            lines.append(f"Free-agent type: {status}")
        return "\n".join(lines)

    if instance.task == "legal":
        case_name = raw.get("case_name", "Unknown")
        petitioner = raw.get("petitioner", "")
        respondent = raw.get("respondent", "")
        term = raw.get("term", "")
        lines = [
            f"Case: {case_name}",
            f"Term: {term}",
            f"Petitioner: {petitioner}",
            f"Respondent: {respondent}",
        ]
        return "\n".join(lines)

    # Fallback for unknown task types: filtered JSON dump
    return json.dumps(
        _filter_ground_truth(instance), indent=2, ensure_ascii=False,
    )


# ---------------------------------------------------------------------------
# Ground truth filtering (safety net for fallback path)
# ---------------------------------------------------------------------------

_STOCK_PAST_PERIOD_GT_KEYS = frozenset({"returns", "ranking", "end_prices"})

_EXCLUDED_KEYS = frozenset({
    # Stock metadata
    "question_id", "window_id",
    # Stock ground truth (safety net for flat data variants)
    "true_ranking", "returns", "ground_truth",
    # Salary ground truth
    "AAV", "aav", "To_Team", "to_team", "Total_Value", "total_value",
    "Years", "years", "actual_salary", "Salary",
    # Salary exclusions (prevent indirect leakage)
    "From_Team", "from_team", "Position", "position",
    # Legal ground truth
    "outcome", "conclusion", "winning_party", "decision_direction",
    "vote_split", "decision_date", "decision_type",
    # Legal exclusions
    "facts", "oral_argument_date",
})


def _filter_ground_truth(instance: TaskInstance) -> dict[str, Any]:
    """Remove ground truth fields from raw data.

    For stock tasks, performs nested filtering inside past_period to keep
    prediction period metadata while removing actual returns and rankings.
    Only used as fallback when _format_case_input does not cover the task.
    """
    result = {}
    for k, v in instance.raw.items():
        if k in _EXCLUDED_KEYS:
            continue
        if k == "past_period" and instance.task == "stock" and isinstance(v, dict):
            result[k] = {
                pk: pv for pk, pv in v.items()
                if pk not in _STOCK_PAST_PERIOD_GT_KEYS
            }
        else:
            result[k] = v
    return result


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------


def _format_valid_claims(claims: list[dict]) -> str:
    if not claims:
        return "(No valid claims available)"
    lines = []
    for c in claims[:15]:
        cat = c.get("category", "UNK")
        text = c.get("claim", c.get("text", ""))
        lines.append(f"[{cat}] {text}")
    return "\n".join(lines)


def _format_valid_searches(searches: list[dict]) -> str:
    if not searches:
        return "(No search results available)"
    lines = []
    for s in searches[:10]:
        query = s.get("query", "")
        num = s.get("num_results", 0)
        citations = s.get("citations", [])
        cite_str = ""
        if citations:
            # Citations may be plain URL strings or dicts with title/url
            urls = []
            for c in citations[:3]:
                if isinstance(c, str):
                    urls.append(c[:60])
                elif isinstance(c, dict):
                    urls.append(c.get("title", c.get("url", ""))[:60])
            cite_str = " | Sources: " + "; ".join(u for u in urls if u)
        lines.append(f'- "{query}" ({num} results{cite_str})')
    return "\n".join(lines)


def _task_desc(instance: TaskInstance) -> str:
    return {
        "stock": "Stock return ranking prediction",
        "salary": "Free-agent salary prediction",
        "legal": "Supreme Court case outcome prediction",
    }.get(instance.task, "Prediction task")


def _event_desc(instance: TaskInstance) -> str:
    if instance.task == "stock":
        p = instance.raw.get("past_period", {})
        return f"Stock ranking {p.get('start_date', '')} to {p.get('end_date', '')}"
    if instance.task == "salary":
        return (
            f"Contract for {instance.raw.get('Player', '')} "
            f"({instance.raw.get('Year', '')})"
        )
    if instance.task == "legal":
        return f"Outcome of {instance.raw.get('case_name', '')}"
    return instance.question_text


# ---------------------------------------------------------------------------
# TimeSPEC Agent
# ---------------------------------------------------------------------------


class TimeSpecAgent:
    """5-phase inference-time temporal leakage blocking agent."""

    def __init__(
        self,
        *,
        model: str,
        detector_model: str = "google/gemini-2.5-flash",
        search_cache: PersistentSearchCache | None = None,
        max_iterations: int = 15,
        min_search_results: int = 10,
    ) -> None:
        self.model = model
        self.llm = LLMClient()
        self.search = SearchClient()
        self.extractor = ClaimExtractor(model=detector_model)
        self.detector = LeakageDetector(
            model=detector_model,
            use_search=True,
            use_llm_judge=True,
            persistent_cache=search_cache,
        )
        self.max_iterations = max_iterations
        self.min_search_results = min_search_results

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(self, instance: TaskInstance) -> dict[str, Any]:
        """Run the full 5-phase pipeline and return prediction + rationale."""
        cutoff = instance.cutoff_date
        logger.info(
            "TimeSPEC | task=%s | id=%s | cutoff=%s",
            instance.task, instance.instance_id, cutoff,
        )

        # Phase 1: Generator
        gen = self._generator_phase(instance)
        logger.info("Phase 1 done: %d searches", len(gen["search_history"]))

        # Phase 2: Supervisor 1
        sup1 = self._supervisor_phase(gen["draft_rationale"], instance)
        logger.info(
            "Phase 2 done: %d/%d valid claims",
            len(sup1["valid_claims"]), sup1["total_claims"],
        )

        # Phase 3: Regenerator (conditional)
        regen = None
        if sup1["regeneration_needed"]:
            logger.info(
                "Phase 3: regenerating (%d violations)", len(sup1["violations"]),
            )
            regen = self._regenerator_phase(instance, sup1, gen["search_history"])
        else:
            logger.info("Phase 3: skipped (no violations)")

        # Phase 4: Supervisor 2 (conditional)
        sup2 = None
        if regen is not None:
            sup2 = self._supervisor_phase(
                regen["regenerated_rationale"], instance,
            )
            logger.info(
                "Phase 4 done: %d/%d valid claims",
                len(sup2["valid_claims"]), sup2["total_claims"],
            )
        else:
            logger.info("Phase 4: skipped")

        # Phase 5: Aggregator (ALWAYS runs)
        all_valid = list(sup1["valid_claims"])
        if sup2:
            all_valid.extend(sup2["valid_claims"])

        agg = self._aggregator_phase(instance, all_valid)
        logger.info("Phase 5 done: final prediction ready")

        return {
            "prediction": agg.get("prediction"),
            "rationale": agg.get("rationale", ""),
            "metadata": {
                "stages_completed": self._stages_completed(regen is not None),
                "regeneration_triggered": regen is not None,
                "supervisor_1_violations": len(sup1["violations"]),
                "supervisor_2_violations": (
                    len(sup2["violations"]) if sup2 else 0
                ),
                "total_valid_claims": len(all_valid),
                "generator_searches": len(gen["search_history"]),
            },
        }

    @staticmethod
    def _stages_completed(regenerated: bool) -> list[str]:
        stages = ["generator", "supervisor_1"]
        if regenerated:
            stages.extend(["regenerator", "supervisor_2"])
        stages.append("aggregator")
        return stages

    # ------------------------------------------------------------------
    # Phase 1: Generator
    # ------------------------------------------------------------------

    def _generator_phase(self, instance: TaskInstance) -> dict[str, Any]:
        messages = [
            {
                "role": "system",
                "content": _generator_system(instance.task, instance.cutoff_date),
            },
            {"role": "user", "content": _generator_user(instance)},
        ]
        tools = [
            _make_search_tool(instance.cutoff_date),
            _make_draft_tool(instance.task),
        ]
        search_history: list[dict] = []

        result = self._call_with_tools(
            messages, tools, instance.cutoff_date, search_history,
        )
        return {
            "draft_prediction": result.get("prediction"),
            "draft_rationale": result.get("rationale", ""),
            "search_history": search_history,
        }

    # ------------------------------------------------------------------
    # Phase 2 / 4: Supervisor
    # ------------------------------------------------------------------

    def _supervisor_phase(
        self, rationale: str, instance: TaskInstance,
    ) -> dict[str, Any]:
        empty = {
            "valid_claims": [],
            "violations": [],
            "total_claims": 0,
            "regeneration_needed": False,
        }
        if not rationale:
            return empty

        extraction = self.extractor.extract(
            rationale,
            task_description=_task_desc(instance),
            event_description=_event_desc(instance),
            cutoff_date=instance.cutoff_date,
        )
        claims = extraction.claims
        if not claims:
            return empty

        leak_report = self.detector.detect(claims, instance.cutoff_date)

        valid_claims: list[dict] = []
        violations: list[dict] = []
        for r in leak_report.results:
            entry = {
                "claim_id": r.claim_id,
                "text": r.text,
                "category": r.category,
                "is_leaked": r.is_leaked,
                "method": r.method,
                "determination_date": r.determination_date,
                "reason": r.reason,
            }
            if r.is_leaked:
                violations.append(entry)
            else:
                valid_claims.append(entry)

        return {
            "valid_claims": valid_claims,
            "violations": violations,
            "total_claims": len(claims),
            "regeneration_needed": len(violations) > 0,
        }

    # ------------------------------------------------------------------
    # Phase 3: Regenerator
    # ------------------------------------------------------------------

    def _regenerator_phase(
        self,
        instance: TaskInstance,
        sup1: dict[str, Any],
        gen_search_history: list[dict],
    ) -> dict[str, Any]:
        previous_queries = [s.get("query", "") for s in gen_search_history]

        messages = [
            {
                "role": "system",
                "content": _regenerator_system(
                    instance.task, instance.cutoff_date,
                ),
            },
            {
                "role": "user",
                "content": _regenerator_user(
                    instance,
                    sup1["valid_claims"],
                    gen_search_history,
                    previous_queries,
                ),
            },
        ]
        tools = [
            _make_search_tool(instance.cutoff_date),
            _make_draft_tool(instance.task),
        ]
        regen_search: list[dict] = []

        result = self._call_with_tools(
            messages, tools, instance.cutoff_date, regen_search,
        )
        return {
            "regenerated_prediction": result.get("prediction"),
            "regenerated_rationale": result.get("rationale", ""),
            "regeneration_search_history": regen_search,
        }

    # ------------------------------------------------------------------
    # Phase 5: Aggregator
    # ------------------------------------------------------------------

    def _aggregator_phase(
        self, instance: TaskInstance, valid_claims: list[dict],
    ) -> dict[str, Any]:
        messages = [
            {
                "role": "system",
                "content": _aggregator_system(
                    instance.task, instance.cutoff_date,
                ),
            },
            {
                "role": "user",
                "content": _aggregator_user(instance, valid_claims),
            },
        ]
        tools = [_make_final_tool(instance.task)]

        response = self.llm.call(
            messages,
            model=self.model,
            temperature=0.0,
            max_tokens=4096,
            tools=tools,
            tool_choice={
                "type": "function",
                "function": {"name": "submit_final_prediction"},
            },
        )

        tc = (response.get("tool_calls") or [None])[0]
        if tc:
            fn = tc.get("function", {})
            args = fn.get("arguments", {}) if isinstance(fn, dict) else {}
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {"prediction": None, "rationale": args}
            return args

        return {"prediction": None, "rationale": response.get("content", "")}

    # ------------------------------------------------------------------
    # Tool-calling loop (generator / regenerator)
    # ------------------------------------------------------------------

    def _call_with_tools(
        self,
        messages: list[dict],
        tools: list[dict],
        cutoff_date: str,
        search_history: list[dict],
    ) -> dict[str, Any]:
        total_results = 0
        sufficient = False

        for _ in range(self.max_iterations):
            response = self.llm.call(
                messages,
                model=self.model,
                temperature=0.0,
                max_tokens=4096,
                tools=tools,
            )

            tool_calls = response.get("tool_calls") or []
            if not tool_calls:
                content = response.get("content", "")
                return self._try_parse_prediction(content)

            for tc in tool_calls:
                fn = tc.get("function", tc.get("name", ""))
                if isinstance(fn, dict):
                    fn_name = fn.get("name", "")
                    fn_args = fn.get("arguments", {})
                else:
                    fn_name = fn
                    fn_args = tc.get("arguments", {})

                if isinstance(fn_args, str):
                    try:
                        fn_args = json.loads(fn_args)
                    except json.JSONDecodeError:
                        fn_args = {}

                tc_id = tc.get("id", f"call_{len(search_history)}")

                if fn_name == "search_information":
                    if sufficient:
                        messages.append({
                            "role": "user",
                            "content": (
                                f"Sufficient evidence collected "
                                f"({total_results} results). "
                                "Submit your prediction now."
                            ),
                        })
                        continue

                    query = fn_args.get("query", "")
                    sr = self.search.search(query, cutoff_date=cutoff_date)
                    citations = sr.get("citations", [])
                    has_content = bool(sr.get("content", "").strip())
                    num = (
                        len(citations) if citations
                        else (1 if has_content else 0)
                    )
                    total_results += num
                    search_history.append({
                        "query": query,
                        "num_results": num,
                        "citations": citations,
                        "timestamp": datetime.now().isoformat(),
                    })
                    if total_results >= self.min_search_results:
                        sufficient = True

                    messages.append({
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [{
                            "id": tc_id,
                            "type": "function",
                            "function": {
                                "name": fn_name,
                                "arguments": json.dumps(fn_args),
                            },
                        }],
                    })
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc_id,
                        "content": json.dumps(
                            sr, ensure_ascii=False, default=str,
                        ),
                    })

                elif fn_name == "submit_draft_prediction":
                    return fn_args

                elif fn_name == "submit_final_prediction":
                    return fn_args

                else:
                    logger.warning("Unknown tool call: %s", fn_name)

        logger.warning("Max iterations reached; returning best-effort prediction")
        return {"prediction": None, "rationale": ""}

    @staticmethod
    def _try_parse_prediction(text: str) -> dict[str, Any]:
        try:
            return json.loads(text)
        except (json.JSONDecodeError, TypeError):
            return {"prediction": None, "rationale": text}
