"""CLM vs. RLM on BrowseComp-Plus (Chen et al., 2025), the CLM paper's headline benchmark.

A deep-research task: answer hard fact-seeking questions by searching a fixed ~100K-document corpus with
`search(query)` (top 5 documents, 512-token snippets) and `get_document(docid)` (capped at 8,192 tokens), as
in the BrowseComp-Plus and CLM papers. Documents have to be read, so they pile up in the agent's context
over many steps. Answers are graded by an LLM judge with the official BrowseComp-Plus grader template.

Methods (same model, tools, and step limit):
  RLM              stock dspy.RLM, append-only history, no budget
  RLM @ budget     dspy.RLM held to the budget (overflow ends the run, as for CLM)
  CLM @ budget     dspy.CLM with DSPy's built-in context prompt
  CLM-paper @ budget  dspy.CLM with the paper's own context prompt, reminders, and shrink edit gate
                   (needs --paper-repo, a local clone of facebookresearch/context-language-models)

Setup:
  pip install "dspy[deno]" datasets bm25s PyStemmer
  python docs/docs/tutorials/clm/browsecomp_prepare.py --data-dir data/bcp
  python docs/docs/tutorials/clm/browsecomp.py --data-dir data/bcp --paper-repo ../context-language-models
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import litellm
import tiktoken
from baselines import run_program

import dspy

ENCODING = tiktoken.get_encoding("o200k_base")

# The official grader template, from texttron/BrowseComp-Plus (MIT), search_agent/prompts.py.
GRADER_TEMPLATE = """
Judge whether the following [response] to [question] is correct or not based on the precise and unambiguous [correct_answer] below.

[question]: {question}

[response]: {response}

Your judgement must be in the format and criteria specified below:

extracted_final_answer: The final exact answer extracted from the [response]. Put the extracted answer as 'None' if there is no exact, final answer to extract from the response.

[correct_answer]: {correct_answer}

reasoning: Explain why the extracted_final_answer is correct or incorrect based on [correct_answer], focusing only on if there are meaningful differences between [correct_answer] and the extracted_final_answer. Do not comment on any background to the problem, do not attempt to solve the problem, do not argue for any answer different than [correct_answer], focus only on whether the answers match.

correct: Answer 'yes' if extracted_final_answer matches the [correct_answer] given above, or is within a small margin of error for numerical problems. Answer 'no' otherwise, i.e. if there if there is any inconsistency, ambiguity, non-equivalency, or if the extracted answer is incorrect.


confidence: The extracted confidence score between 0|\\%| and 100|\\%| from [response]. Put 100 if there is no confidence score available.
""".strip()


class DeepResearch(dspy.Signature):
    """You are a deep research agent. Answer the question by searching a fixed document corpus with the
    search(query) and get_document(docid) tools. Search and read step by step, in an interleaved manner, and
    verify your answer against the documents before submitting. Every question has an answer that can be found
    in the corpus: do not give up. If a line of search stalls, try different queries and read more documents."""

    question: str = dspy.InputField()
    explanation: str = dspy.OutputField(desc="your explanation, citing evidence docids in square brackets, e.g. [20]")
    exact_answer: str = dspy.OutputField(desc="your succinct, final answer")
    confidence: int = dspy.OutputField(desc="your confidence in the answer, 0-100")


# =============================================================================
# Retrieval
# =============================================================================


class Corpus:
    """BM25 over the BrowseComp-Plus corpus, reading documents lazily from corpus.jsonl."""

    def __init__(self, data_dir: str, snippet_tokens: int = 512, document_tokens: int = 8192, k: int = 5):
        import bm25s
        import Stemmer

        self.path = os.path.join(data_dir, "corpus.jsonl")
        self.retriever = bm25s.BM25.load(os.path.join(data_dir, "bm25"), mmap=True)
        self.stemmer = Stemmer.Stemmer("english")
        self.snippet_tokens, self.document_tokens, self.k = snippet_tokens, document_tokens, k
        self.offsets, self.docids, self.position = [], [], {}
        with open(self.path, "rb") as f:
            offset = 0
            for line in f:
                docid = json.loads(line[:200].split(b'", "text"')[0] + b'"}')["docid"]
                self.position[docid] = len(self.offsets)
                self.offsets.append(offset)
                self.docids.append(docid)
                offset += len(line)
        self.lock = threading.Lock()

    def text(self, index: int) -> str:
        with open(self.path, "rb") as f:
            f.seek(self.offsets[index])
            return json.loads(f.readline())["text"]

    @staticmethod
    def _truncate(text: str, n_tokens: int) -> tuple[str, bool]:
        tokens = ENCODING.encode(text, disallowed_special=())
        return (ENCODING.decode(tokens[:n_tokens]), True) if len(tokens) > n_tokens else (text, False)

    def tools(self, log: dict, calls_per_step: int | None = None):
        """Per-question search/get_document tools that record what the agent retrieved.

        With ``calls_per_step``, calls beyond that many in one step are refused; the step hook resets
        ``log["step_calls"]`` before each step.
        """
        import bm25s

        def over_limit() -> str | None:
            log["step_calls"] = log.get("step_calls", 0) + 1
            if calls_per_step is not None and log["step_calls"] > calls_per_step:
                log["refused"] = log.get("refused", 0) + 1
                return (f"[refused: only {calls_per_step} search/get_document call(s) per step. Read this step's "
                        "result first, then make your next call in the next step.]")
            return None

        def search(query: str) -> str:
            """Search the corpus. Returns the top 5 documents for the query, each with its docid and a snippet
            (the first ~512 tokens of the document)."""
            if refused := over_limit():
                return refused
            log["search"] += 1
            with self.lock:
                query_tokens = bm25s.tokenize([query], stopwords="en", stemmer=self.stemmer, show_progress=False)
                hits, scores = self.retriever.retrieve(query_tokens, k=self.k, show_progress=False, n_threads=1)
            results = []
            for index, score in zip(hits[0], scores[0]):
                docid = self.docids[int(index)]
                log["retrieved"].add(docid)
                snippet, _ = self._truncate(self.text(int(index)), self.snippet_tokens)
                results.append(f"docid: {docid} (score {score:.2f})\n{snippet}")
            return "\n\n".join(results) if results else "No results."

        def get_document(docid: str) -> str:
            """Return the full text of the document with this docid (truncated to ~8,192 tokens)."""
            if refused := over_limit():
                return refused
            log["get_document"] += 1
            docid = str(docid).strip()
            if docid not in self.position:
                return f"No document with docid {docid!r}."
            log["retrieved"].add(docid)
            text, truncated = self._truncate(self.text(self.position[docid]), self.document_tokens)
            return f"docid: {docid}\n{text}" + ("\n[... truncated]" if truncated else "")

        return [search, get_document]


# =============================================================================
# Evaluation
# =============================================================================


def make_lm(model: str, **kwargs) -> dspy.LM:
    return dspy.LM(
        model, api_base=os.environ["AZURE_BASE"], api_key=os.environ["AZURE_API_KEY"],
        temperature=1.0, max_tokens=16000, num_retries=8, **kwargs,
    )


def judge(judge_lm: dspy.LM, question: str, response: str, answer: str) -> bool:
    prompt = GRADER_TEMPLATE.format(question=question, response=response, correct_answer=answer)
    verdict = judge_lm(messages=[{"role": "user", "content": prompt}])[0]
    match = re.search(r"correct:\s*\**\s*(yes|no)", verdict, re.I)
    return bool(match and match.group(1).lower() == "yes")


def usage_cost(usage: dict) -> tuple[int, float]:
    """Prompt tokens and USD from dspy's per-model usage, priced with LiteLLM's model map."""
    prompt_tokens, dollars = 0, 0.0
    for model, u in usage.items():
        prices = litellm.model_cost.get(model.split("/", 1)[-1], {})
        cached = ((u.get("prompt_tokens_details") or {}).get("cached_tokens")) or 0
        prompt = u.get("prompt_tokens", 0) or 0
        completion = u.get("completion_tokens", 0) or 0
        prompt_tokens += prompt
        dollars += (prompt - cached) * prices.get("input_cost_per_token", 0)
        dollars += cached * prices.get("cache_read_input_token_cost", prices.get("input_cost_per_token", 0))
        dollars += completion * prices.get("output_cost_per_token", 0)
    return prompt_tokens, dollars


def run_one(method: str, row: dict, corpus: Corpus, args, clm_class, judge_lm) -> dict:
    log = {"search": 0, "get_document": 0, "retrieved": set(), "step_calls": 0, "refused": 0}

    def reset_step_calls():
        log["step_calls"] = 0

    signature = DeepResearch
    if args.calls_per_step:
        signature = DeepResearch.with_instructions(
            DeepResearch.instructions + f" Make at most {args.calls_per_step} search or get_document call(s) per step:"
            " print the result, read it, and decide what to do next in the following step."
        )
    t0 = time.time()
    try:
        pred = run_program(
            method.replace("CLM-paper", "CLM"), signature, corpus.tools(log, args.calls_per_step), args.max_iters,
            args.budget, step_hook=reset_step_calls if args.calls_per_step else None,
            clm_class=clm_class if method.startswith("CLM-paper") else None,
            module_kwargs={"max_output_chars": args.max_output_chars}, question=row["query"],
        )
        error = None
    except Exception as e:  # a crash scores zero but keeps the sweep going
        pred, error = None, f"{type(e).__name__}: {e}"
    response = (
        f"Explanation: {pred.explanation}\nExact Answer: {pred.exact_answer}\nConfidence: {pred.confidence}%"
        if pred is not None else ""
    )
    correct = bool(response) and judge(judge_lm, row["query"], response, row["answer"])
    prompt_tokens, dollars = usage_cost(pred.get_lm_usage() or {}) if pred is not None else (0, 0.0)
    evidence = set(row["evidence_docids"])
    return {
        "method": method,
        "query_id": row["query_id"],
        "correct": correct,
        "exact_answer": getattr(pred, "exact_answer", None),
        "answer": row["answer"],
        "evidence_recall": len(evidence & log["retrieved"]) / len(evidence) if evidence else None,
        "search_calls": log["search"],
        "get_document_calls": log["get_document"],
        "refused_calls": log["refused"],
        "stats": getattr(pred, "context_stats", {}) or {},
        "final_reasoning": getattr(pred, "final_reasoning", None),
        "prompt_tokens": prompt_tokens,
        "usd": dollars,
        "seconds": round(time.time() - t0, 1),
        "error": error,
        "trajectory": getattr(pred, "trajectory", None),
        "final_context": getattr(pred, "final_context", None),
    }


def summarize(rows: list[dict], methods: list[str]) -> None:
    header = (f"{'method':<20} {'n':>3} {'accuracy':>8} {'recall':>7} {'search':>7} {'get_doc':>7} {'overflow':>8} "
              f"{'edits':>6} {'peak ctx':>9} {'prefill':>9} {'prompt tok':>11} {'$/q':>7}")
    print(header)
    print("-" * len(header))
    for m in methods:
        rs = [r for r in rows if r["method"] == m]
        if not rs:
            continue
        n = len(rs)

        def mean(values):
            values = [v for v in values if v is not None]
            return sum(values) / len(values) if values else 0.0

        print(
            f"{m:<20} {n:>3} {mean(r['correct'] for r in rs):>8.2f} {mean(r['evidence_recall'] for r in rs):>7.2f} "
            f"{mean(r['search_calls'] for r in rs):>7.1f} {mean(r['get_document_calls'] for r in rs):>7.1f} "
            f"{sum(bool(r['stats'].get('context_overflow')) for r in rs):>5}/{n:<2} "
            f"{mean(r['stats'].get('edits_applied') for r in rs):>6.1f} "
            f"{mean(r['stats'].get('peak_context_tokens') for r in rs):>9.0f} "
            f"{mean(r['stats'].get('prefill_tokens_with_prefix_reuse') for r in rs):>9.0f} "
            f"{mean(r['prompt_tokens'] for r in rs):>11.0f} {mean(r['usd'] for r in rs):>7.3f}"
        )
    for r in rows:
        if r["error"]:
            print(f"! {r['method']} {r['query_id']}: {r['error'][:200]}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--model", default="openai/gpt-5.6-luna")
    parser.add_argument("--judge-model", default="openai/gpt-5.6-terra")
    parser.add_argument("--n", type=int, default=30, help="number of questions, sampled with --seed")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--budget", type=int, default=24000, help="managed-context budget (~4 chars/token)")
    parser.add_argument("--max-iters", type=int, default=40)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--calls-per-step", type=int, default=None, help="cap tool calls per step (paper: 1)")
    parser.add_argument("--max-output-chars", type=int, default=60000, help="per-step output cap (paper: 60,000)")
    parser.add_argument("--methods", default="RLM,RLM @ budget,CLM-paper @ budget")
    parser.add_argument("--paper-repo", default=None, help="local clone of facebookresearch/context-language-models")
    parser.add_argument("--out", default="browsecomp_results.jsonl", help="per-run results; reruns resume from it")
    args = parser.parse_args()

    methods = [m.strip() for m in args.methods.split(",")]
    clm_class = None
    if any(m.startswith("CLM-paper") for m in methods):
        if not args.paper_repo:
            parser.error("CLM-paper methods need --paper-repo")
        from paper_prompt import paper_clm_class

        clm_class = paper_clm_class(args.paper_repo)

    dspy.configure(lm=make_lm(args.model), track_usage=True)
    judge_lm = make_lm(args.judge_model)
    queries = [json.loads(line) for line in open(os.path.join(args.data_dir, "queries.jsonl"))]
    sample = random.Random(args.seed).sample(queries, args.n)
    corpus = Corpus(args.data_dir)

    done = {}
    if os.path.exists(args.out):
        for line in open(args.out):
            r = json.loads(line)
            done[(r["method"], r["query_id"])] = r
    jobs = [(m, row) for row in sample for m in methods if (m, row["query_id"]) not in done]
    print(f"{len(sample)} questions x {len(methods)} methods; {len(jobs)} runs to go ({len(done)} cached in {args.out}).")

    write_lock = threading.Lock()
    with ThreadPoolExecutor(args.threads) as pool, open(args.out, "a") as out:
        futures = [pool.submit(run_one, m, row, corpus, args, clm_class, judge_lm) for m, row in jobs]
        for i, future in enumerate(as_completed(futures), 1):
            r = future.result()
            done[(r["method"], r["query_id"])] = r
            with write_lock:
                out.write(json.dumps(r, default=str) + "\n")
                out.flush()
            print(f"[{i}/{len(jobs)}] {r['method']:<20} q{r['query_id']:<5} correct={r['correct']!s:<5} "
                  f"steps={r['stats'].get('steps', r['stats'].get('lm_calls'))} ${r['usd']:.3f} {r['seconds']}s", flush=True)

    sample_ids = {row["query_id"] for row in sample}
    summarize([r for (m, q), r in done.items() if q in sample_ids and m in methods], methods)


if __name__ == "__main__":
    main()
