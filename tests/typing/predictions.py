"""Static contracts for signature-typed modules, checked by ty rather than executed by pytest."""

from typing import Any

from typing_extensions import assert_type

import dspy


class QA(dspy.Signature):
    """Answer the question."""

    question: str = dspy.InputField()
    context: list[str] = dspy.InputField(default=[])
    answer: int = dspy.OutputField()


class RAG(dspy.Module):
    def __init__(self) -> None:
        super().__init__()
        self.respond = dspy.ChainOfThought(QA)

    def forward(self, question: str) -> QA:
        return self.respond(question=question, context=["retrieved passage"])


def metric(example: dspy.Example, pred: dspy.Prediction) -> float:
    return float(example.answer == pred.answer)


def use_untyped(predictors: list[dspy.Predict], module: dspy.Module) -> None: ...


def check_predict(lm: dspy.LM) -> None:
    predict = dspy.Predict(QA)
    pred = predict(question="What is 2 + 2?")
    assert_type(pred, QA)
    assert_type(pred.answer, int)
    metric(dspy.Example(answer=4), pred)

    predict(question="q", context=["c"], lm=lm, config={"temperature": 0.0}, demos=[])

    predict(questoin="q")  # ty: ignore[unknown-argument, missing-argument]
    predict(question=1)  # ty: ignore[invalid-argument-type]
    predict(question="q", answer=1)  # ty: ignore[unknown-argument]
    predict("q")  # ty: ignore[too-many-positional-arguments, missing-argument]


async def check_async() -> None:
    assert_type(await dspy.Predict(QA).acall(question="q"), QA)
    assert_type(await dspy.ChainOfThought(QA).acall(question="q"), QA)
    await dspy.Predict(QA).acall(question=1)  # ty: ignore[invalid-argument-type]


def check_chain_of_thought() -> None:
    cot = dspy.ChainOfThought(QA)
    pred = cot(question="q")
    assert_type(pred, QA)
    assert_type(pred.answer, int)
    assert_type(pred.reasoning, Any)
    cot(question=1)  # ty: ignore[invalid-argument-type]


def check_custom_module() -> None:
    rag = RAG()
    assert_type(rag(question="q"), QA)
    assert_type(rag(question="q").answer, int)
    rag(question=1)  # ty: ignore[invalid-argument-type]
    rag(query="q")  # ty: ignore[unknown-argument, missing-argument]


def check_untyped_signatures(dynamic: type[dspy.Signature], program: dspy.Module) -> None:
    assert_type(dspy.Predict("question -> answer")(question="q"), dspy.Signature)
    assert_type(dspy.Predict(dynamic)(anything="q"), dspy.Signature)
    assert_type(dspy.ChainOfThought("question -> answer")(question="q"), dspy.Signature)
    assert_type(dspy.Predict(QA.append("confidence", dspy.OutputField()))(question="q"), dspy.Signature)
    assert_type(program(question="q"), Any)


def check_runtime_built_signatures() -> None:
    from_string = dspy.Signature("question -> answer", "Answer briefly.")
    from_fields = dspy.Signature({"question": (str, dspy.InputField()), "answer": (str, dspy.OutputField())})
    assert_type(dspy.Predict(from_string)(question="q"), dspy.Signature)
    assert_type(dspy.ChainOfThought(from_fields)(question="q"), dspy.Signature)


def check_bare_annotations() -> None:
    predict = dspy.Predict(QA)
    cot = dspy.ChainOfThought(QA)
    use_untyped([predict, dspy.Predict("question -> answer")], cot)

