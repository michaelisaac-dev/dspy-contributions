from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Generic, cast, overload

from pydantic.fields import FieldInfo
from typing_extensions import ParamSpec, TypeVar

import dspy
from dspy.primitives.module import Module
from dspy.signatures.signature import Signature, ensure_signature

if TYPE_CHECKING:
    from dspy.predict.predict import Predict

# Defaults keep a bare `ChainOfThought` annotation meaning an untyped module (PEP 696).
P = ParamSpec("P", default=...)
S = TypeVar("S", bound=Signature, default=Signature, covariant=True)

# NOTE: This restores the legacy rationale_field behavior after PR #8822.


class ChainOfThought(Module, Generic[P, S]):
    """A module that reasons before producing the output.

    Uses native reasoning when supported, otherwise requests a short plan before
    answering. The result exposes either as ``reasoning``; native reasoning may
    be ``None`` when the provider does not return a summary.

    Args:
        signature: The DSPy signature defining the module's inputs and outputs.
        rationale_field: An optional custom ``FieldInfo`` for the reasoning field.
            Its annotation and description are preserved.
        rationale_field_type: Defaults to ``dspy.Reasoning``. Pass ``str`` to
            request an explicit reasoning field instead. Ignored if
            ``rationale_field`` is provided.
        **config: Additional keyword arguments forwarded to the underlying
            ``dspy.Predict`` module.

    Example:
        >>> import dspy
        >>> cot = dspy.ChainOfThought("question -> answer")
        >>> result = cot(question="What is the capital of France?")
        >>> print(result.reasoning)
        >>> print(result.answer)
    """

    # Typed like Predict: calls take the signature's inputs and return a prediction of it. The added
    # `reasoning` field is not part of the signature, so type checkers see it as Any.
    @overload
    def __init__(
        self,
        signature: Callable[P, S],
        rationale_field: FieldInfo | None = None,
        rationale_field_type: type | None = None,
        **config: Any,
    ) -> None: ...

    @overload
    def __init__(
        self: "ChainOfThought[..., Signature]",
        signature: str | type[Signature] | Signature,
        rationale_field: FieldInfo | None = None,
        rationale_field_type: type | None = None,
        **config: Any,
    ) -> None: ...

    def __init__(
        self,
        signature: str | type[Signature] | Signature,
        rationale_field: FieldInfo | None = None,
        rationale_field_type: type | None = None,
        **config: Any,
    ):
        super().__init__()
        signature = ensure_signature(signature)
        rationale_field_type = rationale_field_type or dspy.Reasoning
        use_default = rationale_field is None and rationale_field_type is dspy.Reasoning
        desc = "A short plan for solving the task." if use_default else "${reasoning}"
        rationale_field_type = rationale_field.annotation if rationale_field else rationale_field_type
        rationale_field = rationale_field if rationale_field else dspy.OutputField(desc=desc)
        extended_signature = signature.prepend(name="reasoning", field=rationale_field, type_=rationale_field_type)
        if use_default:
            extended_signature = extended_signature.with_updated_fields("reasoning", __dspy_reasoning_fallback="plan")
        # The extended signature has the same inputs and adds `reasoning` to the outputs.
        self.predict = cast("Predict[P, S]", dspy.Predict(extended_signature, **config))

    def forward(self, *args: P.args, **kwargs: P.kwargs) -> S:
        return self.predict(*args, **kwargs)

    async def aforward(self, *args: P.args, **kwargs: P.kwargs) -> S:
        return await self.predict.acall(*args, **kwargs)
