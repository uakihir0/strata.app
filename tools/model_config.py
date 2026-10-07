"""Model context settings that do not depend on the installer or host platform."""
from __future__ import annotations


def derived_factor(context: int, trained: int = 262_144) -> float:
    """Return final context divided by trained context, clamped to one."""
    return max(1.0, float(context) / float(trained))


def resolve_rope(
    context: int,
    scaling: str | None,
    scale: float | None,
    trained: int = 262_144,
) -> tuple[str | None, float | None]:
    """Resolve RoPE settings for the context that will actually be served."""
    if context <= trained:
        if scale is not None and scaling in (None, "none"):
            raise ValueError(
                "--rope-scale needs --rope-scaling linear or yarn (the chosen context fits the "
                "trained 262144, so there is nothing to scale)"
            )
        if scaling in (None, "none"):
            return None, None
        return scaling, scale if scale is not None else derived_factor(context, trained)
    if scaling == "none":
        raise ValueError(
            f"a {context // 1024}K context is past the model's trained {trained}, and "
            "--rope-scaling none keeps the stock angles there. Pick --rope-scaling yarn or "
            "linear, or use a context within the trained range"
        )
    return scaling or "yarn", scale if scale is not None else derived_factor(context, trained)
