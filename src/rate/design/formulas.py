"""Helpers for reading and validating metadata-driven model specifications."""

from __future__ import annotations

from dataclasses import dataclass, field
import ast
from typing import Any, Mapping

from patsy import ModelDesc


class ModelSpecError(ValueError):
    """Raised when a model definition is missing required fields or is invalid."""


@dataclass(frozen=True)
class ModelSpec:
    """Lightweight normalized model specification.

    Parameters
    ----------
    section
        Top-level config section, for example ``"glm"`` or ``"lmeeeg"``.
    name
        Model name within the section.
    formula
        Patsy-style formula string, including outcome when appropriate.
    outcome
        Explicit outcome variable. When omitted in config it is inferred from
        the left-hand side of ``formula`` when possible.
    eligibility
        Explicit inclusion-rule configuration to apply before formula evaluation.
    test_predictors
        Predictors or patsy term names that downstream code will test.
    standardize
        Numeric columns to z-score in place after eligibility filtering.
    derived
        Mapping of derived-column names to simple pandas-evaluable expressions.
    required_columns
        Additional columns that must be non-missing before formula expansion.
    config
        Original normalized per-model config payload.
    """

    section: str
    name: str
    formula: str
    outcome: str | None = None
    eligibility: Mapping[str, Any] | None = None
    test_predictors: tuple[str, ...] = ()
    standardize: tuple[str, ...] = ()
    derived: Mapping[str, str] = field(default_factory=dict)
    required_columns: tuple[str, ...] = ()
    config: Mapping[str, Any] = field(default_factory=dict)


def load_model_spec(
    config: Mapping[str, Any],
    *,
    section: str,
    model_name: str | None = None,
) -> ModelSpec:
    """Load and normalize a model specification from repository config.

    Parameters
    ----------
    config
        Full repository config or a mapping containing the requested section.
    section
        Config section containing models, for example ``"glm"``.
    model_name
        Optional model name. When omitted, the section must resolve to exactly
        one model definition.

    Returns
    -------
    ModelSpec
        Normalized model specification.
    """
    section_config = config.get(section, config) if isinstance(config, Mapping) else config
    if not isinstance(section_config, Mapping):
        raise ModelSpecError(f"Config section `{section}` must be a mapping.")

    available_models = _normalize_models(section_config, section=section)
    resolved_name = _resolve_model_name(available_models, requested_name=model_name, section=section)
    raw_spec = available_models[resolved_name]
    return _build_model_spec(section=section, model_name=resolved_name, model_config=raw_spec)


def list_model_names(config: Mapping[str, Any], *, section: str) -> list[str]:
    """Return normalized model names available in a config section."""
    section_config = config.get(section, config) if isinstance(config, Mapping) else config
    if not isinstance(section_config, Mapping):
        raise ModelSpecError(f"Config section `{section}` must be a mapping.")
    return list(_normalize_models(section_config, section=section))


def get_model_formula(model_spec: ModelSpec | Mapping[str, Any]) -> str:
    """Return the formula string for a model spec."""
    return _coerce_model_spec(model_spec).formula


def get_test_predictors(model_spec: ModelSpec | Mapping[str, Any]) -> list[str]:
    """Return predictor names marked for downstream testing."""
    return list(_coerce_model_spec(model_spec).test_predictors)


def get_standardize_columns(model_spec: ModelSpec | Mapping[str, Any]) -> list[str]:
    """Return columns that should be standardized before matrix construction."""
    return list(_coerce_model_spec(model_spec).standardize)


def get_derived_terms(model_spec: ModelSpec | Mapping[str, Any]) -> dict[str, str]:
    """Return derived-variable definitions keyed by destination column name."""
    return dict(_coerce_model_spec(model_spec).derived)


def get_required_columns(model_spec: ModelSpec | Mapping[str, Any]) -> list[str]:
    """Return explicit required columns declared in config."""
    return list(_coerce_model_spec(model_spec).required_columns)


def get_formula_variable_names(model_spec: ModelSpec | Mapping[str, Any]) -> list[str]:
    """Extract candidate metadata column names referenced by the model formula."""
    spec = _coerce_model_spec(model_spec)
    model_desc = ModelDesc.from_formula(spec.formula)
    variable_names: list[str] = []
    for term_group in (model_desc.lhs_termlist, model_desc.rhs_termlist):
        for term in term_group:
            for factor in term.factors:
                variable_names.extend(_extract_variable_names(factor.name()))
    return sorted(dict.fromkeys(variable_names))


def infer_outcome_from_formula(formula: str) -> str | None:
    """Infer the single left-hand-side outcome name from a formula."""
    model_desc = ModelDesc.from_formula(formula)
    outcome_names: list[str] = []
    for term in model_desc.lhs_termlist:
        for factor in term.factors:
            outcome_names.extend(_extract_variable_names(factor.name()))
    unique_outcomes = list(dict.fromkeys(outcome_names))
    if len(unique_outcomes) == 1:
        return unique_outcomes[0]
    return None


def _build_model_spec(*, section: str, model_name: str, model_config: Mapping[str, Any]) -> ModelSpec:
    if not isinstance(model_config, Mapping):
        raise ModelSpecError(f"Model `{section}.{model_name}` must be a mapping.")

    formula = model_config.get("formula")
    if not isinstance(formula, str) or "~" not in formula:
        raise ModelSpecError(f"Model `{section}.{model_name}` must define a patsy formula containing `~`.")

    outcome = model_config.get("outcome")
    if outcome is not None and not isinstance(outcome, str):
        raise ModelSpecError(f"Model `{section}.{model_name}` outcome must be a string when provided.")
    outcome = outcome or infer_outcome_from_formula(formula)

    eligibility = _normalize_eligibility(
        model_config.get("eligibility"),
        section=section,
        model_name=model_name,
    )

    contrast_names = None
    if isinstance(model_config.get("contrasts"), list):
        contrast_names = [
            contrast["name"]
            for contrast in model_config["contrasts"]
            if isinstance(contrast, Mapping) and isinstance(contrast.get("name"), str)
        ]

    test_predictors = _normalize_string_list(
        model_config.get("test_predictors", model_config.get("regressors", contrast_names)),
        field_name="test_predictors",
        section=section,
        model_name=model_name,
    )
    standardize = _normalize_string_list(
        model_config.get("standardize"),
        field_name="standardize",
        section=section,
        model_name=model_name,
    )
    required_columns = _normalize_string_list(
        model_config.get("required_columns"),
        field_name="required_columns",
        section=section,
        model_name=model_name,
    )
    derived = _normalize_derived_terms(model_config.get("derived"), section=section, model_name=model_name)

    return ModelSpec(
        section=section,
        name=model_name,
        formula=formula,
        outcome=outcome,
        eligibility=eligibility,
        test_predictors=tuple(test_predictors),
        standardize=tuple(standardize),
        derived=derived,
        required_columns=tuple(required_columns),
        config=dict(model_config),
    )


def _normalize_eligibility(
    eligibility: Any,
    *,
    section: str,
    model_name: str,
) -> Mapping[str, Any] | None:
    if eligibility is None:
        return None
    if not isinstance(eligibility, Mapping):
        raise ModelSpecError(
            f"Model `{section}.{model_name}` eligibility must be a mapping when provided."
        )

    normalized = dict(eligibility)
    for field_name in ("not_null", "expressions"):
        if field_name not in normalized:
            continue
        normalized[field_name] = _normalize_string_list(
            normalized.get(field_name),
            field_name=f"eligibility.{field_name}",
            section=section,
            model_name=model_name,
        )

    for field_name in ("min_values", "max_values", "equals"):
        if field_name not in normalized:
            continue
        field_value = normalized.get(field_name)
        if not isinstance(field_value, Mapping):
            raise ModelSpecError(
                f"Model `{section}.{model_name}` eligibility.{field_name} must be a mapping when provided."
            )
        normalized[field_name] = dict(field_value)

    return normalized


def _normalize_models(section_config: Mapping[str, Any], *, section: str) -> dict[str, Mapping[str, Any]]:
    if "models" in section_config:
        models = section_config["models"]
        if not isinstance(models, Mapping):
            raise ModelSpecError(f"`{section}.models` must be a mapping.")
        return {str(name): value for name, value in models.items()}

    formulas = section_config.get("formulas")
    if isinstance(formulas, Mapping):
        fallback_models: dict[str, Mapping[str, Any]] = {}
        shared_test_predictors = section_config.get("test_predictors", section_config.get("regressors"))
        if shared_test_predictors is None and isinstance(section_config.get("contrasts"), list):
            shared_test_predictors = [
                contrast["name"]
                for contrast in section_config["contrasts"]
                if isinstance(contrast, Mapping) and isinstance(contrast.get("name"), str)
            ]
        shared_standardize = section_config.get("standardize")
        shared_eligibility = section_config.get("eligibility")
        shared_derived = section_config.get("derived")
        for name, formula in formulas.items():
            fallback_models[str(name)] = {
                "formula": formula,
                "eligibility": shared_eligibility,
                "test_predictors": shared_test_predictors,
                "standardize": shared_standardize,
                "derived": shared_derived,
            }
        return fallback_models

    if "formula" in section_config:
        default_name = str(section_config.get("default_model", "default"))
        return {default_name: section_config}

    fixed_effects = section_config.get("fixed_effects")
    if isinstance(fixed_effects, list) and all(isinstance(item, str) for item in fixed_effects):
        default_name = str(section_config.get("default_model", "default"))
        fixed_effect_formula = "~ " + " + ".join(fixed_effects) if fixed_effects else "~ 1"
        return {
            default_name: {
                "formula": fixed_effect_formula,
                "test_predictors": section_config.get("test_predictors", fixed_effects),
                "standardize": section_config.get("standardize"),
                "eligibility": section_config.get("eligibility"),
                "derived": section_config.get("derived"),
            }
        }

    raise ModelSpecError(
        f"Config section `{section}` must define either `models`, `formulas`, or a direct `formula`."
    )


def _resolve_model_name(
    models: Mapping[str, Mapping[str, Any]],
    *,
    requested_name: str | None,
    section: str,
) -> str:
    if requested_name is not None:
        if requested_name not in models:
            available = ", ".join(sorted(models))
            raise ModelSpecError(f"Unknown model `{section}.{requested_name}`. Available models: {available}")
        return requested_name

    if len(models) == 1:
        return next(iter(models))

    available = ", ".join(sorted(models))
    raise ModelSpecError(f"Section `{section}` defines multiple models; choose one of: {available}")


def _normalize_string_list(
    value: Any,
    *,
    field_name: str,
    section: str,
    model_name: str,
) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, list) and all(isinstance(item, str) for item in value):
        return list(value)
    raise ModelSpecError(f"Model `{section}.{model_name}` field `{field_name}` must be a string or list of strings.")


def _normalize_derived_terms(value: Any, *, section: str, model_name: str) -> dict[str, str]:
    if value is None:
        return {}
    if isinstance(value, Mapping):
        normalized = {str(key): str(expr) for key, expr in value.items()}
        if not all(normalized):
            raise ModelSpecError(f"Model `{section}.{model_name}` derived terms must have non-empty names.")
        return normalized
    if isinstance(value, list):
        normalized: dict[str, str] = {}
        for item in value:
            if not isinstance(item, Mapping):
                raise ModelSpecError(
                    f"Model `{section}.{model_name}` derived terms must be mappings with `name` and `expr`."
                )
            name = item.get("name")
            expr = item.get("expr", item.get("expression"))
            if not isinstance(name, str) or not isinstance(expr, str):
                raise ModelSpecError(
                    f"Model `{section}.{model_name}` derived terms must define string `name` and `expr` fields."
                )
            normalized[name] = expr
        return normalized
    raise ModelSpecError(
        f"Model `{section}.{model_name}` field `derived` must be a mapping or list of term definitions."
    )


def _coerce_model_spec(model_spec: ModelSpec | Mapping[str, Any]) -> ModelSpec:
    if isinstance(model_spec, ModelSpec):
        return model_spec
    return _build_model_spec(section="model", model_name=str(model_spec.get("name", "anonymous")), model_config=model_spec)


def _extract_variable_names(expression: str) -> list[str]:
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError:
        return []
    return sorted({node.id for node in ast.walk(tree) if isinstance(node, ast.Name)})
