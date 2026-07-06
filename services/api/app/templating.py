"""Sandboxed prompt-template rendering.

Templates use explicit ``str.format`` placeholders (``{question}``) rather than a full
template language. The field grammar is deliberately restricted to bare identifiers:
attribute/index access (``{obj.attr}``, ``{row[0]}``) and positional fields are rejected at
registration time, so rendering can never traverse into object internals — the whole reason
the roadmap calls for a sandboxed engine. Variables are auto-extracted from the template and
rendering fails with a clear error when the supplied variables are missing or extra.
"""

from __future__ import annotations

from string import Formatter
from typing import Any

_formatter = Formatter()


class TemplateError(ValueError):
    """A template is malformed or its variables do not match the supplied values."""


def extract_variables(template: str) -> list[str]:
    """Return the template's variable names in first-use order.

    Raises :class:`TemplateError` for unbalanced braces, positional fields, attribute/index
    access, non-identifier names, or nested fields inside a format spec.
    """
    try:
        fields = list(_formatter.parse(template))
    except ValueError as exc:
        raise TemplateError(f"invalid template: {exc}") from exc

    variables: list[str] = []
    for _, field_name, format_spec, _ in fields:
        if field_name is None:  # literal text (or escaped braces)
            continue
        if field_name == "":
            raise TemplateError("positional fields ('{}') are not allowed; name every variable")
        if not field_name.isidentifier():
            raise TemplateError(
                f"invalid variable {{{field_name}}}: only bare identifiers are allowed "
                "(no attribute or index access)"
            )
        if format_spec and "{" in format_spec:
            raise TemplateError(f"nested fields in format specs are not allowed: {{{field_name}}}")
        if field_name not in variables:
            variables.append(field_name)
    return variables


def render(template: str, variables: dict[str, Any]) -> str:
    """Render ``template`` with exactly the variables it declares.

    Raises :class:`TemplateError` naming the missing and/or extra variables.
    """
    declared = extract_variables(template)
    missing = [name for name in declared if name not in variables]
    extra = [name for name in variables if name not in declared]
    if missing or extra:
        problems = []
        if missing:
            problems.append(f"missing: {', '.join(missing)}")
        if extra:
            problems.append(f"extra: {', '.join(extra)}")
        raise TemplateError(f"template variables do not match ({'; '.join(problems)})")
    return template.format(**variables)
