"""Extract exact Python definitions for an MDX source excerpt (stdin JSON)."""

import ast
import json
import re
import sys
import textwrap


def extract(source, symbols):
    lines = source.splitlines()
    nodes = []

    def visit(body, prefix=""):
        for node in body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                name = prefix + node.name
                nodes.append((name, node))
                visit(node.body, name + ".")

    visit(ast.parse(source).body)
    excerpts = []
    for name in symbols:
        candidates = [node for qualified, node in nodes if qualified == name]
        if len(candidates) != 1:
            raise ValueError(f"Expected one definition for {name!r}; found {len(candidates)}")
        node = candidates[0]
        start = min([node.lineno, *[decorator.lineno for decorator in node.decorator_list]])
        excerpts.append(
            {
                "code": textwrap.dedent("\n".join(lines[start - 1 : node.end_lineno])),
                "start": start,
                "end": node.end_lineno,
            }
        )
    return excerpts


def extract_region(source, name):
    lines = source.splitlines()
    markers = [
        (index, re.fullmatch(r"\s*(?:#|//)\s*(region|endregion)(?:\s+(.*?))?\s*", line))
        for index, line in enumerate(lines)
    ]
    starts = [
        index for index, marker in markers if marker and marker[1] == "region" and marker[2] == name
    ]
    if len(starts) != 1:
        raise ValueError(f"Expected one region {name!r}; found {len(starts)}")
    start = starts[0]
    stack = [name]
    for index, marker in markers:
        if index <= start or marker is None:
            continue
        kind, label = marker[1], marker[2]
        if kind == "region":
            stack.append(label)
        else:
            if label and label != stack[-1]:
                raise ValueError(f"Region closing {label!r} does not match {stack[-1]!r}")
            stack.pop()
            if not stack:
                code = textwrap.dedent("\n".join(lines[start + 1 : index]))
                if not code.strip():
                    raise ValueError(f"Empty region {name!r}")
                return [{"code": code, "start": start + 2, "end": index}]
    raise ValueError(f"Unclosed region {name!r}")


if __name__ == "__main__":
    if sys.argv[1:] == ["--check"]:
        sample = 'class Model:\n    @staticmethod\n    def forward(\n        x,\n    ):\n        return x + 1\n\nvalue = "not part of the method"\n'
        result = extract(sample, ["Model.forward"])[0]
        assert result["start"] == 2 and result["end"] == 6
        assert result["code"].startswith("@staticmethod\ndef forward(")
        assert "not part" not in result["code"]
        for invalid, source in [
            ("missing", sample),
            ("same", "def same(): pass\ndef same(): pass"),
        ]:
            try:
                extract(source, [invalid])
            except ValueError:
                pass
            else:
                raise AssertionError(f"Expected missing/ambiguous symbol to fail: {invalid}")
        assert (
            extract_region("// region loop\n  launch();\n// endregion loop", "loop")[0]["code"]
            == "launch();"
        )
        assert (
            extract_region(
                "# region outer\n# region inner\nx=1\n# endregion inner\n# endregion", "outer"
            )[0]["end"]
            == 4
        )
        for malformed in ["# region bad\nx=1", "# region bad\nx=1\n# endregion wrong"]:
            try:
                extract_region(malformed, "bad")
            except ValueError:
                pass
            else:
                raise AssertionError("Malformed region must fail")
        print(
            "Source excerpts: decorators, multiline signatures, boundaries and ambiguity verified"
        )
    else:
        request = json.load(sys.stdin)
        result = (
            extract_region(request["source"], request["region"])
            if request.get("region")
            else extract(request["source"], request["symbols"])
        )
        json.dump(result, sys.stdout)
