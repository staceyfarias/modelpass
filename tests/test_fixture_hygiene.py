"""The committed live fixtures carry vendor wire shapes, not one machine's life.

These files are real recordings. A ``codex app-server`` thread listing answers
with **every** thread on the account -- each one's first prompt, its auto-titled
name, its working directory, its git origin and branch, and the path to its
rollout file under the operator's home. Recorded verbatim and committed, that is
the operator's private work, published, in a package anyone can ``pip download``.

The 2026-08-31 capture was committed that way and sanitized on 2026-09-15. This
test is what keeps the next capture from arriving the same way. It fails on the
*shapes* a real machine leaves behind rather than on the particular strings that
were scrubbed, so it still bites when the next operator has a different name, a
different repository and a different drive letter.

It checks decoded values, not raw file text: a path inside a recorded shell
command is escaped twice over on the wire, and a rule written against the raw
bytes reads one of those and misses the other.

What it cannot check is content it has no way to recognize -- a prompt preview
is just a sentence. A reviewer still has to read a new capture before committing
it. This catches only what is mechanical.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures"

#: Shapes that mean "this came off somebody's actual computer", matched against
#: decoded string values.
LEAKS: tuple[tuple[str, re.Pattern[str]], ...] = (
    # A Windows user profile, any user.
    ("a Windows user profile path", re.compile(r"[A-Za-z]:\\Users\\", re.I)),
    # A POSIX home that is not the neutral stand-in these fixtures use.
    # Deliberately unanchored: a home directory turns up *inside* a recorded
    # shell command ("cd /home/alice/project && ...") at least as often as it
    # turns up as a whole field, and this module's docstring claims to handle
    # exactly that case.
    ("a POSIX home directory", re.compile(r"/(?:home|Users)/(?!dev/)[^/]+/")),
    # Any other drive-letter path except the two system roots. A path under
    # ``\Windows\`` or ``\Program Files`` is the same on every Windows box and
    # says nothing about whose box it is -- the tools fixture records a blocked
    # ``powershell.exe`` invocation, which is the payload it exists to prove.
    (
        "a drive-letter path outside the system roots",
        re.compile(r"[A-Za-z]:\\(?!Windows\\|Program Files)", re.I),
    ),
    # A real remote. Fixtures point at example.invalid, which cannot resolve.
    ("a real git remote", re.compile(r"(?:https://|git@)(?:github|gitlab|bitbucket)\.")),
)

#: Fields whose value identifies the workstation itself rather than the work on
#: it, mapped to the placeholder each must hold.
IDENTIFYING_FIELDS = {
    "installationId": "00000000-0000-0000-0000-000000000000",
    "serverName": "workstation",
    # What the operator pays the vendor. It is not a wire *shape* -- the field
    # is the shape, and any string exercises it -- so a recording that keeps
    # the real tier publishes a fact about the operator's account for nothing.
    "planType": "example-plan",
}

#: A branch name carries work-in-progress subject matter as reliably as a prompt
#: preview does. Fixtures use ``main``.
ALLOWED_BRANCHES = {"main", "master", None, ""}


def fixture_files() -> list[Path]:
    return sorted(
        p for p in FIXTURES.rglob("*") if p.suffix in {".json", ".jsonl"} and p.is_file()
    )


def documents(path: Path) -> list[object]:
    """Every JSON document in a fixture: one per line, or one per file."""
    text = path.read_text(encoding="utf-8")
    try:
        return [json.loads(text)]
    except json.JSONDecodeError:
        return [json.loads(line) for line in text.splitlines() if line.strip()]


def strings(node: object, key: str | None = None):
    """Every string in a document, paired with the field it was recorded under.

    A string inside a **list** is yielded under the field the list belongs to.
    It used to be skipped: the walk recursed into lists but only ever yielded
    from dict entries, so a workspace root recorded as a one-element list --
    a real home directory, one level of nesting away -- passed every rule in
    this file.
    """
    if isinstance(node, dict):
        for name, value in node.items():
            yield from strings(value, name)
    elif isinstance(node, list):
        for value in node:
            yield from strings(value, key)
    elif isinstance(node, str):
        yield key, node


def test_there_are_fixtures_to_check():
    """A rename that empties the glob must fail loudly, not pass vacuously."""
    assert len(fixture_files()) >= 7


def findings(document: object) -> list[str]:
    """Everything in one document that says whose machine it came off.

    A function rather than a loop inside the test, so the rules can be run
    against a document written here -- the shapes that got through before are
    worth holding directly, and a fixture cannot be the only witness to a rule
    that is supposed to catch the next fixture.
    """
    found = []
    for key, value in strings(document):
        # A path inside a recorded *shell command* is escaped a second time
        # on top of JSON's own escaping, so it decodes to ``C:\\Windows``
        # where a plain path field decodes to ``C:\Windows``. Collapse runs
        # of backslashes so one rule covers both depths.
        flattened = re.sub(r"\\{2,}", "\\\\", value)
        for what, pattern in LEAKS:
            if pattern.search(flattened) is not None:
                found.append(f"field {key!r} contains {what}: {value[:120]!r}")
        expected = IDENTIFYING_FIELDS.get(key)
        if expected is not None and value != expected:
            found.append(f"field {key!r} is {value!r}; fixtures carry {expected!r}")
    return found


@pytest.mark.parametrize("path", fixture_files(), ids=lambda p: p.name)
def test_a_fixture_names_no_real_machine(path: Path):
    where = path.relative_to(FIXTURES.parent)
    for document in documents(path):
        found = findings(document)
        assert not found, (
            f"{where}: {found[0]}. Sanitize the capture before committing it "
            "-- see this module's docstring."
        )


# --- the rules, against the shapes that got past them ----------------------------


def test_a_home_directory_inside_a_list_is_found():
    """The walk recursed into lists and yielded from dicts only, so a string
    that *was* a list element -- a workspace root, say -- was never read."""
    document = {"params": {"runtimeWorkspaceRoots": ["C:\\Users\\someone\\code"]}}
    found = findings(document)
    assert found and "runtimeWorkspaceRoots" in found[0]


def test_a_home_directory_inside_a_recorded_command_is_found():
    """The POSIX rule was anchored at the start of the value, and a home
    directory inside a shell command is never at the start of one -- which is
    the case this module's docstring says it handles."""
    assert findings({"command": "cd /home/alice/project && pytest -q"})


def test_an_identifying_field_is_checked_inside_a_list_too():
    assert findings({"accounts": [{"planType": "plus"}]})


def test_the_stand_ins_these_fixtures_use_are_not_findings():
    """The exemptions, so no rule above can be satisfied by flagging everything."""
    document = {
        "cwd": "/home/dev/project",
        "command": "powershell.exe -Command dir C:\\Windows\\System32",
        "origin": "https://example.invalid/acme/widgets.git",
        "installationId": "00000000-0000-0000-0000-000000000000",
        "serverName": "workstation",
        "planType": "example-plan",
        "roots": ["/home/dev/project", "/workspace"],
    }
    assert findings(document) == []


@pytest.mark.parametrize("path", fixture_files(), ids=lambda p: p.name)
def test_a_fixture_names_no_real_branch(path: Path):
    """``gitInfo.branch`` is a sentence about unshipped work. Keep it neutral."""
    for document in documents(path):
        for branch in _branches(document):
            assert branch in ALLOWED_BRANCHES, (
                f"{path.name} records the branch {branch!r}. Fixtures use 'main'."
            )


def _branches(node: object):
    if isinstance(node, dict):
        git_info = node.get("gitInfo")
        if isinstance(git_info, dict) and "branch" in git_info:
            yield git_info["branch"]
        for value in node.values():
            yield from _branches(value)
    elif isinstance(node, list):
        for value in node:
            yield from _branches(value)
