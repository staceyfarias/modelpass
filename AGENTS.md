# modelpass agent guidance

Persistent instructions for coding agents working in this repository (the `modelpass` library, formerly `subpass`).

## Read first

- `README.md` — what modelpass is, what each runtime does, and how it is installed
  and run.
- `docs/api-and-runtimes.md` — the public contract: every entry point's promises and
  refusals, the capability matrix, and the dated verification record behind it
  (§2.2a). A capability claim that is not in there is not a claim this project
  makes.
- `CHANGELOG.md` — what each release changed, with the contract names it touched.
- `tests/` — the suite is the specification in practice. Each file's docstring says
  what it is holding and why, and a behaviour with no test here is not yet a
  behaviour of this library.

Decisions are referenced by number — the D-numbers in comments — and the record
they are numbered in is not published. You do not need it: everything those
decisions bind is either in the documents above or in a comment beside the code
it governs. Work from the contract and the tests, and say what you assumed.

## Rules

- User-visible message wording, `event.type` strings, error class names, and `pip install` hints are consumer contract. Do not change them outside a ticket that says so.
- Capability cells move only with recorded evidence: a test plus the SDK surface read, each with a dated note.
- Never run a live call that spends allowance; live tests go under the `live` marker and the owner runs them.
- Core keeps zero runtime dependencies; vendor SDKs are extras.
- Run `python -m pytest -q` and `ruff check src tests` before every commit.

## Commits

Commit completed work as you go: each finished ticket, fix, or documentation change gets its own commit on the current branch, with a message in the repository's style that says what changed and why. An uncommitted tree is the larger risk, so nothing is gained by holding finished changes back. Pushing remains the maintainer's call: do not push unless asked.
