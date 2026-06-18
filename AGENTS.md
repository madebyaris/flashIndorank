# AGENTS.md

## Cursor Cloud specific instructions

As of this writing, this repository is an **empty starter**: the only tracked
file is `LICENSE` (plus this file). There is no application source code,
package manifest, lockfile, Dockerfile, Makefile, or setup script yet.

Intended product (from the GitHub repo description, not yet implemented):
**flashIndorank** — "a flashrank specialist for Indonesia language", i.e. an
Indonesian-language specialization built on top of
[FlashRank](https://github.com/PrithivirajDamodaran/FlashRank), a Python
re-ranking library. This strongly suggests the eventual stack will be Python.

Because nothing is implemented, there is currently:

- nothing to install (no dependency manifest),
- nothing to lint/test/build,
- no service to run, and therefore no end-to-end / "hello world" flow to exercise.

Once real code lands, update this section with how to install, lint, test,
build, and run the relevant service(s).

### Available runtimes on the VM

The base VM already provides (no install needed): Python 3.12, `pip`,
Node.js 22 (`npm`, `pnpm`), and Go 1.22. `uv` is not installed.

### Update script

The startup update script is intentionally a guarded no-op for the current
empty repo: it only installs dependencies if a recognized manifest
(`requirements.txt`, `pyproject.toml`, or `package.json`) is present. This keeps
future startups working both now (empty repo) and after the project is
scaffolded.
