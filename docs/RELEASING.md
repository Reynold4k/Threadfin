# Releasing Threadfin

## Every release

1. Update the version in `pyproject.toml`, `threadfin/__init__.py` and `CITATION.cff` (with `date-released`), add
   the entry to `CHANGELOG.md` and run
   `python docs/make_api_reference.py` (the function reference carries the version).
2. Push to `main` and wait for the tests (Linux with Python 3.10–3.14 and the oldest allowed dependencies, macOS,
   Windows).
3. Create the GitHub release `vX.Y.Z` from `main`, with the changelog entry as its notes.

Publishing the release starts [`.github/workflows/publish.yml`](../.github/workflows/publish.yml), which builds
the sdist and the wheel, checks them, installs the wheel and runs the simulated tutorial, and uploads both to PyPI.
No password or token is stored: PyPI trusts this repository's workflow (trusted publishing).

## One-time setup on PyPI (account owner, about two minutes)

1. Sign in at <https://pypi.org> (create the account if needed, with two-factor authentication).
2. Open *Your account → Publishing → Add a new pending publisher → GitHub* and enter
   - PyPI project name: `threadfin`
   - Owner: `Reynold4k`
   - Repository name: `Threadfin`
   - Workflow name: `publish.yml`
   - Environment name: `pypi`
3. Publish the next GitHub release (or run the workflow by hand from the *Actions* tab). The first upload creates
   the `threadfin` project on PyPI; from then on `pip install threadfin` installs the latest release.
