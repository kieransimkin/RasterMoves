# Publishing RasterMoves

## Destinations and triggers

| Destination | Artifacts | Trigger/configuration |
|---|---|---|
| PyPI | `rastermoves` wheel and source distribution, with publish attestations | Production release; one-time Trusted Publisher configuration |
| GitHub Releases | The same tested wheel/sdist plus `SHA256SUMS` | Production release; repository workflow token |
| GHCR | `ghcr.io/kieransimkin/rastermoves` CPU container | Production release; repository workflow token |
| TestPyPI | Wheel/sdist rehearsal | Manual `target=testpypi`, or repository variable `PUBLISH_TESTPYPI=true` |
| Docker Hub | Mirror of the GHCR image, without rebuilding | Optional `DOCKERHUB_USERNAME` variable and `DOCKERHUB_TOKEN` secret |

Ordinary branch pushes and pull requests run tests only. Pushing a `v*` tag starts
`.github/workflows/release.yml`. A manual production run requires an existing matching
version tag. A manual TestPyPI rehearsal can omit the tag and use the selected branch.
Publication is guarded to `kieransimkin/RasterMoves`; forks must deliberately change
that guard, project URLs and publisher configuration.

A Python wheel belongs on PyPI. The container belongs on GHCR/Docker Hub; this project
does not pretend that GitHub Packages offers a native pip/PyPI registry, nor publish a
Python package as an unrelated npm package. Conda-forge/Homebrew are not automated here;
they require separately maintained recipes and acceptance workflows.

## One-time PyPI configuration

Sign in to PyPI and create a pending Trusted Publisher under **Publishing** if this
project is not registered yet. For a project you already own, use its **Manage →
Publishing** page. Use exactly:

| Field | Value |
|---|---|
| PyPI project name | `rastermoves` |
| GitHub owner | `kieransimkin` |
| Repository name | `RasterMoves` |
| Workflow filename | `release.yml` (not the directory path) |
| Environment name | `pypi` |

Do **not** create a `PYPI_API_TOKEN` secret: this workflow exchanges GitHub's OIDC identity
for short-lived publishing credentials. Its publishing job has `id-token: write` and
no source checkout or build step. Builds run separately without publishing authority.
The PyPI project name must actually be available or already owned by you; a pending
publisher is not a name reservation or proof of availability.

In GitHub **Settings → Environments**, create `pypi`. Restrict deployment to the `v*`
tag pattern and enable required review where appropriate for your account/plan. Protect
version tags against force-push/deletion and require reviewed changes to workflows.
Environment approval can make a release show **Waiting**; approve it in Actions.

For optional TestPyPI, independently register the same fields on TestPyPI with
**environment `testpypi`**. Create that GitHub environment and permit your rehearsal
branch (normally `main`) as well as version tags. PyPI and TestPyPI accounts/publisher
settings are separate. Do not enable `PUBLISH_TESTPYPI=true` until TestPyPI is configured.

## GHCR and optional Docker Hub

GHCR uses `GITHUB_TOKEN` with `packages: write`; no personal token is required. The
workflow lowercases the image name, avoiding the mixed-case `RasterMoves` repository
name problem. The OCI source label links the image to this repository. If a package
already exists, grant this repository Actions access to it. Check the new package's
visibility and make it **public** to permit anonymous pulls; repository visibility
does not prove that an existing container package is public.

The supplied image is **CPU-only, Linux AMD64** and includes PyTorch/Spandrel, ONNX
Runtime and the optional Google Drive downloader. It installs the already-tested
wheel rather than waiting for PyPI indexing. No pretrained checkpoints are embedded.
The image runs as UID/GID 10001 and uses `/work` and `/cache`. CUDA and multi-architecture
container variants are not included; native installs retain their existing GPU options.

To add Docker Hub, configure GitHub Actions repository settings:

- Variable `DOCKERHUB_USERNAME`: your login name.
- Secret `DOCKERHUB_TOKEN`: an access token permitted to push to the target repository.
- Optional variable `DOCKERHUB_REPOSITORY`: lowercase `namespace/rastermoves`; defaults
  to `DOCKERHUB_USERNAME/rastermoves`.

Without the username variable, the mirror job is skipped. With a username but no token,
it fails with an explicit setup error; this does not prevent PyPI, GHCR or release-asset
jobs from running. The mirror copies the published GHCR digest; it does not rebuild the
Python code. Image tags are `0.2.0` and `0.2.0-cpu` for that version; stable releases
also update `latest` and `cpu`. Prereleases never update these floating stable tags.
Use explicit versions/digests for reproducible deployment; republishing an older stable
tag can move the floating tags backwards and should be avoided.

## First release of this patch

The patch sets **both** `pyproject.toml` and `src/rastermoves/__init__.py` to `0.2.0`.
Apply it to the repository, commit, then push the branch. Do not tag the old commit.

```bash
git apply --check rastermoves-0.2.0-ci-all-models.patch
git apply rastermoves-0.2.0-ci-all-models.patch
python -m pip install -e ".[dev]"
python -m pytest -q
git add .github .dockerignore Dockerfile MANIFEST.in README.md CHANGELOG.md pyproject.toml scripts src tests docs
git commit -m "Add release publishing and all-model image comparisons"
git push origin main
```

After configuring the publishers, a TestPyPI-only rehearsal of the committed branch is
available without triggering production publishing:

```bash
gh workflow run release.yml --ref main -f target=testpypi
```

A successful rehearsal does **not** publish a production release or container. Once
ready, create the version tag at the tested commit:

```bash
git tag -a v0.2.0 -m "RasterMoves 0.2.0"
git push origin v0.2.0
```

This runs tests on the exact tag, not on a later `main`. The source and wheel version
must match the tag or publishing stops. Examples of allowed future tags include
`v0.2.1`, `v0.3.0rc1` and `v0.3.0.dev1`. Local versions with `+...` are rejected.
Update both declarations and `CHANGELOG.md` before a future tag. Do not move/reuse tags.

For a manual production run of an existing tag:

```bash
gh workflow run release.yml --ref v0.2.0 -f tag=v0.2.0 -f target=production
```

Using `--ref v0.2.0` also makes the workflow's environment/OIDC context a tag rather
than `main`; this matters with tag-only environment protections. This command is not
a substitute for **Re-run failed jobs** after a partial publication: versions cannot
be uploaded repeatedly to PyPI/TestPyPI.

## What is tested and what is published

CI covers core tests on Python 3.10–3.13 on Linux and Python 3.12 on Windows/macOS,
plus CPU optional-runtime tests. Its package job builds the wheel and sdist, validates
both versions and bundled data, normalizes source-archive timestamps, runs strict
Twine metadata checks and smoke-tests isolated wheel/sdist installations outside the
source checkout. The resulting `python-dist` artifact is the one used for publication.

The release container is built from that wheel and smoke-tested without network
access before it is pushed. Its exact built image is tagged/pushed to GHCR. This
validates installed runtimes and CLI/package data, **not every pretrained model**.
Network-dependent model smoke tests remain an explicit **Tests → Run workflow →
live_models** option. All-model mode still reports unsupported entries as failures.

Third-party Actions are pinned to commit SHAs, with Dependabot configured to propose
updates. The repository's Python runtime dependency ranges and Docker base tag are not
a complete dependency lock; builds made much later can select newer dependencies.
Pin a published image digest for exact runtime reproduction.

## Recovery and verification

The workflow summary lists the actual result of each destination. Publication across
independent registries is not transactional: one can succeed while another fails.
Fix the failing destination and choose **Re-run failed jobs**, preserving already
successful jobs and the tested build artifact. Artifacts are retained for 14 days.

Production and TestPyPI uploads deliberately do not use `skip-existing`: a duplicate
version must not conceal changed artifacts. Neither service normally permits reuse
of an uploaded filename/version after deletion. If package code or release tooling
needs changing, bump the version and publish a new tag instead of rewriting a release.

GitHub assets are uploaded while the release is a draft, then it is published. A rerun
will only accept an already published release when its assets match byte-for-byte;
it will not replace immutable/published assets. Creating/publishing a manual GitHub
release before this workflow uploads the assets can block that step. Prefer the
workflow's tag-driven draft→upload→publish sequence.

Common setup failures: `invalid-publisher` means the owner/repo/workflow/environment
fields or selected workflow ref do not match; GHCR permission errors require package
Actions access; name conflicts or duplicate PyPI versions are not fixed by retrying;
private GHCR packages cannot be pulled anonymously. Optional destinations marked
**skipped** were not published. Run these after the relevant jobs succeed:

```bash
python -m pip install --upgrade "rastermoves[all]==0.2.0"
rastermoves --version
docker pull ghcr.io/kieransimkin/rastermoves:0.2.0
docker run --rm ghcr.io/kieransimkin/rastermoves:0.2.0 --version
gh release view v0.2.0
```

Example Linux/macOS container invocation (Docker Desktop needs Linux AMD64 emulation
on non-AMD64 hosts):

```bash
mkdir -p comparison .rastermoves-cache
docker run --rm --user "$(id -u):$(id -g)" \
  -v "$PWD:/work" -v "$PWD/.rastermoves-cache:/cache" \
  ghcr.io/kieransimkin/rastermoves:0.2.0 \
  upscale /work/input.png --all-models --sync-models -o /work/comparison
```

Use paths/quoting appropriate to your shell. The output/cache bind mounts must be
writable by the chosen container user. The cache avoids downloading the same models
again across disposable container runs.

## Primary documentation used

Checked 2026-10-02:

- PyPA release workflow guide: https://packaging.python.org/guides/publishing-package-distribution-releases-using-github-actions-ci-cd-workflows/
- PyPI publisher setup: https://docs.pypi.org/trusted-publishers/adding-a-publisher/
- First project via pending publisher: https://docs.pypi.org/trusted-publishers/creating-a-project-through-oidc/
- OIDC usage/TestPyPI: https://docs.pypi.org/trusted-publishers/using-a-publisher/
- GitHub container publishing: https://docs.github.com/en/actions/tutorials/publish-packages/publish-docker-images
- GHCR authentication/access: https://docs.github.com/packages/working-with-a-github-packages-registry/working-with-the-container-registry
- GitHub release lifecycle: https://docs.github.com/en/repositories/releasing-projects-on-github/about-releases
