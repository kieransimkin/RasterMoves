---
name: use-and-improve-rastermoves
description: Use RasterMoves to upscale, compare, trace, or refine still images, then turn evidence from the run into a tested, documented improvement to the RasterMoves repository. Apply to RasterMoves CLI or Python API work and RasterMoves workflow improvements; do not use for ordinary resizing, video upscaling, or unrelated image-generation tools.
---

# Use and Improve RasterMoves

Use the current repository as the source of truth. Preserve its safety, provenance,
licensing, image-handling, and refinement contracts rather than restating or weakening
them. A RasterMoves use is complete only when the image result is verified and a
small, evidence-backed workflow improvement has been committed to this repository.

## Establish the run

- Read `README.md` and only the task-relevant guide under `docs/`. Inspect `git status`,
  the current branch, recent relevant commits, and the applicable tests before changing
  anything. Preserve unrelated work and never discard a dirty tree.
- Identify the canonical input, intended output dimensions and format, content type,
  alpha/profile requirements, acceptable invention of detail, device/runtime constraints,
  and whether a model licence is suitable for the intended use.
- Preserve the input. Use a new output or run directory unless the user explicitly
  authorises `--overwrite`. Record input/output paths, hashes and dimensions in the
  run evidence or retained task report.
- Run `rastermoves doctor` before live inference when the environment is not already
  verified. Use `models`, `info`, and `--dry-run` to inspect model, licence, output,
  and download implications before a large or unfamiliar run.
- Prefer the smallest run that answers the question. Use one suitable model first;
  use `--all-models` only when comparison is the task. Reuse verified cache and resume
  artifacts rather than repeating downloads or inference without need.

## Produce and verify the image

- Use `--report` for material outputs. Use `--trace` when performance, memory, backend
  choice, downloading, tiling, or a failure is part of the question.
- Treat neural upscaling as plausible reconstruction, not recovery of original facts.
  Treat ControlNet refinement as generative synthesis. Use a fixed seed and protect
  faces, lettering, logos, line art, or other identity-critical areas when refinement
  could alter them.
- Check the saved file independently: dimensions, format, alpha, orientation, colour,
  report/provenance sidecar, and absence of accidental input replacement. Inspect the
  full image and representative 100% crops for seams, halos, ringing, invented text,
  distorted features, texture repetition, and over-sharpening.
- Compare like-for-like outputs. Do not call one model “best” without naming the visual
  criterion and the evidence. Metrics against the source measure consistency, not the
  truth of newly generated detail.
- For detailed visual QA and screenshot decisions, read
  [references/evaluation-checklist.md](references/evaluation-checklist.md).

## Improve the next run

During every use, keep a short friction log: confusing choice, repeated manual step,
missing guardrail, poor diagnostic, unsafe default, documentation gap, difficult
comparison, or avoidable performance cost. Distinguish observed evidence from a
hypothesis.

Before fixing an error, capture its exact text and environment, check existing docs,
tests and issues, then research matching current reports when needed. Prefer official
or maintainer evidence. Choose the smallest reusable improvement supported by the run:
code, test, diagnostic, example, documentation, or a narrow workflow helper. Do not
fabricate a change or broaden the product merely to create a commit.

Implement the improvement in the RasterMoves repository:

1. Add or update a meaningful regression test when behaviour changes.
2. Run the narrow test first, then the largest relevant local suite. Report optional
   runtimes, network checks, or pretrained models that were not exercised.
3. Update `README.md` in the same patch with a concise user-facing example. Document
   the problem observed, the improved command or API, expected output, and relevant
   limits. Update the specialised guide and `CHANGELOG.md` as well when scope warrants.
4. Add screenshots only when they materially demonstrate a visual result, comparison,
   UI, or diagnostic that prose cannot. Use rights-cleared, non-sensitive material;
   include useful alt text, a caption, model/settings, and focused crops. Do not add
   decorative or redundant screenshots.
5. Review the diff for secrets, caches, downloaded weights, large generated files,
   private images, and unrelated edits. Never commit them.

If no worthwhile code change emerges, improve a verified example, test, diagnostic,
or workflow explanation grounded in the run. If even that would be churn, stop and
explain why the mandatory improvement commit cannot be made rather than creating an
empty or misleading patch.

## Commit and verify GitHub

- Re-run `git status` and the relevant validation immediately before committing.
- Make one focused commit whose message describes the reusable improvement, not merely
  the image that triggered it. The image output itself need not enter Git.
- Push the focused commit to the configured GitHub remote when the user's request or
  standing project instruction authorises repository updates. Verify the remote branch
  resolves to the new commit. A local commit or successful push request is not proof of
  remote persistence.
- If authentication, branch protection, network access, a dirty overlapping change, or
  a failing unrelated test prevents completion, retain the local patch and report the
  exact narrow blocker. Do not rewrite history, bypass checks, or weaken tests.

Finish with the image result, model/settings and evidence location; the improvement;
tests run and limits; README/example/screenshot changes; commit SHA; remote verification;
and a brief blogworthiness assessment under the parent project rules.

## Potential problems

### Restricted command environment redirects Git to a dead local proxy

- **Symptom (3 October 2026):** a read-only `git ls-remote` failed with
  `Failed to connect to 127.0.0.1 port 9: Connection refused` even though the remote
  URL named GitHub.
- **Evidence:** the same exact command succeeded outside the restricted command
  environment and returned the repository HEAD. A materially similar 2026 report was
  found at <https://www.reddit.com/r/cursor/comments/1qxwztr/latest_update_broke_git/>;
  it corroborates the symptom but does not establish this environment's cause.
- **Corrective action that succeeded:** use one narrowly approved unsandboxed Git
  network check, then perform only the required clone/fetch/push operations. Do not
  change credentials merely because the sandbox-local proxy refused the connection.
- **Verification:** compare the expected remote branch SHA with `git ls-remote` after
  pushing.
- **Limit:** this workaround applies only when the restricted command shows the dead
  `127.0.0.1:9` proxy symptom and the equivalent approved command succeeds. Other Git
  errors require their own evidence and diagnosis.

### Pytest cannot access its default or workspace temporary directory

- **Symptom (3 October 2026):** tests using `tmp_path` failed during setup with
  `PermissionError: [WinError 5] Access is denied`; using a base below the mapped
  workspace failed in the same way.
- **Evidence:** pytest's temporary-directory documentation, checked 3 October 2026,
  explains the default `{temproot}/pytest-of-{user}` layout and the `--basetemp`
  override: <https://docs.pytest.org/en/stable/how-to/tmp_path.html#temporary-directory-location-and-retention>.
- **Corrective action that succeeded:** use a new, explicit, task-only directory below
  the accessible Windows user temp folder with `--basetemp`, and disable cache writes
  with `-p no:cacheprovider` when the repository cache is also restricted.
- **Verification:** the suite progressed from 216 setup errors to 345 passes, five
  skips, and two independent optional-runtime dependency failures.
- **Limit:** pytest clears the supplied `--basetemp` directory. Verify the exact path
  is new and dedicated to that run; never point it at a workspace or shared folder.

### Skill validator inherits a legacy Windows text encoding

- **Symptom (3 October 2026):** `quick_validate.py` raised a `UnicodeDecodeError` in
  `Path.read_text()` while reading a UTF-8 `SKILL.md`.
- **Evidence:** Python's command-line documentation, checked 3 October 2026, states
  that `PYTHONUTF8=1` enables UTF-8 mode:
  <https://docs.python.org/3/using/cmdline.html#envvar-PYTHONUTF8>.
- **Corrective action that succeeded:** set `PYTHONUTF8=1` for the validator process
  and run the normal skill validator unchanged.
- **Verification:** the validator returned `Skill is valid!`.
- **Limit:** use this only for an encoding failure. It does not waive validation errors
  in the skill itself or justify rewriting user content to ASCII.

### uv cannot initialise its default Windows cache

- **Symptom (3 October 2026):** `uv venv` failed with
  `Failed to initialize cache at C:\Users\Kieran\AppData\Local\uv\cache` followed by
  `Access is denied`.
- **Evidence:** Astral's cache documentation, checked 3 October 2026, states that
  `--cache-dir`, `UV_CACHE_DIR`, or project configuration overrides uv's platform
  default: <https://docs.astral.sh/uv/concepts/cache/#cache-directory>.
- **Corrective action that succeeded:** create a dedicated cache beside the isolated
  RasterMoves environment and pass its exact path with `--cache-dir` to every `uv venv`
  and `uv pip` command.
- **Verification:** uv created the CPython 3.13.14 environment and installed all 67
  compatible packages from that cache.
- **Limit:** keep the cache and environment on the same filesystem when practical for
  fast linking. Do not clear or edit uv's cache directly while another uv command runs.

### User-wide Diffusers packages do not match RasterMoves' pinned runtime

- **Symptom (3 October 2026):** the optional refinement runtime tests failed while
  importing Diffusers 0.40.0 with `cannot import name 'get_cached_repo_tree' from
  'huggingface_hub'`.
- **Cause supported by current evidence:** the user-wide environment had Diffusers
  0.40.0 while RasterMoves 0.3.0 declares `diffusers==0.35.2`; it was not the project's
  declared refinement environment. A matching public report described a version
  mismatch, but the local package metadata and isolated rerun provide the stronger
  evidence.
- **Corrective action that succeeded:** leave user-wide packages untouched and install
  `.[torch,onnx-gpu,refine,trace,gdrive,extra-arches,dev]` in a dedicated Python 3.13
  environment after the official CUDA PyTorch build.
- **Verification:** `pip check` found all 67 packages compatible, both SD1.5 and SDXL
  tiny-runtime tests passed, and the complete suite reported 347 passed and five skipped.
- **Limit:** this does not verify pretrained multi-gigabyte model downloads or visual
  quality. Keep those live tests opt-in and assess representative images separately.
