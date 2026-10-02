import hashlib
import importlib.util
import io
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tarfile
import zipfile

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("release_tools", ROOT / "scripts/release_tools.py")
tools = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tools)


@pytest.fixture
def project(tmp_path):
    (tmp_path / "src/rastermoves").mkdir(parents=True)
    (tmp_path / "pyproject.toml").write_text('[project]\nname="rastermoves"\nversion="0.2.0"\n')
    (tmp_path / "src/rastermoves/__init__.py").write_text('__version__ = "0.2.0"\n')
    return tmp_path


def test_source_and_metadata_versions_agree():
    import rastermoves
    assert tools.version_from_source(ROOT) == rastermoves.__version__


def test_mismatched_versions_rejected(project):
    (project / "src/rastermoves/__init__.py").write_text('__version__ = "0.1.1"\n')
    with pytest.raises(ValueError, match="disagree"):
        tools.version_from_source(project)


@pytest.mark.parametrize("tag", ["main", "0.2.0", "v0.2.1", "v0.2.0+local", "v0.2.0;echo bad", "refs/tags/v0.2.0", ""])
def test_production_rejects_bad_or_mismatched_tags(project, tag):
    with pytest.raises(ValueError):
        tools.release_metadata(tag, "kieransimkin/RasterMoves", project, check_git=False)


@pytest.mark.parametrize("version,prerelease", [("0.2.0", "false"), ("0.3.0rc1", "true"),
                                                ("0.3.0a1", "true"), ("0.3.0b1", "true"),
                                                ("0.3.0.dev1", "true"), ("0.2.0.post1", "false")])
def test_canonical_release_tags_and_lowercase_image(project, version, prerelease):
    (project / "pyproject.toml").write_text(f'[project]\nname="rastermoves"\nversion="{version}"\n')
    (project / "src/rastermoves/__init__.py").write_text(f'__version__ = "{version}"\n')
    result = tools.release_metadata("v" + version, "kieransimkin/RasterMoves", project, check_git=False)
    assert result["image"] == "ghcr.io/kieransimkin/rastermoves"
    assert result["prerelease"] == prerelease and result["version"] == version


def test_rehearsal_must_explicitly_allow_no_tag(project):
    result = tools.release_metadata("", "kieransimkin/RasterMoves", project, check_git=False, allow_untagged=True)
    assert result["tag"] == "" and result["version"] == "0.2.0"
    with pytest.raises(ValueError):
        tools.release_metadata("main", "kieransimkin/RasterMoves", project, check_git=False, allow_untagged=True)


def test_check_git_requires_exact_tagged_commit(project):
    if not shutil.which("git"):
        pytest.skip("git is required")
    def git(*args):
        return subprocess.check_output(["git", *args], cwd=project, text=True).strip()
    git("init", "-q")
    git("config", "user.name", "Tests")
    git("config", "user.email", "tests@example.invalid")
    git("add", ".")
    git("commit", "-qm", "initial")
    git("tag", "v0.2.0")
    assert tools.release_metadata("v0.2.0", "kieransimkin/RasterMoves", project)["commit"] == git("rev-parse", "HEAD")
    (project / "new-file").write_text("new source")
    git("add", ".")
    git("commit", "-qm", "later")
    with pytest.raises(ValueError, match="not the requested release tag"):
        tools.release_metadata("v0.2.0", "kieransimkin/RasterMoves", project)


def make_dist(project):
    dist = project / "dist"
    dist.mkdir()
    wheel = dist / "rastermoves-0.2.0-py3-none-any.whl"
    with zipfile.ZipFile(wheel, "w") as z:
        z.writestr("rastermoves-0.2.0.dist-info/METADATA", "Name: rastermoves\nVersion: 0.2.0\n")
        for name in ("sweep.py", "py.typed", *(f"models/{i}.json" for i in range(8))):
            z.writestr("rastermoves/" + name, "")
    sdist = dist / "rastermoves-0.2.0.tar.gz"
    with tarfile.open(sdist, "w:gz") as tar:
        for name in ("PKG-INFO", "docs/RELEASING.md", "Dockerfile", "scripts/release_tools.py", ".github/workflows/release.yml"):
            data = b"Name: rastermoves\nVersion: 0.2.0\n" if name == "PKG-INFO" else b"test"
            info = tarfile.TarInfo("rastermoves-0.2.0/" + name)
            info.size, info.mtime = len(data), 123456
            tar.addfile(info, io.BytesIO(data))
    return dist, wheel, sdist


def test_dist_validation_and_reproducible_sdist(project):
    dist, wheel, sdist = make_dist(project)
    assert tools.validate_dist(dist, project) == [wheel, sdist]
    tools.normalize_sdist(sdist, 1700000000)
    first = hashlib.sha256(sdist.read_bytes()).hexdigest()
    tools.normalize_sdist(sdist, 1700000000)
    assert hashlib.sha256(sdist.read_bytes()).hexdigest() == first
    with tarfile.open(sdist) as tar:
        assert all(m.mtime == 1700000000 and m.uid == 0 for m in tar.getmembers())
    tools.validate_dist(dist, project)


def test_dist_rejects_multiple_wheels_and_wrong_metadata(project):
    dist, wheel, _ = make_dist(project)
    extra = dist / "other.whl"
    extra.write_bytes(wheel.read_bytes())
    with pytest.raises(ValueError, match="exactly one"):
        tools.validate_dist(dist, project)
    extra.unlink()
    with zipfile.ZipFile(wheel, "w") as z:
        z.writestr("rastermoves-0.2.0.dist-info/METADATA", "Name: rastermoves\nVersion: 9.9.9\n")
    with pytest.raises(ValueError, match="name/version"):
        tools.validate_dist(dist, project)


def workflow(name):
    # BaseLoader avoids YAML 1.1 interpreting the GitHub 'on' key as Boolean true.
    return yaml.load((ROOT / ".github/workflows" / name).read_text(), Loader=yaml.BaseLoader)


def test_publication_gate_and_target_permissions():
    release = workflow("release.yml")
    assert release["on"]["push"]["tags"] == ["v*"]
    assert "pull_request" not in release["on"] and "release" not in release["on"]
    assert release["concurrency"]["cancel-in-progress"] == "false"
    jobs = release["jobs"]
    assert "kieransimkin/RasterMoves" in jobs["prepare"]["if"]
    for name in ("pypi", "testpypi", "github-release", "ghcr"):
        assert set(jobs[name]["needs"]) == {"prepare", "checks"}
    assert jobs["pypi"]["environment"]["name"] == "pypi"
    assert jobs["testpypi"]["environment"]["name"] == "testpypi"
    assert jobs["pypi"]["permissions"] == {"id-token": "write"}
    assert jobs["ghcr"]["permissions"]["packages"] == "write"
    assert "DOCKERHUB_USERNAME" in jobs["dockerhub"]["if"]
    assert jobs["checks"]["with"]["ref"] == "${{ needs.prepare.outputs.commit }}"


def test_pypi_uses_only_built_artifacts_and_not_a_checkout():
    release = workflow("release.yml")
    for name in ("pypi", "testpypi"):
        steps = release["jobs"][name]["steps"]
        assert len(steps) == 2
        assert steps[0]["uses"].startswith("actions/download-artifact@")
        assert steps[0]["with"]["name"] == "python-dist"
        assert steps[1]["uses"].startswith("pypa/gh-action-pypi-publish@")
        assert steps[1]["with"]["skip-existing"] == "false"
        assert "password" not in steps[1].get("with", {})


@pytest.mark.parametrize("name", ["tests.yml", "release.yml"])
def test_workflow_dependencies_and_action_pins(name):
    data = workflow(name)
    jobs = data["jobs"]
    for job in jobs.values():
        needs = job.get("needs", [])
        needs = [needs] if isinstance(needs, str) else needs
        assert all(n in jobs for n in needs)
        for step in job.get("steps", []):
            if "uses" in step:
                assert re.fullmatch(r"[^@]+@[0-9a-f]{40}", step["uses"]), step["uses"]
            if step.get("uses", "").startswith("actions/checkout@"):
                assert step["with"]["persist-credentials"] == "false"


def test_shell_steps_parse_without_running_them():
    if sys.platform == "win32" or not shutil.which("bash"):
        pytest.skip("bash is not installed")
    for name in ("tests.yml", "release.yml"):
        for job in workflow(name)["jobs"].values():
            for step in job.get("steps", []):
                if "run" in step:
                    result = subprocess.run(["bash", "-n"], input=step["run"], text=True, capture_output=True)
                    assert result.returncode == 0, result.stderr


def test_container_build_uses_wheel_and_smokes_before_login_push():
    dockerfile = (ROOT / "Dockerfile").read_text()
    assert "COPY dist/*.whl" in dockerfile
    assert "USER 10001:10001" in dockerfile
    assert "download.pytorch.org/whl/cpu" in dockerfile
    jobs = workflow("release.yml")["jobs"]
    steps = jobs["ghcr"]["steps"]
    smoke = next(i for i, s in enumerate(steps) if "Smoke-test" in s.get("name", ""))
    push = next(i for i, s in enumerate(steps) if s.get("id") == "publish")
    assert smoke < push and "--network none" in steps[smoke]["run"]
    assert "imagetools create" in jobs["dockerhub"]["steps"][-1]["run"]
