import gzip
import io
import tarfile
import zipfile

import pytest

from scripts.validation.check_release_privacy import artifact_payloads, check_payload


def test_compressed_review_paths_are_checked():
    # A complete nested path, including one embedded in a JSON string.
    payload = gzip.compress(b'{"source": "' + b'/home/' + b'private-owner/repo"}')
    assert check_payload("records.json.gz", payload) == ["personal-home-path"]
    assert check_payload("records.json.gz", b"invalid") == ["invalid-gzip"]
    assert check_payload("test.log", b"/tmp/pytest-of-" + b"private-owner/test") == ["personal-test-path"]


def test_public_urls_and_placeholders_are_allowed():
    assert check_payload("profile.yaml", b"https://vendor.example/home/product?id=1") == []
    assert check_payload("example.py", b"<home>/repo /home/user/repo /home/runner/work") == []
    assert check_payload("notes.md", b"https://example.org " + b"/Users/" + b"private-owner/repo") == ["personal-home-path"]


def test_private_files_and_credentials_are_rejected():
    assert check_payload("package/.aws/credentials", b"") == ["private-file"]
    assert check_payload("package/.env.production", b"") == ["private-file"]
    assert check_payload("module.py", b"ghp_" + b"a" * 36) == ["github-token"]
    assert check_payload("key.pem", b"-----BEGIN " + b"PRIVATE KEY-----") == ["private-key"]


def test_distribution_members_and_links_are_checked_without_extraction(tmp_path):
    wheel = tmp_path / "test.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("package/.env", "secret")
    assert list(artifact_payloads(wheel)) == [("package/.env", b"secret")]
    source = tmp_path / "test.tar.gz"
    with tarfile.open(source, "w:gz") as archive:
        entry = tarfile.TarInfo("package/file.txt")
        entry.size = 4
        archive.addfile(entry, io.BytesIO(b"text"))
        link = tarfile.TarInfo("package/link")
        link.type = tarfile.SYMTYPE
        link.linkname = "/Users/" + "private-owner/secret"
        archive.addfile(link)
    entries = list(artifact_payloads(source))
    assert entries[0] == ("package/file.txt", b"text")
    assert check_payload(*entries[1]) == ["personal-home-path"]
    assert not (tmp_path / "package").exists()


def test_source_archive_ownership_must_be_neutral(tmp_path):
    source = tmp_path / "test.tar.gz"
    with tarfile.open(source, "w:gz") as archive:
        entry = tarfile.TarInfo("package/file.txt")
        entry.uname = "private-owner"
        archive.addfile(entry, io.BytesIO(b""))
    with pytest.raises(ValueError, match="non-neutral archive ownership"):
        list(artifact_payloads(source))
