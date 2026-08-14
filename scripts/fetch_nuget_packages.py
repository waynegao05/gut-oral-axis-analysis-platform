from __future__ import annotations

import argparse
import base64
from dataclasses import dataclass
from hashlib import sha512
import io
import json
from pathlib import Path
import re
import xml.etree.ElementTree as ET
import zipfile

import requests


NUGET_FLAT_CONTAINER = "https://api.nuget.org/v3-flatcontainer"
PACKAGE_PATTERN = re.compile(r"^[A-Za-z0-9_.-]+$")
NUGET_VERSION_PATTERN = re.compile(
    r"^(?P<core>\d+(?:\.\d+){0,3})(?:-(?P<prerelease>[0-9A-Za-z.-]+))?(?:\+[0-9A-Za-z.-]+)?$"
)


@dataclass(frozen=True, order=True)
class Package:
    package_id: str
    version: str
    minimum_inclusive: bool = True
    maximum_version: str | None = None
    maximum_inclusive: bool = False

    @property
    def normalized_id(self) -> str:
        return self.package_id.lower()


def dependency_minimum_version(version_range: str) -> str:
    value = version_range.strip()
    if not value:
        raise ValueError("NuGet dependency does not declare a version.")
    if value[0] in "[(":
        body = value[1:-1] if value[-1] in ")]" else value[1:]
        if "," not in body:
            minimum = body.strip()
        else:
            minimum = body.split(",", maxsplit=1)[0].strip()
        if not minimum:
            raise ValueError(f"NuGet dependency has no minimum version: {value}")
        return minimum
    return value


def parse_dependency(package_id: str, version_range: str) -> Package:
    value = version_range.strip()
    minimum = dependency_minimum_version(value)
    if value[0] not in "[(":
        return Package(package_id, minimum)

    body = value[1:-1] if value[-1] in ")]" else value[1:]
    if "," not in body:
        return Package(
            package_id,
            minimum,
            minimum_inclusive=value[0] == "[",
            maximum_version=minimum,
            maximum_inclusive=value[-1] == "]",
        )
    _, maximum = (part.strip() for part in body.split(",", maxsplit=1))
    return Package(
        package_id,
        minimum,
        minimum_inclusive=value[0] == "[",
        maximum_version=maximum or None,
        maximum_inclusive=value[-1] == "]",
    )


def parse_dependencies(package_bytes: bytes) -> list[Package]:
    with zipfile.ZipFile(io.BytesIO(package_bytes)) as archive:
        nuspec_names = [name for name in archive.namelist() if name.lower().endswith(".nuspec")]
        if len(nuspec_names) != 1:
            raise ValueError("NuGet package must contain exactly one .nuspec file.")
        root = ET.fromstring(archive.read(nuspec_names[0]))

    dependencies: list[Package] = []
    for element in root.iter():
        if element.tag.rsplit("}", maxsplit=1)[-1] != "dependency":
            continue
        package_id = element.attrib.get("id", "").strip()
        if package_id:
            dependencies.append(parse_dependency(package_id, element.attrib.get("version", "")))
    return dependencies


def registration_url(package: Package, version: str) -> str:
    return (
        "https://api.nuget.org/v3/registration5-gz-semver2/"
        f"{package.normalized_id}/{version.lower()}.json"
    )


def nuget_version_key(value: str) -> tuple[tuple[int, ...], int, tuple[tuple[int, int | str], ...]]:
    match = NUGET_VERSION_PATTERN.fullmatch(value)
    if match is None:
        raise ValueError(f"Unsupported NuGet version: {value}")
    core = tuple(int(part) for part in match.group("core").split("."))
    core = core + (0,) * (4 - len(core))
    prerelease = match.group("prerelease")
    if prerelease is None:
        return core, 1, ()
    identifiers: list[tuple[int, int | str]] = []
    for identifier in prerelease.split("."):
        if identifier.isdigit():
            identifiers.append((0, int(identifier)))
        else:
            identifiers.append((1, identifier.lower()))
    return core, 0, tuple(identifiers)


def nuget_is_prerelease(value: str) -> bool:
    match = NUGET_VERSION_PATTERN.fullmatch(value)
    if match is None:
        raise ValueError(f"Unsupported NuGet version: {value}")
    return match.group("prerelease") is not None


def version_satisfies(candidate: str, requirement: Package) -> bool:
    value = nuget_version_key(candidate)
    minimum = nuget_version_key(requirement.version)
    if value < minimum or (value == minimum and not requirement.minimum_inclusive):
        return False
    if requirement.maximum_version is None:
        return True
    maximum = nuget_version_key(requirement.maximum_version)
    return value < maximum or (value == maximum and requirement.maximum_inclusive)


def resolve_package(session: requests.Session, package: Package) -> Package:
    if package.minimum_inclusive:
        direct_response = session.get(registration_url(package, package.version), timeout=30)
        if direct_response.ok:
            return Package(package.package_id, package.version)
        if direct_response.status_code != 404:
            direct_response.raise_for_status()

    index_url = f"{NUGET_FLAT_CONTAINER}/{package.normalized_id}/index.json"
    index_response = session.get(index_url, timeout=30)
    index_response.raise_for_status()
    candidates = [
        version
        for version in index_response.json().get("versions", [])
        if version_satisfies(version, package)
    ]
    if not nuget_is_prerelease(package.version):
        stable_candidates = [version for version in candidates if not nuget_is_prerelease(version)]
        if stable_candidates:
            candidates = stable_candidates
    if not candidates:
        raise RuntimeError(
            f"No published NuGet version satisfies {package.package_id} {package.version}."
        )
    resolved_version = min(candidates, key=nuget_version_key)
    return Package(package.package_id, resolved_version)


def fetch_package(
    session: requests.Session,
    package: Package,
    output_directory: Path,
) -> tuple[Package, bytes, str]:
    if not PACKAGE_PATTERN.fullmatch(package.package_id) or not PACKAGE_PATTERN.fullmatch(package.version):
        raise ValueError(f"Invalid NuGet package coordinate: {package}")
    package = resolve_package(session, package)
    registration_response = session.get(registration_url(package, package.version), timeout=30)
    registration_response.raise_for_status()
    registration = registration_response.json()
    catalog_reference = registration.get("catalogEntry")
    if not isinstance(catalog_reference, str):
        raise RuntimeError(f"NuGet registration has no catalog entry for {package}.")
    catalog_response = session.get(catalog_reference, timeout=30)
    catalog_response.raise_for_status()
    catalog = catalog_response.json()
    if catalog.get("packageHashAlgorithm") != "SHA512":
        raise RuntimeError(f"NuGet package does not publish a SHA-512 hash: {package}.")

    expected = base64.b64decode(str(catalog["packageHash"]))
    filename = f"{package.normalized_id}.{package.version.lower()}.nupkg"
    cached_path = output_directory / filename
    package_bytes = cached_path.read_bytes() if cached_path.is_file() else b""
    actual = sha512(package_bytes).digest() if package_bytes else b""
    if actual != expected:
        package_url = registration.get("packageContent")
        if not isinstance(package_url, str):
            raise RuntimeError(f"NuGet registration has no package URL for {package}.")
        package_response = session.get(package_url, timeout=120)
        package_response.raise_for_status()
        package_bytes = package_response.content
        actual = sha512(package_bytes).digest()
    if actual != expected:
        raise RuntimeError(f"SHA-512 mismatch for {package.package_id} {package.version}.")
    return package, package_bytes, base64.b64encode(actual).decode("ascii")


def fetch_with_dependencies(
    roots: list[Package],
    output_directory: Path,
) -> list[dict[str, str]]:
    output_directory.mkdir(parents=True, exist_ok=True)
    pending = list(roots)
    completed: dict[tuple[str, str], dict[str, str]] = {}
    session = requests.Session()
    session.headers["User-Agent"] = "GutOralAxisDesktopBuild/1.0"

    while pending:
        requested_package = pending.pop(0)
        package, package_bytes, digest = fetch_package(session, requested_package, output_directory)
        key = (package.normalized_id, package.version.lower())
        if key in completed:
            continue
        filename = f"{package.normalized_id}.{package.version.lower()}.nupkg"
        (output_directory / filename).write_bytes(package_bytes)
        completed[key] = {
            "id": package.package_id,
            "version": package.version,
            "sha512_base64": digest,
            "source": "api.nuget.org",
        }
        pending.extend(parse_dependencies(package_bytes))

    manifest = sorted(completed.values(), key=lambda item: (item["id"].lower(), item["version"]))
    (output_directory / "packages-manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return manifest


def parse_coordinate(value: str) -> Package:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Package must use ID=VERSION syntax.")
    package_id, version = value.split("=", maxsplit=1)
    if not PACKAGE_PATTERN.fullmatch(package_id) or not PACKAGE_PATTERN.fullmatch(version):
        raise argparse.ArgumentTypeError("Package ID or version contains invalid characters.")
    return Package(
        package_id,
        version,
        maximum_version=version,
        maximum_inclusive=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a SHA-512-verified local NuGet source.")
    parser.add_argument("packages", nargs="+", type=parse_coordinate)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    manifest = fetch_with_dependencies(args.packages, args.output_dir.resolve())
    print(f"Fetched {len(manifest)} verified NuGet packages to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
