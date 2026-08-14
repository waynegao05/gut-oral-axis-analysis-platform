from __future__ import annotations

from scripts.fetch_nuget_packages import (
    Package,
    dependency_minimum_version,
    nuget_version_key,
    parse_dependency,
    version_satisfies,
)


def test_exact_nuget_dependency_range() -> None:
    assert dependency_minimum_version("[2.3.1]") == "2.3.1"


def test_bounded_nuget_dependency_range() -> None:
    assert dependency_minimum_version("[2.3.0, 3.0.0)") == "2.3.0"


def test_unbounded_minimum_is_rejected() -> None:
    try:
        dependency_minimum_version("(, 3.0.0]")
    except ValueError as error:
        assert "no minimum version" in str(error)
    else:
        raise AssertionError("Expected an unbounded minimum to be rejected.")


def test_dependency_range_preserves_upper_bound() -> None:
    requirement = parse_dependency("Example.Package", "[2.1.1, 3.0.0)")
    assert version_satisfies("2.1.3", requirement)
    assert not version_satisfies("3.0.0", requirement)


def test_exact_range_rejects_other_versions() -> None:
    requirement = parse_dependency("Example.Package", "[2.3.1]")
    assert version_satisfies("2.3.1", requirement)
    assert not version_satisfies("2.3.2", requirement)


def test_open_lower_bound_excludes_minimum() -> None:
    requirement = Package(
        "Example.Package",
        "2.0.0",
        minimum_inclusive=False,
        maximum_version="3.0.0",
    )
    assert not version_satisfies("2.0.0", requirement)
    assert version_satisfies("2.0.1", requirement)


def test_nuget_prerelease_names_are_ordered_before_stable() -> None:
    assert nuget_version_key("1.0.0-experimental1") < nuget_version_key("1.0.0-preview.2")
    assert nuget_version_key("1.0.0-preview.2") < nuget_version_key("1.0.0")


def test_nuget_four_part_versions_are_supported() -> None:
    assert nuget_version_key("1.2.3") < nuget_version_key("1.2.3.1")
