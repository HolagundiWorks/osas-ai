"""Dataset licence checks."""

from __future__ import annotations

from myna_ai.common.errors import DatasetLicenceError

NONCOMMERCIAL_LICENCES = {
    "cc-by-nc-4.0",
    "cc-by-nc-sa-4.0",
    "cc-by-nc-nd-4.0",
    "cc-by-nc-2.0",
}


def normalise_licence(licence: str) -> str:
    return licence.strip().lower().replace(" ", "-")


def assert_licence_allows_use(
    licence: str,
    *,
    commercial_use: bool,
    acknowledge_noncommercial_licence: bool,
    dataset_id: str,
) -> None:
    """Refuse commercial training on a non-commercial source."""
    normalised = normalise_licence(licence)
    if normalised not in NONCOMMERCIAL_LICENCES:
        return
    if commercial_use or not acknowledge_noncommercial_licence:
        raise DatasetLicenceError(
            f"{dataset_id} is licensed {licence}. "
            "That licence does not allow commercial use. "
            "Refusing to build a training set for a commercial Myna deployment. "
            "For non-commercial research, set commercial_use=false and "
            "acknowledge_noncommercial_licence=true. "
            "Do not upload customer drawings to a third-party host to work around this."
        )
