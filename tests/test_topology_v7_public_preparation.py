from __future__ import annotations

import pandas as pd

from experiments.topology_v7_compositional_temporal_v1.prepare_public_data import (
    FIVE_GENERA,
    _relative_genus_abundance,
    _taxonomy_genus,
)


def test_relative_abundance_uses_full_library_size() -> None:
    counts = pd.DataFrame(
        {
            "otu_f": [2.0, 0.0],
            "otu_p": [3.0, 0.0],
            "other": [5.0, 0.0],
        },
        index=["a", "b"],
    )
    result = _relative_genus_abundance(
        counts,
        {
            "otu_f": "Fusobacterium",
            "otu_p": "Prevotella",
            "other": "Bacteroides",
        },
    )

    assert list(result.columns) == [
        f"abundance_{genus}" for genus in FIVE_GENERA
    ]
    assert result.loc["a", "abundance_Fusobacterium"] == 0.2
    assert result.loc["a", "abundance_Prevotella"] == 0.3
    assert result.loc["b"].sum() == 0.0


def test_debelius_taxonomy_parser_uses_genus() -> None:
    assert (
        _taxonomy_genus(
            "k__Bacteria;p__Fusobacteria;g__Fusobacterium;"
            "s__Fusobacterium_nucleatum"
        )
        == "Fusobacterium"
    )
    assert _taxonomy_genus("k__Bacteria;p__Firmicutes") is None
