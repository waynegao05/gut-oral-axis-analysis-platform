# Pharmacy knowledge data

This directory contains versioned, source-traceable medication knowledge used by
the research pharmacy assistance module.

## Files

- `medication_seed_v1.json`: reviewed collection scope and search terms.
- `high_priority_ddi_v1.json`: a normalized minimum high-priority interaction
  set derived from the 2012 ONC expert-panel publication.
- `openfda_label_evidence_v1.json`: generated RxNorm identifiers and selected
  openFDA Structured Product Label sections. Rebuild it with
  `python -m research.build_drug_knowledge_v1`.

## Source and license boundaries

- RxNorm normalized names and RXCUIs are provided by the U.S. National Library
  of Medicine. NLM does not endorse this project. See
  <https://www.nlm.nih.gov/research/umls/rxnorm/docs/termsofservice.html>.
- openFDA label records are transformed FDA Structured Product Label data. Each
  generated record retains its SPL SET ID, effective date, API query, and a
  DailyMed link. See <https://open.fda.gov/apis/drug/label/> and
  <https://open.fda.gov/terms/>.
- The high-priority DDI set records normalized facts from Phansalkar et al.
  (JAMIA 2012, doi:10.1136/amiajnl-2011-000612). It is a small historical
  consensus starter set, not a current or comprehensive interaction database.
- No DrugBank, Micromedex, First Databank, CredibleMeds, or other restricted
  commercial content is copied into this directory.

## Safety contract

The data supports evidence lookup and limited screening only. A negative result
does not establish that a drug, dose, duration, combination, or probiotic is
safe or appropriate. The application must not convert this data into an order,
prescription, start/stop instruction, or patient-specific dose without qualified
clinical review and the required patient information.
