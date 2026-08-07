# Data Dictionary — Post Feature Engineering & NLP

This data dictionary describes all variables in the final modelling dataset after:
- data cleaning
- feature engineering
- CPV hierarchy extraction
- competition & risk features
- text preprocessing
- NLP feature generation (TF‑IDF + SVD + NMF + SVM)

The final dataset contains 4,039,906 rows × 147 columns.

### 1. Core Metadata & Administrative Fields

| Column | Type | Description |
| --- | --- | --- |
| **YEAR** | int64 | Year of tender publication. |
| **ID_TYPE** | int64 | Internal TED notice type classification. |
| **XSD_VERSION** | string | XML schema version used in the TED notice. |
| **CANCELLED** | int64 | Indicator whether the tender was cancelled (0/1). |
| **CORRECTIONS** | int64 | Number of corrections issued for the notice. |
| **ISO_COUNTRY_CODE** | string | Country of the contracting authority (ISO‑2). |
| **CAE_TYPE** | string | Type of contracting authority (public body category). |
| **B_AWARDED_BY_CENTRAL_BODY** | string | Whether award was made by a central purchasing body. |
| **TYPE_OF_CONTRACT** | string | Contract type: Works (W), Supplies (S), Services (U). |
| **TAL_LOCATION_NUTS** | string | NUTS regional code of contract execution. |
| **B_DYN_PURCH_SYST** | int64 | Dynamic purchasing system indicator. |
| **ID_LOT** | string | Lot identifier (if applicable). |
| **B_EU_FUNDS** | string | Whether the contract is co-financed by EU funds. |
| **TOP_TYPE** | string | Type of procurement procedure. |
| **B_ACCELERATED** | int64 | Accelerated procedure indicator. |
| **OUT_OF_DIRECTIVES** | int64 | Whether the tender falls outside EU directives. |
| **CRIT_CODE** | string | Award criterion type (price, quality, mixed). |
| **CRIT_PRICE_WEIGHT** | float64 | Weight of price in award criteria (retained despite missingness). |
| **B_ELECTRONIC_AUCTION** | int64 | Electronic auction indicator. |
| **NUMBER_AWARDS** | int64 | Number of awarded lots or contracts. |
| **B_AWARDED_TO_A_GROUP** | int64 | Whether award was made to a consortium. |
| **WIN_COUNTRY_CODE** | string | Country of the winning bidder. |
| **B_CONTRACTOR_SME** | int64 | SME indicator for the awarded contractor. |
| **B_SUBCONTRACTED** | string | Whether subcontracting was declared. |


### 2. Date & Duration Features

| Column | Type | Description |
| --- | --- | --- |
| **AWARD_QUARTER** | float64 | Quarter of award decision (1–4). |
| **DAYS_TO_AWARD** | float64 | Duration between publication and award (may be negative due to reporting delays). |


### 3. CPV Hierarchy Features

| Column | Type | Description |
| --- | --- | --- |
| **CPV_DIVISION** | string | First 2 digits of CPV code (high-level category). |
| **CPV_GROUP** | string | First 3 digits of CPV code (mid-level category). |
| **CPV_CLASS** | string | First 4 digits of CPV code (detailed category). |


### 4. Competition & Risk Features

| Column | Type | Description |
| --- | --- | --- |
| **IS_FAILED_TENDER** | int64 | Indicator: ≤1 offer → failed tender. |
| **IS_LOW_COMPETITION** | int64 | Indicator: exactly 2 offers → low competition. |
| **OFFERS_BIN** | category | Competition level: ``failed``, ``low``, ``medium``, ``high``. |
| **VALUE_BIN** | category | Contract value category: ``micro``, ``small``, ``medium``, ``large``, ``mega``. |
| **HAS_MULTIPLE_LOTS** | int64 | Whether the tender contains multiple lots. |
| **LOTS_BIN** | category | Lot complexity category: ``single``, ``few``, ``many``, ``mega``. |
| **VALUE_EURO_MISSING** | int64 | Missingness flag for VALUE_EURO. |
| **AWARD_VALUE_EURO_MISSING** | int64 | Missingness flag for AWARD_VALUE_EURO. |
| **NUMBER_OFFERS_MISSING** | int64 | Missingness flag for NUMBER_OFFERS. |



### 5. NLP Features — TF‑IDF + SVD (100 Components)

| Column Pattern | Type | Description |
| --- | --- | --- |
| **NLP_SVD_k** | float64 | Dense semantic components extracted from TF‑IDF using Truncated SVD (k = 0…99). These represent compressed semantic structure of ``TEXT_ALL`` and are used as numerical features for modelling. |