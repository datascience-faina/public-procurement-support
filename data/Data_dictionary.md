# Data Dictionary
### (TED Procurement Dataset)

The dataset contains 4,362,869 rows and 75 columns.
It covers awarded public procurement procedures across EU countries.
Column meanings follow the TED Open Data CAN (Contract Award Notice) schema.


### 1. Notice Identification & Metadata

| Column | Description |
| --- | --- |
| **ID_NOTICE_CAN** | Unique identifier of the Contract Award Notice (CAN). |
| **TED_NOTICE_URL** | URL of the notice on the TED portal. |
| **YEAR** | Publication year (added during preprocessing). |
| **ID_TYPE** | Type of notice (e.g., CAN, PIN). |
| **DT_DISPATCH** | Dispatch/publication date. |
| **XSD_VERSION** | XML schema version used. |
| **CANCELLED** | Indicates whether the notice was cancelled. |
| **CORRECTIONS** | Indicates whether corrections were issued. |

### 2. Contracting Authority (Buyer)

| Column | Description |
| --- | --- |
| **B_MULTIPLE_CAE** | Multiple contracting authorities involved. |
| **CAE_NAME** | Name of the contracting authority. |
| **CAE_NATIONALID** | National identifier of the authority. |
| **CAE_ADDRESS** | Address of the authority. |
| **CAE_TOWN** | Town/city of the authority. |
| **CAE_POSTAL_CODE** | Postal code. |
| **CAE_GPA_ANNEX** | GPA Annex classification. |
| **ISO_COUNTRY_CODE** | Country code of the authority. |
| **ISO_COUNTRY_CODE_GPA** | GPA country classification. |
| **B_MULTIPLE_COUNTRY** | Indicates multiple countries involved. |
| **ISO_COUNTRY_CODE_ALL** | All relevant country codes. |
| **CAE_TYPE** | Type of contracting authority (e.g., ministry, agency). |
| **EU_INST_CODE** | EU institution code (if applicable). |
| **MAIN_ACTIVITY** | Main activity of the authority. |

### 3. Procurement Object (What is being purchased)

| Column | Description |
| --- | --- |
| **TYPE_OF_CONTRACT** | Goods / Services / Works. |
| **TAL_LOCATION_NUTS** | NUTS region of contract execution. |
| **CPV** | Main CPV code describing the procurement object. |
| **MAIN_CPV_CODE_GPA** | GPA classification of CPV. |
| **ADDITIONAL_CPVS** | Additional CPV codes. |
| **TITLE** | Title/description of the procurement. |

### 4. Procedure Characteristics

| Column | Description |
| --- | --- |
| **B_FRA_AGREEMENT** | Framework agreement indicator. |
| **FRA_ESTIMATED** | Estimated value of the framework agreement. |
| **B_FRA_CONTRACT** | Contract under a framework agreement. |
| **B_DYN_PURCH_SYST** | Dynamic purchasing system indicator. |
| **TOP_TYPE** | Type of procedure (open, restricted, negotiated). |
| **B_ACCELERATED** | Accelerated procedure. |
| **OUT_OF_DIRECTIVES** | Outside EU procurement directives. |
| **B_ELECTRONIC_AUCTION** | Electronic auction used. |


### 5. Contract Value

| Column | Description |
| --- | --- |
| **VALUE_EURO** | Estimated contract value (initial). |
| **VALUE_EURO_FIN_1** | Financial value field 1. |
| **VALUE_EURO_FIN_2** | Financial value field 2. |
| **AWARD_EST_VALUE_EURO** | Estimated value at award stage. |
| **AWARD_VALUE_EURO** | Awarded (actual) contract value. |
| **AWARD_VALUE_EURO_FIN_1** | Additional financial value. |


### 6. Award Criteria

| Column | Description |
| --- | --- |
| **CRIT_CODE** | Award criterion code. |
| **CRIT_PRICE_WEIGHT** | Weight of price in evaluation. |
| **CRIT_CRITERIA** | Non-price criteria. |
| **CRIT_WEIGHTS** | Weights of criteria. |

### 7. Award Outcome

| Column | Description |
| --- | --- |
| **NUMBER_AWARDS** | Number of awards in the notice. |
| **ID_AWARD** | Award identifier. |
| **ID_LOT_AWARDED** | Awarded lot identifier. |
| **INFO_ON_NON_AWARD** | Information on non-award. |
| **INFO_UNPUBLISHED** | Unpublished information. |
| **B_AWARDED_TO_A_GROUP** | Awarded to a group of suppliers. |

### 8. Winner (Supplier)

| Column | Description |
| --- | --- |
| **WIN_NAME** | Name of the winning supplier. |
| **WIN_NATIONALID** | National identifier of the supplier. |
| **WIN_ADDRESS** | Address of the supplier. |
| **WIN_TOWN** | Town/city. |
| **WIN_POSTAL_CODE** | Postal code. |
| **WIN_COUNTRY_CODE** | Country code of the supplier. |
| **B_CONTRACTOR_SME** | SME indicator. |

### 9. Competition

| Column | Description |
| --- | --- |
| **NUMBER_OFFERS** | Total number of offers received. |
| **NUMBER_TENDERS_SME** | Offers from SMEs. |
| **NUMBER_TENDERS_OTHER_EU** | Offers from other EU countries. |
| **NUMBER_TENDERS_NON_EU** | Offers from non-EU countries. |
| **NUMBER_OFFERS_ELECTR** | Electronic offers. |

### 10. Contract Details

| Column | Description |
| --- | --- |
| **CONTRACT_NUMBER** | Contract number. |
| **B_SUBCONTRACTED** | Subcontracting indicator. |
| **DT_AWARD** | Award date. |
