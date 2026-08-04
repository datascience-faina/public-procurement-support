# Datenwörterbuch (Data Dictionary)

Dieses Dokument beschreibt die wichtigsten Variablen des TED‑Datensatzes (Tenders Electronic Daily), die in diesem Projekt verwendet werden.  
Die vollständige Struktur umfasst 81 Spalten. Die folgende Übersicht fasst die Variablen in thematische Kategorien zusammen und erläutert ihre Bedeutung im Kontext öffentlicher Vergaben.

---

## 1. Allgemeine Metadaten
| Variable | Beschreibung |
|---------|--------------|
| ID_NOTICE_CAN | Eindeutige Kennung der Bekanntmachung |
| TED_NOTICE_URL | URL zur Ausschreibung auf ted.europa.eu |
| YEAR | Veröffentlichungsjahr |
| DT_DISPATCH | Datum der Veröffentlichung |
| XSD_VERSION | Version des Datenformats |

---

## 2. Auftraggeber (Contracting Authority)
| Variable | Beschreibung |
|---------|--------------|
| CAE_NAME | Name der Vergabestelle |
| CAE_ADDRESS | Adresse |
| CAE_TOWN | Stadt |
| CAE_POSTAL_CODE | Postleitzahl |
| ISO_COUNTRY_CODE | Land |
| MAIN_ACTIVITY | Haupttätigkeit der Vergabestelle |
| CAE_TYPE | Typ der Behörde (Ministerium, Agentur etc.) |

---

## 3. Verfahrensmerkmale
| Variable | Beschreibung |
|---------|--------------|
| TYPE_OF_CONTRACT | Art des Auftrags (Lieferungen, Dienstleistungen, Bau) |
| TOP_TYPE | Verfahrenstyp (z. B. Offenes Verfahren) |
| B_ACCELERATED | Beschleunigtes Verfahren |
| OUT_OF_DIRECTIVES | Verfahren außerhalb EU‑Richtlinien |
| B_DYN_PURCH_SYST | Dynamisches Beschaffungssystem |
| B_ELECTRONIC_AUCTION | Elektronische Auktion |
| CRIT_CODE | Bewertungskriterien |
| CRIT_PRICE_WEIGHT | Gewichtung des Preises |
| CRIT_WEIGHTS | Gewichtung aller Kriterien |

---

## 4. Wettbewerb (Competition)
| Variable | Beschreibung |
|---------|--------------|
| NUMBER_OFFERS | Anzahl eingegangener Angebote |
| NUMBER_TENDERS_SME | Angebote von KMU |
| NUMBER_TENDERS_OTHER_EU | Angebote aus anderen EU‑Ländern |
| NUMBER_TENDERS_NON_EU | Angebote aus Nicht‑EU‑Ländern |
| LOTS_NUMBER | Anzahl der Lose |
| B_AWARDED_TO_A_GROUP | Zuschlag an eine Bietergemeinschaft |

---

## 5. Wirtschaftliche Kennzahlen
| Variable | Beschreibung |
|---------|--------------|
| VALUE_EURO | Geschätzter Auftragswert |
| VALUE_EURO_FIN_1 | Finaler Auftragswert (Variante 1) |
| VALUE_EURO_FIN_2 | Finaler Auftragswert (Variante 2) |
| AWARD_EST_VALUE_EURO | Geschätzter Wert des Zuschlags |
| AWARD_VALUE_EURO | Tatsächlicher Zuschlagswert |
| AWARD_VALUE_EURO_FIN_1 | Finaler Zuschlagswert |

---

## 6. Auftragnehmer (Winner)
| Variable | Beschreibung |
|---------|--------------|
| WIN_NAME | Name des Auftragnehmers |
| WIN_ADDRESS | Adresse |
| WIN_TOWN | Stadt |
| WIN_POSTAL_CODE | Postleitzahl |
| WIN_COUNTRY_CODE | Land |
| B_CONTRACTOR_SME | KMU‑Status des Auftragnehmers |

---

## 7. Textfelder
| Variable | Beschreibung |
|---------|--------------|
| TITLE | Titel der Ausschreibung |
| LOT_DESCRIPTION | Beschreibung des Loses |
| CRIT_CRITERIA | Textliche Bewertungskriterien |

---

## 8. Zeitliche Angaben
| Variable | Beschreibung |
|---------|--------------|
| DT_AWARD | Datum des Zuschlags |
| DT_CONTRACT_START | Vertragsbeginn (falls vorhanden) |
| DT_CONTRACT_END | Vertragsende (falls vorhanden) |

---

## Hinweis
Dieses Datenwörterbuch fasst die wichtigsten Variablen zusammen.  
Die vollständige technische Dokumentation ist auf der offiziellen EU‑Open‑Data‑Plattform 