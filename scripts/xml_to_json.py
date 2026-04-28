"""
PatentPilot AI — USPTO XML to JSON Converter
=============================================
Converts the bulk USPTO patent application XML file (concatenated records)
into a clean JSON dataset suitable for embedding and analysis.

The USPTO bulk XML file is NOT valid XML — it contains thousands of
concatenated <us-patent-application> records, each preceded by an
<?xml ...?> declaration. This script stream-splits the file and parses
each record individually.

Usage:
    python scripts/xml_to_json.py [--input ipa260423.xml] [--output data/patents.json] [--limit 500]
"""

import argparse
import json
import re
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional


def strip_xml_tags(text: str) -> str:
    """Remove XML/HTML tags from text and normalize whitespace."""
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def extract_text_recursive(element) -> str:
    """
    Recursively extract all text content from an XML element and its children.
    This handles mixed content like <claim-text> containing <claim-ref> tags.
    """
    parts = []
    if element.text:
        parts.append(element.text)
    for child in element:
        parts.append(extract_text_recursive(child))
        if child.tail:
            parts.append(child.tail)
    return " ".join(parts)


def parse_patent_record(xml_string: str) -> Optional[dict]:
    """
    Parse a single <us-patent-application> XML record and extract
    the key fields: doc_number, title, abstract, claims, date, inventors.
    
    Returns None if parsing fails or required fields are missing.
    """
    try:
        root = ET.fromstring(xml_string)
    except ET.ParseError:
        # Try fixing common issues — remove DOCTYPE line
        lines = xml_string.split("\n")
        cleaned = "\n".join(
            line for line in lines
            if not line.strip().startswith("<!DOCTYPE")
        )
        try:
            root = ET.fromstring(cleaned)
        except ET.ParseError:
            return None

    record = {}

    # --- Document number ---
    pub_ref = root.find(".//publication-reference/document-id")
    if pub_ref is not None:
        doc_num_el = pub_ref.find("doc-number")
        kind_el = pub_ref.find("kind")
        date_el = pub_ref.find("date")
        record["doc_number"] = doc_num_el.text if doc_num_el is not None else ""
        record["kind"] = kind_el.text if kind_el is not None else ""
        record["pub_date"] = date_el.text if date_el is not None else ""
    else:
        record["doc_number"] = ""
        record["kind"] = ""
        record["pub_date"] = ""

    # --- Title ---
    title_el = root.find(".//invention-title")
    if title_el is not None:
        record["title"] = extract_text_recursive(title_el).strip()
    else:
        record["title"] = ""

    # --- Abstract ---
    abstract_el = root.find(".//abstract")
    if abstract_el is not None:
        abstract_text = extract_text_recursive(abstract_el)
        record["abstract"] = strip_xml_tags(abstract_text).strip()
    else:
        record["abstract"] = ""

    # --- Claims ---
    claims = []
    claims_el = root.find(".//claims")
    if claims_el is not None:
        for claim_el in claims_el.findall("claim"):
            claim_id = claim_el.get("id", "")
            claim_num = claim_el.get("num", "")
            claim_text = extract_text_recursive(claim_el)
            claim_text = strip_xml_tags(claim_text).strip()
            # Remove leading claim number (e.g., "1. " or "15. ")
            claim_text = re.sub(r"^\d+\.\s*", "", claim_text)
            if claim_text:
                claims.append({
                    "claim_id": claim_id,
                    "claim_number": int(claim_num) if claim_num.isdigit() else 0,
                    "text": claim_text
                })
    record["claims"] = claims

    # --- Inventors ---
    inventors = []
    for inventor_el in root.findall(".//inventors/inventor"):
        addr = inventor_el.find("addressbook")
        if addr is not None:
            first = addr.findtext("first-name", "")
            last = addr.findtext("last-name", "")
            if first or last:
                inventors.append(f"{first} {last}".strip())
    record["inventors"] = inventors

    # --- Classification (CPC) ---
    cpc_codes = []
    for cpc_el in root.findall(".//classifications-cpc//classification-cpc"):
        section = cpc_el.findtext("section", "")
        cls = cpc_el.findtext("class", "")
        subcls = cpc_el.findtext("subclass", "")
        main_group = cpc_el.findtext("main-group", "")
        subgroup = cpc_el.findtext("subgroup", "")
        code = f"{section}{cls}{subcls}{main_group}/{subgroup}"
        if len(code) > 3:
            cpc_codes.append(code)
    record["cpc_codes"] = list(set(cpc_codes))  # deduplicate

    # --- Skip records with no abstract and no claims ---
    if not record["abstract"] and not record["claims"]:
        return None

    return record


def split_and_parse(input_path: str, output_path: str, limit: Optional[int] = None):
    """
    Stream-read the concatenated USPTO XML file, split into individual
    patent records, parse each, and write to JSON.
    
    Memory-safe: processes one record at a time.
    """
    input_file = Path(input_path)
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    if not input_file.exists():
        print(f"ERROR: Input file not found: {input_file}")
        sys.exit(1)

    file_size = input_file.stat().st_size
    print(f"Input file: {input_file} ({file_size / 1e6:.1f} MB)")
    print(f"Output file: {output_file}")
    if limit:
        print(f"Limiting to first {limit} records")
    print()

    patents = []
    current_record_lines = []
    in_record = False
    record_count = 0
    parsed_count = 0
    skipped_count = 0
    start_time = time.time()

    with open(input_file, "r", encoding="utf-8", errors="replace") as f:
        for line_num, line in enumerate(f, 1):
            # Detect start of a new record
            if line.strip().startswith("<?xml"):
                # If we were building a record, process it
                if current_record_lines and in_record:
                    xml_str = "\n".join(current_record_lines)
                    record = parse_patent_record(xml_str)
                    record_count += 1

                    if record:
                        patents.append(record)
                        parsed_count += 1
                    else:
                        skipped_count += 1

                    if parsed_count % 200 == 0 and parsed_count > 0:
                        elapsed = time.time() - start_time
                        rate = parsed_count / elapsed
                        print(
                            f"  Parsed {parsed_count} patents "
                            f"(skipped {skipped_count}) "
                            f"[{rate:.1f} records/sec]"
                        )

                    if limit and parsed_count >= limit:
                        break

                current_record_lines = [line.rstrip()]
                in_record = False
                continue

            # Detect DOCTYPE line (skip it, we handle parsing without it)
            if line.strip().startswith("<!DOCTYPE"):
                continue

            # Detect the opening tag of a patent record
            if "<us-patent-application" in line:
                in_record = True

            if in_record or current_record_lines:
                current_record_lines.append(line.rstrip())

        # Don't forget the last record
        if current_record_lines and in_record and (not limit or parsed_count < limit):
            xml_str = "\n".join(current_record_lines)
            record = parse_patent_record(xml_str)
            record_count += 1
            if record:
                patents.append(record)
                parsed_count += 1
            else:
                skipped_count += 1

    elapsed = time.time() - start_time

    # Write output
    print(f"\nWriting {len(patents)} patents to {output_file}...")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(patents, f, indent=2, ensure_ascii=False)

    output_size = output_file.stat().st_size
    print(f"\n{'='*60}")
    print(f"CONVERSION COMPLETE")
    print(f"{'='*60}")
    print(f"  Total records found:  {record_count}")
    print(f"  Successfully parsed:  {parsed_count}")
    print(f"  Skipped (empty):      {skipped_count}")
    print(f"  Output size:          {output_size / 1e6:.1f} MB")
    print(f"  Time elapsed:         {elapsed:.1f} seconds")
    print(f"  Output path:          {output_file}")
    print(f"{'='*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert USPTO bulk XML to JSON for PatentPilot AI"
    )
    parser.add_argument(
        "--input", "-i",
        default="ipa260423.xml",
        help="Path to the USPTO XML file (default: ipa260423.xml)"
    )
    parser.add_argument(
        "--output", "-o",
        default="data/patents.json",
        help="Output JSON file path (default: data/patents.json)"
    )
    parser.add_argument(
        "--limit", "-l",
        type=int,
        default=None,
        help="Limit number of patents to parse (default: all)"
    )
    args = parser.parse_args()
    split_and_parse(args.input, args.output, args.limit)
