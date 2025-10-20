"""AI-powered data extraction from K-1 tax forms."""

import json
import requests
from typing import Dict, List, Any, Type, Tuple
from pydantic import BaseModel

from .config import (
    OPENROUTER_URL, 
    OPENROUTER_HEADERS, 
    DEFAULT_MODEL, 
    MAX_TOKENS, 
    TEMPERATURE, 
    MAX_RETRIES
)
from .pdf_processor import encode_pdf_to_base64
from .pydantic_model import k1_cover_page, k1_federal_footnotes


def _strip_code_fences(text: str) -> str:
    """Remove leading/trailing markdown code fences if present."""
    if not text:
        return ""
    stripped = text.strip()
    if stripped.startswith("```"):
        first_newline = stripped.find("\n")
        if first_newline != -1:
            body = stripped[first_newline + 1 :]
            if body.endswith("```"):
                body = body[: -3]
            return body.strip()
    return stripped


def _find_first_json_object(text: str) -> str | None:
    """Return the substring of the first balanced JSON object, or None if not found."""
    if not text:
        return None
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    i = start
    while i < len(text):
        ch = text[i]
        if ch == '{':
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
        i += 1
    return None


def create_field_guide(model_class) -> str:
    """Create a human-readable field guide for the extraction prompt.
    Adds concise hints for many common fields while staying schema-agnostic.
    """
    guides = []
    
    for field_name in list(model_class.model_fields.keys())[:200]:
        if "partnership_name" in field_name:
            guides.append(f"- {field_name}: Partnership's legal name (top of form)")
        elif "employer_identification_number" in field_name:
            guides.append(f"- {field_name}: Partnership EIN (nine digits)")
        elif "line_1_ordinary_business_income_loss_passive" in field_name:
            guides.append(f"- {field_name}: Part III Line 1 passive indicator (checkbox/text)")
        elif field_name == "line_1_ordinary_business_income_loss":
            guides.append(f"- {field_name}: Part III Line 1 - Ordinary business income (loss)")
        elif field_name == "line_2_net_rental_real_estate_income_loss":
            guides.append(f"- {field_name}: Part III Line 2 - Net rental real estate income (loss)")
        elif field_name == "line_3_other_rental_income_loss":
            guides.append(f"- {field_name}: Part III Line 3 - Other net rental income (loss)")
        elif field_name == "line_4a_guaranteed_payments_for_services":
            guides.append(f"- {field_name}: Part III Line 4a - Guaranteed payments for services")
        elif field_name == "line_4b_guaranteed_payments_for_capital":
            guides.append(f"- {field_name}: Part III Line 4b - Guaranteed payments for capital")
        elif field_name == "line_4c_total_guaranteed_payments":
            guides.append(f"- {field_name}: Part III Line 4c - Total guaranteed payments")
        elif field_name == "line_5_interest_income":
            guides.append(f"- {field_name}: Part III Line 5 - Interest income (main form)")
        elif field_name == "line_5_interest_income_us_government_interest":
            guides.append(f"- {field_name}: Part III Line 5 subset - U.S. government interest")
        elif field_name == "line_6a_ordinary_dividends":
            guides.append(f"- {field_name}: Part III Line 6a - Ordinary dividends")
        elif field_name == "line_6b_qualified_dividends":
            guides.append(f"- {field_name}: Part III Line 6b - Qualified dividends")
        elif field_name == "line_7_royalties":
            guides.append(f"- {field_name}: Part III Line 7 - Royalties")
        elif field_name == "line_8_net_short_term_capital_gain_loss":
            guides.append(f"- {field_name}: Part III Line 8 - Net short-term capital gain (loss)")
        elif field_name == "line_9a_net_long_term_capital_gain_loss":
            guides.append(f"- {field_name}: Part III Line 9a - Net long-term capital gain (loss)")
        elif field_name == "line_9b_collectibles_28_percent_gain_loss":
            guides.append(f"- {field_name}: Part III Line 9b - Collectibles (28%) gain (loss)")
        elif field_name == "line_9c_uncaptured_section_1250_gain":
            guides.append(f"- {field_name}: Part III Line 9c - Unrecaptured section 1250 gain")
        elif field_name == "line_10_net_section_1231_gain_loss":
            guides.append(f"- {field_name}: Part III Line 10 - Net section 1231 gain (loss)")
        elif field_name == "line_11a_other_income_total":
            guides.append(f"- {field_name}: Statement: Other income (loss) total")
        elif field_name == "line_11c_section_1256_gain_loss":
            guides.append(f"- {field_name}: Statement: Section 1256 contracts & straddles gain (loss)")
        elif field_name == "line_11ZZ_ordinary_income_section_475f":
            guides.append(f"- {field_name}: Statement: Section 475(f) mark-to-market income")
        elif field_name == "line_11ZZ_pfic_qef_income":
            guides.append(f"- {field_name}: Statement: PFIC QEF income")
        elif field_name == "line_11ZZ_section_988_total":
            guides.append(f"- {field_name}: Statement: Section 988 foreign currency gain (loss)")
        elif field_name == "line_11ZZ_swap_net_income_loss":
            guides.append(f"- {field_name}: Statement: Swap/derivative net income (loss)")
        elif field_name == "line_11ZZ_other_income_loss":
            guides.append(f"- {field_name}: Statement: Other income (loss)")
        elif field_name == "line_11ZZ_other_portfolio_income_loss":
            guides.append(f"- {field_name}: Statement: Other portfolio income (loss)")
        elif field_name == "line_11ZZ_other_ordinary_income_loss_total":
            guides.append(f"- {field_name}: Statement: Other ordinary income (loss) total")
        elif field_name == "line_11ZZ_interest_income_domestic":
            guides.append(f"- {field_name}: Statement: Interest income - domestic")
        elif field_name == "line_11ZZ_interest_income_foreign":
            guides.append(f"- {field_name}: Statement: Interest income - foreign")
        elif field_name == "line_11ZZ_dividends_qualified_domestic":
            guides.append(f"- {field_name}: Statement: Qualified dividends - domestic")
        elif field_name == "line_11ZZ_dividends_qualified_foreign":
            guides.append(f"- {field_name}: Statement: Qualified dividends - foreign")
        elif field_name == "line_11ZZ_dividends_non_qualified_domestic":
            guides.append(f"- {field_name}: Statement: Non-qualified dividends - domestic")
        elif field_name == "line_11ZZ_dividends_non_qualified_foreign":
            guides.append(f"- {field_name}: Statement: Non-qualified dividends - foreign")
        elif field_name.startswith("line_13h_investment_interest_investing_schedule_A"):
            guides.append(f"- {field_name}: Statement/Schedule A - Investment interest (investing)")
        elif field_name.startswith("line_13h_investment_interest_trading_schedule_E"):
            guides.append(f"- {field_name}: Statement/Schedule E - Investment interest (trading)")
        elif field_name == "line_13l_deductions_portfolio_other":
            guides.append(f"- {field_name}: Statement: Portfolio deductions - other")
        elif "13ZZ" in field_name:
            guides.append(f"- {field_name}: Statement (supplemental) - detailed item in line 13 category")
        elif field_name == "line_15o_backup_withholding":
            guides.append(f"- {field_name}: Statement: Backup withholding / Form 1099 withholding")
        elif field_name == "line_15zz_other_credits":
            guides.append(f"- {field_name}: Statement: Other credits")
        elif field_name == "line_18a_tax_exempt_interest_income":
            guides.append(f"- {field_name}: Tax-exempt interest income (keep decimals when shown)")
        elif field_name == "line_18b_other_tax_exempt_income":
            guides.append(f"- {field_name}: Other tax-exempt income (keep decimals when shown)")
        elif field_name == "line_18c_nondeductible_expenses":
            guides.append(f"- {field_name}: Nondeductible expenses (keep decimals when shown)")
        elif field_name == "line_20V_unrelated_business_taxable_income":
            guides.append(f"- {field_name}: Statement: Unrelated business taxable income (UBTI)")
        elif field_name == "line_20AA_section_704c_information":
            guides.append(f"- {field_name}: Statement: Section 704(c) information")
        elif field_name == "line_20AG_gross_receipts_section_448_c":
            guides.append(f"- {field_name}: Statement: Gross receipts per Section 448(c)")
        elif field_name == "capital_contributions_during_year":
            guides.append(f"- {field_name}: Part II - Capital contributions during year")
        elif field_name == "withdrawals_and_distributions_cash":
            guides.append(f"- {field_name}: Part II - Withdrawals & distributions (negative if parentheses)")
        elif field_name == "ending_capital_account":
            guides.append(f"- {field_name}: Part II - Ending capital account")
        else:
            guides.append(f"- {field_name}: {field_name.replace('_', ' ')}")
    
    return "\n".join(guides)


def validate_field_groups(extracted_data: dict) -> dict:
    """Simple validation to prevent duplicate values across related fields."""
    return extracted_data  # For now, trust the model's field assignments


def extract_with_pdf(
    pdf_path: str, 
    fields_to_extract: List[str], 
    section_name: str, 
    model_class: Type[BaseModel], 
    examples: str = "", 
    partial: bool = True, 
    max_retries: int = MAX_RETRIES
) -> Dict[str, Any]:
    """Extract specific fields from PDF via OpenRouter with structured output."""
    
    last_error = None
    for attempt in range(max_retries):
        try:
            # Convert PDF to base64 with progressively smaller size limits
            max_size = 20 >> attempt  # 20MB, 10MB, 5MB
            base64_pdf = encode_pdf_to_base64(pdf_path, max_size_mb=max_size)
            
            # Prepare content with system message
            content = [
                {
                    "type": "text",
                    "text": "You are an expert at extracting data from IRS Schedule K-1 (Form 1065) tax forms. "
                           "Your task is to find EXACT matches for field labels and extract their values. "
                           "DO NOT move values between fields or split/combine values. "
                           "If a field's label is not found, use 0 for numbers or \"\" for text."
                },
                {
                    "type": "file",
                    "file": {
                        "filename": "document.pdf",
                        "file_data": f"data:application/pdf;base64,{base64_pdf}"
                    }
                }
            ]
            
            # Create a more general, strategy-flexible prompt with strict JSON formatting
            field_list_csv = ', '.join(fields_to_extract)
            field_guide_text = create_field_guide(model_class or k1_cover_page)
            prompt = f"""
You are an expert at extracting structured data from IRS Schedule K-1 (Form 1065) PDFs.
Your goal is to return values for EXACTLY these fields (whitelist): {field_list_csv}

Strategy (flexible):
- Use reliable cues: printed labels, nearby headers, layout, and attached statements.
- Prefer authoritative sections (Part III for primary line items; statements for ZZ items); handle format variations pragmatically.
- Do not speculate. If a value is not clearly present, use the default (0 for numbers, "" for strings).

Normalization rules (must apply):
- Remove currency symbols/commas/spaces ("$1,234" -> 1234).
- Parentheses mean negative ("(100)" -> -100).
- Round decimals unless a field is clearly marked to preserve decimals.
- Strings: trim; use "" for absent string fields (e.g., *_logic, names/EIN).
- Never copy a value across fields unless the document explicitly states it.

 Section constraints (strict):
 - Lines 1–10: Only from the main K‑1 first page, Part III.
 - 11a/11c and all 11ZZ/13ZZ/20XX: Only from attached statements/supplemental schedules (not the main form).
 - Capital account (contributions/withdrawals/ending): Part II on main page; treat parentheses as negative.

 Minimal aliases permitted (do not expand beyond this list):
 - 20V: UBTI or "Unrelated Business" indicates unrelated business taxable income.
 - 20AA: Mentions of "Section 704(c)".
 - 20AG: Mentions of "Section 448(c)" or "gross receipts".
 - 11c: Mentions of "Section 1256" for contracts & straddles.
 - PFIC QEF: Mentions "PFIC QEF".

Output format (STRICT):
- Return ONLY a single JSON object mapping each requested field to its value.
- Include ALL and ONLY the requested fields as top-level keys.
- Numeric fields must be integers. String fields must be strings. No extra keys or text.

Field guide (informative hints only):
{field_guide_text}

Examples:
- "(300)" -> -300
- "$1,234" -> 1234
- Missing/unclear -> 0 (numbers) or "" (strings)

Return only the JSON object with the requested fields.
{examples}
"""
            
            # Prepare API request payload
            payload = {
                "model": DEFAULT_MODEL,
                "max_tokens": MAX_TOKENS,
                "temperature": TEMPERATURE,
                "messages": [
                    {
                        "role": "user",
                        "content": content + [{"type": "text", "text": prompt}]
                    }
                ],
                "response_format": {"type": "json_object"}
            }
            
            # Make API call
            response = requests.post(OPENROUTER_URL, headers=OPENROUTER_HEADERS, json=payload)
            response.raise_for_status()
            
            # Parse response
            response_data = response.json()
            response_text = _strip_code_fences(response_data["choices"][0]["message"]["content"]) 
            
            # Parse JSON, with fallback to first balanced object
            try:
                json_data = json.loads(response_text)
            except json.JSONDecodeError:
                candidate = _find_first_json_object(response_text)
                if not candidate:
                    raise
                json_data = json.loads(candidate)
            # Support strict wrapper {values, not_found, evidence} or flat responses
            if isinstance(json_data, dict) and "values" in json_data and isinstance(json_data["values"], dict):
                json_payload = json_data["values"]
            else:
                json_payload = json_data
            
            # Validate with Pydantic
            if partial:
                # For partial data, only validate the fields we asked for
                filtered_data = {k: v for k, v in json_payload.items() if k in fields_to_extract}
                # Add default values for missing fields
                for field in fields_to_extract:
                    if field not in filtered_data:
                        filtered_data[field] = "" if field.endswith('_logic') or field in ['partnership_name', 'partnership_employer_identification_number'] else 0
                extracted_data = model_class.model_validate(filtered_data) if model_class else filtered_data
            else:
                # For complete data, validate all fields
                extracted_data = model_class.model_validate(json_payload) if model_class else json_payload
            
            # Apply field group validation
            validated_data = validate_field_groups(extracted_data.model_dump() if model_class else extracted_data)
            
            # Final validation pass
            final_data = {}
            for field in fields_to_extract:
                value = validated_data.get(field, 0)
                # Handle string fields (including _logic fields)
                if field.endswith('_logic') or field in ['partnership_name', 'partnership_employer_identification_number']:
                    final_data[field] = str(value) if value else ""
                else:
                    final_data[field] = int(value) if value else 0
            
            return final_data
            
        except requests.RequestException as e:
            last_error = e
            if attempt < max_retries - 1:
                print(f"Attempt {attempt + 1} failed, retrying with smaller PDF size...")
                continue
            raise
        except Exception as e:
            print(f"Error during {section_name} extraction: {e}")
            import traceback
            traceback.print_exc()
            return {field: "" if field.endswith('_logic') or field in ['partnership_name', 'partnership_employer_identification_number'] else 0 
                   for field in fields_to_extract}


def extract_k1_data(pdf_path: str) -> Tuple[k1_cover_page, k1_federal_footnotes]:
    """Extract all K-1 data in a single pass."""
    
    # Get all fields from both models
    cover_fields = list(k1_cover_page.model_fields.keys())
    footnote_fields = list(k1_federal_footnotes.model_fields.keys())
    all_fields = cover_fields + footnote_fields
    
    # Create comprehensive examples
    examples = """
EXACT FIELD MATCHING EXAMPLES:
1. Main Form Fields (ONLY extract if label matches EXACTLY):
   - Label "Line 1. Ordinary business income (loss)" with "$100" → "line_1_ordinary_business_income_loss": 100
   - Label "Line 2. Net rental real estate income (loss)" with "(50)" → "line_2_net_rental_real_estate_income_loss": -50
   - Label "Line 5. Interest income" with "$75" → "line_5_interest_income": 75
   - Label "Line 6a. Ordinary dividends" with "$200" → "line_6a_ordinary_dividends": 200
   - Label "Line 6b. Qualified dividends" with "$150" → "line_6b_qualified_dividends": 150

2. Supplemental Fields (ONLY extract if label matches EXACTLY):
   - Label "Section 1256 contracts & straddles" with "(300)" → "line_11c_section_1256_gain_loss": -300
   - Label "Investment interest expense - Schedule A" with "$25" → "line_13h_investment_interest_investing_schedule_A": 25
   - Label "Investment interest expense - Schedule E" with "$10" → "line_13h_investment_interest_trading_schedule_E": 10
   - Label "Other deductions" with "$125" → "line_13ZZ_other_deductions_total": 125

IMPORTANT:
- ONLY extract a value if the label matches EXACTLY
- DO NOT move values between fields
- DO NOT split or combine values
- If a field's label is not found, use 0 for numbers or "" for text"""
    
    try:
        # Extract all data at once
        data = extract_with_pdf(pdf_path, all_fields, "K-1 Form (All Sections)", None, examples, partial=False)
        
        # Split data into cover page and footnotes
        cover_data = {k: v for k, v in data.items() if k in cover_fields}
        footnote_data = {k: v for k, v in data.items() if k in footnote_fields}
        
        # Create and validate models
        cover_page = k1_cover_page(**cover_data)
        footnotes = k1_federal_footnotes(**footnote_data)
        
        return cover_page, footnotes
    except Exception as e:
        print(f"Error extracting data: {e}")
        import traceback
        traceback.print_exc()
        
        # Return empty models with default values
        cover_page = k1_cover_page(**{k: "" if k in ['partnership_name', 'partnership_employer_identification_number'] else 0 
                                    for k in cover_fields})
        footnotes = k1_federal_footnotes(**{k: "" if k.endswith('_logic') else 0 
                                          for k in footnote_fields})
        return cover_page, footnotes
