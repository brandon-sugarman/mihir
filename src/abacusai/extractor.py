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


def create_field_guide(model_class) -> str:
    """Create a human-readable field guide for the extraction prompt."""
    guides = []
    
    for field_name in list(model_class.model_fields.keys())[:30]:  # Show first 30 as examples
        if "partnership_name" in field_name:
            guides.append(f"- {field_name}: Partnership's legal name")
        elif "partnership_employer" in field_name:
            guides.append(f"- {field_name}: Partnership's EIN")
        elif "line_1_ordinary_business_income_loss_passive" in field_name:
            guides.append(f"- {field_name}: Part III Line 1 (check passive box)")
        elif "line_1_ordinary_business_income_loss" in field_name:
            guides.append(f"- {field_name}: Part III Line 1 - Ordinary business income/loss")
        elif "line_2_net_rental" in field_name:
            guides.append(f"- {field_name}: Part III Line 2 - Net rental real estate income/loss")
        elif "line_3_other_rental" in field_name:
            guides.append(f"- {field_name}: Part III Line 3 - Other net rental income/loss")
        elif "line_4a_guaranteed_payments_for_services" in field_name:
            guides.append(f"- {field_name}: Part III Line 4a - Guaranteed payments for services")
        elif "line_4b_guaranteed_payments_for_capital" in field_name:
            guides.append(f"- {field_name}: Part III Line 4b - Guaranteed payments for capital")
        elif "line_4c_total_guaranteed" in field_name:
            guides.append(f"- {field_name}: Part III Line 4c - Total guaranteed payments")
        elif "line_5_interest_income" in field_name and "us_government" not in field_name:
            guides.append(f"- {field_name}: Part III Line 5 - Interest income")
        elif "us_government_interest" in field_name:
            guides.append(f"- {field_name}: Part III Line 5 - U.S. government interest (subset)")
        elif "line_6a_ordinary_dividends" in field_name:
            guides.append(f"- {field_name}: Part III Line 6a - Ordinary dividends")
        elif "line_6b_qualified_dividends" in field_name:
            guides.append(f"- {field_name}: Part III Line 6b - Qualified dividends")
        elif "line_7_royalties" in field_name:
            guides.append(f"- {field_name}: Part III Line 7 - Royalties")
        elif "capital_contributions_during_year" in field_name:
            guides.append(f"- {field_name}: Part II - Capital contributed during year")
        elif "withdrawals_and_distributions" in field_name:
            guides.append(f"- {field_name}: Part II - Withdrawals & distributions")
        elif "ending_capital_account" in field_name:
            guides.append(f"- {field_name}: Part II - Ending capital account")
        elif "line_11" in field_name:
            guides.append(f"- {field_name}: Supplemental schedule or attached statement")
        elif "line_13" in field_name:
            guides.append(f"- {field_name}: Supplemental schedule - deductions")
        elif "line_15" in field_name:
            guides.append(f"- {field_name}: Supplemental schedule - credits")
        elif "line_20" in field_name:
            guides.append(f"- {field_name}: Supplemental schedule - other information")
        else:
            guides.append(f"- {field_name}: {field_name.replace('_', ' ')}")
    
    return "\n".join(guides) + "\n... (and more fields)"


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
            
            # Create the prompt
            prompt = f"""You are extracting data from a K-1 tax form. Extract EXACTLY these fields from the {section_name}:
{', '.join(fields_to_extract)}

FIELD-BY-FIELD EXTRACTION GUIDE:

1. MAIN FORM FIELDS (Part III, First Page Only):
   line_1_ordinary_business_income_loss: "Line 1. Ordinary business income (loss)"
   line_2_net_rental_real_estate_income_loss: "Line 2. Net rental real estate income (loss)"
   line_3_other_rental_income_loss: "Line 3. Other net rental income (loss)"
   line_4a_guaranteed_payments_for_services: "Line 4a. Guaranteed payments for services"
   line_4b_guaranteed_payments_for_capital: "Line 4b. Guaranteed payments for capital"
   line_4c_total_guaranteed_payments: "Line 4c. Total guaranteed payments"
   line_5_interest_income: "Line 5. Interest income" (main form ONLY)
   line_6a_ordinary_dividends: "Line 6a. Ordinary dividends"
   line_6b_qualified_dividends: "Line 6b. Qualified dividends"
   line_7_royalties: "Line 7. Royalties"
   line_9a_net_long_term_capital_gain_loss: "Line 9a. Net long-term capital gain (loss)"
   line_9b_collectibles_28_percent_gain_loss: "Line 9b. Collectibles (28%) gain (loss)"
   line_9c_uncaptured_section_1250_gain: "Line 9c. Unrecaptured section 1250 gain"

2. SUPPLEMENTAL STATEMENT FIELDS (Attached Statements Only):
   line_11ZZ_business_interest_expense: "Business interest expense" (NOT for corporate partners)
   line_11ZZ_ordinary_income_section_475f: "Section 475(f) mark-to-market income"
   line_11ZZ_pfic_qef_income: "PFIC QEF income"
   line_11ZZ_section_988_total: "Section 988 foreign currency gain (loss)"
   line_11ZZ_swap_net_income_loss: "Swap/derivative net income (loss)"
   line_11ZZ_other_income_loss: "Other income (loss)" (must be explicitly labeled)

3. INVESTMENT & PORTFOLIO FIELDS:
   line_13h_investment_interest_investing_schedule_A: ONLY from Schedule A
   line_13h_investment_interest_trading_schedule_E: ONLY from Schedule E
   line_13l_deductions_portfolio_other: "Portfolio deductions - other"
   line_13ZZ_other_deductions_total: ONLY from statements, not main form

4. SPECIAL FIELDS:
   line_15o_backup_withholding: "Backup withholding" or "Form 1099 withholding"
   line_15zz_other_credits: "Other credits" in statements
   line_18b_other_tax_exempt_income: Keep decimals (e.g., "737.00")
   line_18c_nondeductible_expenses: Keep decimals (e.g., "818.00")
   line_20AA_section_704c_information: Must reference "Section 704(c)"
   line_20AG_gross_receipts_section_448_c: Must reference "Section 448(c)"
   line_20N_interest_expense_for_corporate_partners: Business interest for corporate partners ONLY
   line_20V_unrelated_business_taxable_income: Must say "UBTI" or "Unrelated Business"

5. CAPITAL ACCOUNT FIELDS:
   capital_contributions_during_year: From Part II
   withdrawals_and_distributions_cash: From Part II (negative if in parentheses)
   ending_capital_account: From Part II

CRITICAL RULES:
1. EXACT matches only - if label doesn't match word-for-word, use 0
2. Main form fields ONLY from first page Part III
3. Statement fields (11ZZ, 13ZZ) ONLY from attached statements
4. Keep decimals ONLY for tax-exempt fields (18b, 18c)
5. Make parentheses values negative: "(100)" → -100
6. Remove $ and commas: "$1,234" → 1234
7. If field not found or unclear, use 0 for numbers, "" for text
8. NEVER copy values between fields

RESPONSE FORMAT:
1. Return ONLY a valid JSON object
2. No text before or after the JSON
3. Use EXACTLY these field names
4. Example response:
{{
    "line_1_ordinary_business_income_loss": 1234,
    "line_5_interest_income": 75,
    "line_11ZZ_other_deductions_total": 0
}}

{examples}"""
            
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
            response_text = response_data["choices"][0]["message"]["content"]
            
            # Parse JSON
            json_data = json.loads(response_text)
            
            # Validate with Pydantic
            if partial:
                # For partial data, only validate the fields we asked for
                filtered_data = {k: v for k, v in json_data.items() if k in fields_to_extract}
                # Add default values for missing fields
                for field in fields_to_extract:
                    if field not in filtered_data:
                        filtered_data[field] = "" if field.endswith('_logic') or field in ['partnership_name', 'partnership_employer_identification_number'] else 0
                extracted_data = model_class.model_validate(filtered_data) if model_class else filtered_data
            else:
                # For complete data, validate all fields
                extracted_data = model_class.model_validate(json_data) if model_class else json_data
            
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
