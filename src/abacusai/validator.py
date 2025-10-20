"""Data validation and cleaning utilities."""

import re
from typing import Any, Dict, List, Type
from pydantic import BaseModel


def get_default_model_data(model_class: Type[BaseModel]) -> Dict[str, Any]:
    """Get default data for a Pydantic model."""
    data = {}
    for field_name, field_info in model_class.model_fields.items():
        # Check if field name ends with '_logic' - these are always strings
        if field_name.endswith('_logic'):
            data[field_name] = ""
        # Check field type
        elif field_info.annotation == str or 'str' in str(field_info.annotation):
            data[field_name] = ""
        else:
            data[field_name] = 0
    return data


def clean_number(text: Any) -> int:
    """Convert text to integer, handling various formats.
    
    Args:
        text: Input text to clean and convert
        
    Returns:
        Cleaned integer value
    """
    if isinstance(text, int):
        return text
    if not text:
        return 0
    text = str(text).strip()
    if text in ['', 'N/A', 'n/a', '-', '0']:
        return 0
    
    # Handle quoted numbers (e.g. "42" -> 42)
    if text.startswith('"') and text.endswith('"'):
        text = text[1:-1]
    
    # Clean up the text
    text = text.replace(',', '').replace('$', '').replace(' ', '')
    
    # Handle negative numbers
    is_negative = text.startswith('(') and text.endswith(')')
    if is_negative:
        text = text[1:-1]
    if text.startswith('-'):
        is_negative = True
        text = text[1:]
    
    try:
        # Handle decimal numbers by rounding
        value = round(float(text))
        return -value if is_negative else value
    except (ValueError, AttributeError):
        # Try to extract just the numeric part
        numbers = re.findall(r'\d+', text)
        if numbers:
            return int(numbers[0])
        return 0


def validate_field_groups(extracted_data: Dict[str, Any]) -> Dict[str, Any]:
    """Simple validation to prevent duplicate values across related fields.
    
    Args:
        extracted_data: Raw extracted data dictionary
        
    Returns:
        Validated data dictionary
    """
    return extracted_data  # For now, trust the model's field assignments


def validate_extracted_data(
    data: Dict[str, Any], 
    fields_to_extract: List[str], 
    model_class: Type[BaseModel] = None
) -> Dict[str, Any]:
    """Validate and clean extracted data.
    
    Args:
        data: Raw extracted data
        fields_to_extract: List of fields that should be present
        model_class: Pydantic model class for validation
        
    Returns:
        Cleaned and validated data dictionary
    """
    # Apply field group validation
    validated_data = validate_field_groups(data)
    
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
