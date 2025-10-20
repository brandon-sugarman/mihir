"""Evaluation utilities for comparing extracted results with ground truth."""

from typing import Dict, Any, Tuple
from pathlib import Path

from .config import EVAL_SET_FILE
from .validator import clean_number
from .pydantic_model import k1_cover_page, k1_federal_footnotes


def load_eval_set() -> Dict[str, Dict[str, Any]]:
    """Load the evaluation set from CSV.
    
    Returns:
        Dictionary mapping document names to field values
    """
    eval_data = {}
    with open(EVAL_SET_FILE, 'r') as f:
        lines = f.readlines()
        headers = lines[0].strip().split(',')
        for line in lines[1:]:
            parts = line.strip().split(',')
            if len(parts) < 2:
                continue
            field_name = parts[0]
            for i, doc_name in enumerate(headers[1:], 1):
                if i < len(parts):
                    if doc_name not in eval_data:
                        eval_data[doc_name] = {}
                    eval_data[doc_name][field_name] = parts[i]
    return eval_data


def compare_with_eval_set(results: Dict[str, Tuple[k1_cover_page, k1_federal_footnotes]]) -> None:
    """Compare extracted results with evaluation set (non-zero fields only).
    
    Args:
        results: Dictionary mapping PDF names to extracted data tuples
    """
    eval_data = load_eval_set()
    total_fields, correct_fields = 0, 0
    
    print("\n" + "="*80)
    print("COMPARISON WITH EVALUATION SET (Non-Zero Fields Only)")
    print("="*80)
    
    for pdf_name, (cover_page, footnotes) in results.items():
        if pdf_name not in eval_data:
            continue
        
        print(f"\n{pdf_name}:")
        print("-" * 80)
        
        doc_correct, doc_total = 0, 0
        expected = eval_data[pdf_name]
        
        # Check cover page fields
        for field_name in k1_cover_page.model_fields:
            extracted = getattr(cover_page, field_name)
            expected_value = expected.get(field_name, '')
            
            if isinstance(extracted, int) and (extracted != 0 or clean_number(expected_value) != 0):
                doc_total += 1
                total_fields += 1
                match = extracted == clean_number(expected_value)
                if match:
                    doc_correct += 1
                    correct_fields += 1
                print(f" {'✓' if match else '✗'} {field_name}: extracted={extracted}, expected={expected_value}")
        
        # Check footnotes fields
        for field_name in k1_federal_footnotes.model_fields:
            extracted = getattr(footnotes, field_name)
            expected_value = expected.get(field_name, '')
            
            if isinstance(extracted, int) and (extracted != 0 or clean_number(expected_value) != 0):
                doc_total += 1
                total_fields += 1
                match = extracted == clean_number(expected_value)
                if match:
                    doc_correct += 1
                    correct_fields += 1
                print(f" {'✓' if match else '✗'} {field_name}: extracted={extracted}, expected={expected_value}")
        
        accuracy = (doc_correct / doc_total * 100) if doc_total > 0 else 0
        print(f"\n Document Accuracy: {doc_correct}/{doc_total} ({accuracy:.1f}%)")
    
    overall_accuracy = (correct_fields / total_fields * 100) if total_fields > 0 else 0
    print("\n" + "="*80)
    print(f"OVERALL ACCURACY: {correct_fields}/{total_fields} ({overall_accuracy:.1f}%)")
    print("="*80)


def generate_evaluation_report(
    results: Dict[str, Tuple[k1_cover_page, k1_federal_footnotes]], 
    output_dir: str = "evaluation_reports"
) -> None:
    """Generate a detailed evaluation report.
    
    Args:
        results: Dictionary mapping PDF names to extracted data tuples
        output_dir: Directory to save the report
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    eval_data = load_eval_set()
    
    for pdf_name, (cover_page, footnotes) in results.items():
        if pdf_name not in eval_data:
            continue
            
        report_file = output_path / f"{pdf_name}_evaluation.txt"
        
        with open(report_file, 'w') as f:
            f.write(f"Evaluation Report for {pdf_name}\n")
            f.write("=" * 50 + "\n\n")
            
            expected = eval_data[pdf_name]
            doc_correct, doc_total = 0, 0
            
            # Cover page fields
            f.write("COVER PAGE FIELDS:\n")
            f.write("-" * 30 + "\n")
            for field_name in k1_cover_page.model_fields:
                extracted = getattr(cover_page, field_name)
                expected_value = expected.get(field_name, '')
                
                if isinstance(extracted, int) and (extracted != 0 or clean_number(expected_value) != 0):
                    doc_total += 1
                    match = extracted == clean_number(expected_value)
                    if match:
                        doc_correct += 1
                    f.write(f"{'✓' if match else '✗'} {field_name}: {extracted} (expected: {expected_value})\n")
            
            # Footnotes fields
            f.write("\nFOOTNOTES FIELDS:\n")
            f.write("-" * 30 + "\n")
            for field_name in k1_federal_footnotes.model_fields:
                extracted = getattr(footnotes, field_name)
                expected_value = expected.get(field_name, '')
                
                if isinstance(extracted, int) and (extracted != 0 or clean_number(expected_value) != 0):
                    doc_total += 1
                    match = extracted == clean_number(expected_value)
                    if match:
                        doc_correct += 1
                    f.write(f"{'✓' if match else '✗'} {field_name}: {extracted} (expected: {expected_value})\n")
            
            accuracy = (doc_correct / doc_total * 100) if doc_total > 0 else 0
            f.write(f"\nAccuracy: {doc_correct}/{doc_total} ({accuracy:.1f}%)\n")
        
        print(f"Evaluation report saved to: {report_file}")
