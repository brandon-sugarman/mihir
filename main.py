#!/usr/bin/env python3
"""
PDF Data Extraction Pipeline for K-1 Tax Forms
Uses vision models to extract structured data directly from PDF images.
"""

import os
import sys
from pathlib import Path
from typing import Dict, Tuple

# Ensure `src/` is on the import path so `abacusai` can be imported without installation
project_root = Path(__file__).parent
src_path = project_root / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from abacusai.config import PDFS_DIR
from abacusai.extractor import extract_k1_data
from abacusai.evaluator import compare_with_eval_set
from abacusai.pydantic_model import k1_cover_page, k1_federal_footnotes


def process_all_pdfs() -> Dict[str, Tuple[k1_cover_page, k1_federal_footnotes]]:
    """Process all PDFs in the pdfs directory.
    
    Returns:
        Dictionary mapping PDF filenames to extracted data tuples
    """
    pdf_dir = Path(PDFS_DIR)
    results = {}
    
    for pdf_file in sorted(pdf_dir.glob("*.pdf")):
        print(f"\nProcessing {pdf_file.name}...")
        try:
            cover_page, footnotes = extract_k1_data(str(pdf_file))
            print(f" ✓ Successfully extracted all data from {pdf_file.name}")
            results[pdf_file.name] = (cover_page, footnotes)
        except Exception as e:
            print(f" ✗ Error processing {pdf_file.name}: {e}")
            import traceback
            traceback.print_exc()
    
    return results


def main():
    """Main execution function."""
    print("K-1 Tax Form Data Extraction Pipeline")
    print("=" * 80)
    
    if not os.environ.get("OPENROUTER_API_KEY"):
        print("ERROR: OPENROUTER_API_KEY environment variable not set!")
        print("Please set it with: export OPENROUTER_API_KEY='your-key-here'")
        return
    
    results = process_all_pdfs()
    
    if results:
        compare_with_eval_set(results)
    
    print("\n" + "="*80)
    print("EXTRACTION COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()