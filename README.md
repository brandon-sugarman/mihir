# Abacus Interview Challenge

Welcome to the Abacus Interview Challenge! This is your chance to demonstrate how you'd approach building a robust data extraction pipeline.

## 🗂️ What’s in this repo?

We've provided the following files to get you started:

- **`pdfs/`**  
  This folder contains **three tax forms** in PDF format. These are your input files.

- **`eval_set.csv`**  
  This CSV contains the **expected field names and extracted values** for each PDF. Each column corresponds to one of the provided K-1 forms.

- **`main.py`**  
  This is where you’ll write your **extraction logic**. Your pipeline should process each PDF and output structured data in accordance with the provided schema.

- **`pydantic_model.py`**  
  This defines a **Pydantic schema** for the two key sections we care about:
  - The **cover page**
  - The **footnotes** (the supplemental materials following the cover page)  
  
  It also includes a helper function to **chunk the schema** for easier parsing.

## 🎯 Your Goal

Build a data extraction pipeline that:

1. **Reads each PDF** from the `pdfs/` directory.
2. **Extracts structured data** for the cover page and footnotes, matching the Pydantic schema in `pydantic_model.py`.
3. **Compares** your extracted data against the reference values in `eval_set.csv`.

Your logic should live in `main.py`.

## 🚀 Setup and Running

### Prerequisites

1. **Install dependencies**:
   ```bash
   uv sync
   ```

2. **Set your API key** (required):
   ```bash
   export OPENROUTER_API_KEY='your-api-key-here'
   ```

### Running the Pipeline

```bash
uv run python main.py
```

The script will:
1. Use [Marker](https://github.com/datalab-to/marker) to convert PDFs to structured markdown
2. Extract structured data using AI (Grok-4-Fast via OpenRouter)
3. Validate results against the evaluation set

**Note**: On first run, Marker will download its models (~2GB). This is a one-time setup.

## 🔧 Technical Approach

This solution uses a two-stage pipeline:

1. **PDF to Markdown Conversion** - [Marker](https://github.com/datalab-to/marker) converts PDFs to high-quality markdown, preserving:
   - Table structures
   - Form layouts
   - Reading order
   - Achieves 95.6% accuracy on financial documents

2. **Structured Data Extraction** - An LLM extracts specific fields from the markdown:
   - Uses the Pydantic schema to guide extraction
   - Handles complex tax form structures and supplemental schedules
   - Falls back to basic extraction if Marker fails

## ✅ Evaluation

We'll assess your solution based on:

- **Accuracy** of extraction
- **Code clarity and structure**
- **Use of the provided schema**
- **Comparison results against the eval set**

Feel free to use any libraries or tooling you like—just be sure to document what you use.

---

Good luck, and happy parsing! 🧮

— The Abacus Team
