# Document Image Classification Dataset

## Overview
This dataset contains representative business document images designed for visual document layout understanding and multi-class category classification.

## Classes
The dataset consists of 5 mutually exclusive document categories (30 images per category, 150 images total):
1. **`invoice`**: Billing statements containing vendor/client headers, multi-column itemized tables (Item, Qty, Rate, Total), subtotal boxes, and payment terms.
2. **`receipt`**: Thermal point-of-sale receipts featuring store details, timestamp, item price lists, sales tax calculations, amount due, and bottom barcode lines.
3. **`form`**: Structured application and registration forms containing boxed text input fields, status checkboxes (`[X]`), section borders, and signature confirmation lines.
4. **`letter`**: Official business correspondence with company letterheads, recipient address blocks, formal salutations (*"Dear Distinguished Members,"*), body text paragraphs, and sign-offs.
5. **`report`**: Multi-section executive and technical research reports with dark colored banner headers, executive summaries, embedded statistical bar charts, and footer pagination.

## Specifications
- **Image Resolution:** $224 \times 224$ pixels, 3 channels (RGB)
- **Format:** Lossless PNG
- **Data Splits:**
  - Training: 70% (105 images, 21 per class)
  - Validation: 15% (20 images, 4 per class)
  - Held-out Test: 15% (25 images, 5 per class)
- **License:** Open Access / MIT-compatible synthetic representation derived from common open-source document layout taxonomy (compatible with RVL-CDIP and Tobacco3482 benchmark classifications).
