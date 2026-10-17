"""
generate_dataset.py
===================
Generates a representative, reproducible document image dataset for
Document Understanding and Classification across five distinct categories:
1. invoice
2. receipt
3. form
4. letter
5. report

Uses PIL drawing primitives to render realistic structural layouts,
tables, typography, form boxes, barcodes, and section headers.
"""

import os
import random
from PIL import Image, ImageDraw, ImageFont

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "documents")
CATEGORIES = ["invoice", "receipt", "form", "letter", "report"]
SAMPLES_PER_CLASS = 30
IMG_SIZE = (224, 224)


def draw_invoice(draw, idx):
    random.seed(1000 + idx)
    # Background
    draw.rectangle([0, 0, 224, 224], fill="#fcfcfc")
    # Top header bar / Company block
    draw.rectangle([10, 10, 80, 22], fill="#1e3a8a")
    draw.text((15, 12), "COMPANY", fill="#ffffff")
    draw.text((140, 12), "INVOICE", fill="#1e3a8a")
    draw.line([10, 32, 214, 32], fill="#cbd5e1", width=1)

    # Bill to & Info
    draw.text((12, 38), f"Invoice #: INV-2026-{idx:03d}", fill="#334155")
    draw.text((12, 50), "Billed To: ACME Corp LLC", fill="#64748b")
    draw.text((140, 38), "Date: 2026-10-15", fill="#64748b")

    # Table Header
    draw.rectangle([10, 68, 214, 82], fill="#e2e8f0")
    draw.text((15, 71), "ITEM", fill="#0f172a")
    draw.text((95, 71), "QTY", fill="#0f172a")
    draw.text((135, 71), "RATE", fill="#0f172a")
    draw.text((175, 71), "TOTAL", fill="#0f172a")

    # Table Rows
    y = 86
    items = ["Consulting", "Server Hosting", "API License", "Data Pipeline", "Support"]
    for row in range(4):
        item_name = items[(idx + row) % len(items)]
        qty = random.randint(1, 10)
        rate = random.randint(50, 400)
        draw.text((15, y), item_name[:12], fill="#334155")
        draw.text((100, y), str(qty), fill="#334155")
        draw.text((135, y), f"${rate}", fill="#334155")
        draw.text((175, y), f"${qty*rate}", fill="#334155")
        draw.line([10, y + 14, 214, y + 14], fill="#f1f5f9", width=1)
        y += 18

    # Table grid borders
    draw.rectangle([10, 68, 214, y], outline="#94a3b8", width=1)
    draw.line([85, 68, 85, y], fill="#94a3b8", width=1)
    draw.line([125, 68, 125, y], fill="#94a3b8", width=1)
    draw.line([165, 68, 165, y], fill="#94a3b8", width=1)

    # Total Box
    draw.rectangle([130, y + 8, 214, y + 36], fill="#f8fafc", outline="#cbd5e1", width=1)
    draw.text((135, y + 12), "Subtotal:", fill="#64748b")
    draw.text((180, y + 12), f"${random.randint(1200, 4500)}", fill="#334155")
    draw.text((135, y + 24), "TOTAL:", fill="#0f172a")
    draw.text((180, y + 24), f"${random.randint(1300, 4800)}", fill="#1e3a8a")

    # Payment footer
    draw.text((12, 205), "Thank you for your business. Terms: Net 30", fill="#94a3b8")


def draw_receipt(draw, idx):
    random.seed(2000 + idx)
    # Background (off-white receipt paper)
    draw.rectangle([0, 0, 224, 224], fill="#f9f9f6")
    # Receipt body centered narrow margins
    draw.rectangle([25, 6, 199, 218], fill="#ffffff", outline="#e2e8f0")

    # Store header
    draw.text((70, 12), "** STORE MART **", fill="#1e293b")
    draw.text((65, 24), "123 Market St, Metro", fill="#64748b")
    draw.text((75, 34), "TEL: 555-0199", fill="#64748b")
    draw.line([35, 48, 189, 48], fill="#334155", width=1)

    draw.text((35, 52), f"REC: #{idx+1040:05d}   REG: 04", fill="#475569")
    draw.text((35, 62), "DATE: 2026-10-16 14:22", fill="#475569")
    draw.line([35, 74, 189, 74], fill="#94a3b8", width=1)

    # Item rows
    items = ["Milk Organic", "Bread Whole", "Eggs 12pk", "Apples 1kg", "Coffee Beans", "Cereal Box", "Butter Salt"]
    y = 78
    total = 0.0
    for r in range(5):
        it = items[(idx + r) % len(items)]
        price = round(random.uniform(2.5, 14.0), 2)
        total += price
        draw.text((35, y), it[:13], fill="#1e293b")
        draw.text((155, y), f"${price:.2f}", fill="#1e293b")
        y += 14

    draw.line([35, y + 4, 189, y + 4], fill="#334155", width=1)
    draw.text((35, y + 8), "SUBTOTAL", fill="#475569")
    draw.text((155, y + 8), f"${total:.2f}", fill="#475569")
    draw.text((35, y + 20), "TAX (8%)", fill="#475569")
    draw.text((155, y + 20), f"${total*0.08:.2f}", fill="#475569")
    draw.text((35, y + 34), "AMOUNT DUE", fill="#0f172a")
    draw.text((150, y + 34), f"${total*1.08:.2f}", fill="#0f172a")

    # Barcode representation
    bc_y = 192
    draw.line([35, bc_y - 4, 189, bc_y - 4], fill="#cbd5e1", width=1)
    bx = 45
    while bx < 180:
        bar_w = random.choice([1, 2, 3])
        draw.line([bx, bc_y, bx, bc_y + 16], fill="#0f172a", width=bar_w)
        bx += bar_w + random.choice([1, 2, 3])


def draw_form(draw, idx):
    random.seed(3000 + idx)
    draw.rectangle([0, 0, 224, 224], fill="#ffffff")

    # Form header box
    draw.rectangle([10, 10, 214, 30], fill="#f1f5f9", outline="#0284c7", width=1)
    draw.text((20, 14), f"APPLICATION & VERIFICATION FORM #{idx:02d}", fill="#0369a1")

    # Section 1: Personal info
    draw.text((12, 36), "1. APPLICANT DETAILS", fill="#0f172a")
    draw.line([12, 48, 214, 48], fill="#e2e8f0", width=1)

    draw.text((12, 54), "Full Name:", fill="#475569")
    draw.rectangle([65, 52, 214, 66], fill="#f8fafc", outline="#cbd5e1")
    draw.text((70, 54), "Johnathan Doe", fill="#334155")

    draw.text((12, 72), "Email ID:", fill="#475569")
    draw.rectangle([65, 70, 214, 84], fill="#f8fafc", outline="#cbd5e1")
    draw.text((70, 72), "j.doe@example.org", fill="#334155")

    # Checkbox options
    draw.text((12, 92), "Employment Status:", fill="#475569")
    # Checkbox 1
    draw.rectangle([115, 92, 123, 100], outline="#0f172a", width=1)
    draw.text((117, 91), "X", fill="#0284c7")
    draw.text((127, 92), "Full-Time", fill="#334155")
    # Checkbox 2
    draw.rectangle([170, 92, 178, 100], outline="#0f172a", width=1)
    draw.text((182, 92), "Contract", fill="#334155")

    # Section 2: Address
    draw.text((12, 110), "2. RESIDENTIAL ADDRESS", fill="#0f172a")
    draw.line([12, 122, 214, 122], fill="#e2e8f0", width=1)

    draw.rectangle([12, 126, 214, 150], fill="#f8fafc", outline="#cbd5e1")
    draw.text((16, 130), "452 Elm Street, Suite 800", fill="#334155")
    draw.text((16, 140), "Austin, TX 78701, United States", fill="#64748b")

    # Declaration checkboxes
    draw.rectangle([12, 160, 20, 168], outline="#0f172a", width=1)
    draw.text((13, 159), "X", fill="#0f172a")
    draw.text((25, 160), "I certify all statements are true and accurate.", fill="#475569")

    draw.rectangle([12, 174, 20, 182], outline="#0f172a", width=1)
    draw.text((25, 174), "I agree to the terms and privacy regulations.", fill="#475569")

    # Signature line
    draw.line([120, 204, 210, 204], fill="#64748b", width=1)
    draw.text((120, 192), "J. Doe", fill="#1e3a8a")
    draw.text((135, 208), "Authorized Signature", fill="#94a3b8")


def draw_letter(draw, idx):
    random.seed(4000 + idx)
    draw.rectangle([0, 0, 224, 224], fill="#ffffff")

    # Letterhead / Header
    draw.text((15, 12), "HARVARD BUSINESS ASSOCIATES", fill="#475569")
    draw.text((15, 22), "100 University Avenue, Cambridge, MA", fill="#94a3b8")
    draw.line([15, 34, 209, 34], fill="#cbd5e1", width=1)

    draw.text((15, 40), f"October {idx % 28 + 1}, 2026", fill="#64748b")

    # Recipient
    draw.text((15, 54), "Board of Directors", fill="#1e293b")
    draw.text((15, 64), "Apex Global Solutions Inc.", fill="#475569")

    # Salutation
    draw.text((15, 80), "Dear Distinguished Members,", fill="#1e293b")

    # Body paragraphs (justified horizontal lines & text)
    body_lines = [
        "I am writing to provide you with the formal update",
        "regarding our operational performance and initiatives",
        "completed over the preceding fiscal quarter.",
        "",
        "Our strategic realignment has achieved steady growth",
        "and reinforced core compliance standards. We look",
        "forward to presenting further findings at our summit.",
    ]
    y = 96
    for line in body_lines:
        if line:
            draw.text((15, y), line, fill="#334155")
        y += 11

    # Closing
    draw.text((15, 180), "Sincerely yours,", fill="#475569")
    draw.text((15, 194), "Eleanor Vance", fill="#1e3a8a")
    draw.text((15, 206), "Managing Partner", fill="#94a3b8")


def draw_report(draw, idx):
    random.seed(5000 + idx)
    draw.rectangle([0, 0, 224, 224], fill="#ffffff")

    # Top banner
    draw.rectangle([0, 0, 224, 32], fill="#0f172a")
    draw.text((12, 6), "ANNUAL RESEARCH REPORT", fill="#38bdf8")
    draw.text((12, 18), "Artificial Intelligence & Automation Insights", fill="#94a3b8")

    # Section 1
    draw.rectangle([12, 40, 212, 54], fill="#f1f5f9")
    draw.text((16, 42), "1. EXECUTIVE BENCHMARK SUMMARY", fill="#0f172a")

    draw.text((14, 58), "Evaluation across multi-modal architectures indicates", fill="#475569")
    draw.text((14, 68), "a 34% reduction in inference latency under quantization.", fill="#475569")

    # Chart representation (embedded bar chart)
    draw.rectangle([12, 84, 212, 160], fill="#f8fafc", outline="#e2e8f0")
    draw.text((16, 88), "Figure 1: Benchmark Throughput (req/s)", fill="#0284c7")

    # Draw chart bars
    draw.line([25, 104, 25, 150], fill="#64748b", width=1)
    draw.line([25, 150, 200, 150], fill="#64748b", width=1)

    bar_heights = [18, 28, 42, 34, 40]
    bar_colors = ["#38bdf8", "#0284c7", "#0369a1", "#075985", "#0c4a6e"]
    for i, h in enumerate(bar_heights):
        bx = 35 + i * 32
        draw.rectangle([bx, 150 - h, bx + 18, 150], fill=bar_colors[i])
        draw.text((bx + 2, 152), f"Q{i+1}", fill="#64748b")

    # Section 2
    draw.rectangle([12, 168, 212, 182], fill="#f1f5f9")
    draw.text((16, 170), "2. RECOMMENDATIONS & NEXT STEPS", fill="#0f172a")

    draw.text((14, 186), "Further scaling of model parallelism is recommended", fill="#475569")
    draw.text((14, 196), "for distributed multi-node production deployment.", fill="#475569")

    # Page footer
    draw.text((12, 212), "Confidential Document | Page 1 of 12", fill="#cbd5e1")


def main():
    print("Generating Document Classification Dataset...")
    os.makedirs(DATA_DIR, exist_ok=True)
    draw_functions = {
        "invoice": draw_invoice,
        "receipt": draw_receipt,
        "form": draw_form,
        "letter": draw_letter,
        "report": draw_report,
    }

    total_generated = 0
    for cat in CATEGORIES:
        cat_dir = os.path.join(DATA_DIR, cat)
        os.makedirs(cat_dir, exist_ok=True)
        draw_fn = draw_functions[cat]

        for i in range(SAMPLES_PER_CLASS):
            img = Image.new("RGB", IMG_SIZE, color="#ffffff")
            draw = ImageDraw.Draw(img)
            draw_fn(draw, i)
            file_path = os.path.join(cat_dir, f"doc_{i:03d}.png")
            img.save(file_path, "PNG")
            total_generated += 1

        print(f"  [OK] Generated {SAMPLES_PER_CLASS} samples for category: {cat}")

    print(f"\nSuccessfully generated {total_generated} document images in {DATA_DIR}!")


if __name__ == "__main__":
    main()
