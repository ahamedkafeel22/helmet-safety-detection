import cv2
import csv
import json
import os
from ultralytics import YOLO
from datetime import datetime

# ── CONFIG ────────────────────────────────────────────
MODEL_PATH = r"C:\Users\syedk\Documents\Self Projects\Project 4\best.pt"
IMAGE_FOLDER = r"C:\Users\syedk\Documents\Self Projects\Project 4\Hard-Hat-Workers-1\test\images"
OUTPUT_FOLDER = r"C:\Users\syedk\Documents\Self Projects\Project 4\output"
CSV_OUTPUT = r"C:\Users\syedk\Documents\Self Projects\Project 4\safety_report.csv"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ── LOAD MODEL ────────────────────────────────────────
model = YOLO(MODEL_PATH)
CLASS_NAMES = {0: 'head', 1: 'helmet', 2: 'person'}

# ── PROCESS IMAGES ────────────────────────────────────
results_summary = []
total_helmets = 0
total_heads = 0
total_persons = 0

image_files = [f for f in os.listdir(IMAGE_FOLDER) if f.endswith(('.jpg', '.jpeg', '.png'))]
print(f"Processing {len(image_files)} images...")

for img_file in image_files:
    img_path = os.path.join(IMAGE_FOLDER, img_file)
    img = cv2.imread(img_path)

    # Run detection
    results = model(img_path, conf=0.5)[0]

    # Count per class
    helmet_count = 0
    head_count = 0
    person_count = 0

    for box in results.boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        if cls == 0:  # head (no helmet)
            head_count += 1
            color = (0, 0, 255)   # Red
            label = f"NO HELMET {conf:.2f}"
        elif cls == 1:  # helmet
            helmet_count += 1
            color = (0, 255, 0)   # Green
            label = f"HELMET {conf:.2f}"
        else:  # person
            person_count += 1
            color = (255, 165, 0) # Orange
            label = f"PERSON {conf:.2f}"

        # Draw bounding box
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img, label, (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Compliance calculation
    total_workers = helmet_count + head_count
    compliance = (helmet_count / total_workers * 100) if total_workers > 0 else 100.0

    # Add compliance text on image
    status = "SAFE" if compliance == 100 else "VIOLATION"
    status_color = (0, 255, 0) if compliance == 100 else (0, 0, 255)
    cv2.putText(img, f"{status} | Compliance: {compliance:.1f}%",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)

    # Save annotated image
    out_path = os.path.join(OUTPUT_FOLDER, img_file)
    cv2.imwrite(out_path, img)

    # Track totals
    total_helmets += helmet_count
    total_heads += head_count
    total_persons += person_count

    results_summary.append({
        'image': img_file,
        'helmet_count': helmet_count,
        'head_count': head_count,
        'person_count': person_count,
        'compliance_%': round(compliance, 2),
        'status': status,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    })

# ── OVERALL STATS ─────────────────────────────────────
total_workers = total_helmets + total_heads
overall_compliance = (total_helmets / total_workers * 100) if total_workers > 0 else 100.0

print(f"\n{'='*50}")
print(f"📊 SAFETY ANALYTICS REPORT")
print(f"{'='*50}")
print(f"Total Images Processed : {len(image_files)}")
print(f"Total Helmets Detected : {total_helmets}")
print(f"Total Violations (No Helmet): {total_heads}")
print(f"Overall Compliance Rate: {overall_compliance:.2f}%")
print(f"{'='*50}")

# ── EXPORT CSV ────────────────────────────────────────
with open(CSV_OUTPUT, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=results_summary[0].keys())
    writer.writeheader()
    writer.writerows(results_summary)

print(f"✅ CSV report saved: {CSV_OUTPUT}")
print(f"✅ Annotated images saved: {OUTPUT_FOLDER}")
print(f"\n🎉 Detection Complete!")