from ultralytics import YOLO
import cv2
import smtplib
import time
import os
import csv
from datetime import datetime
from email.message import EmailMessage

# ---------------- CONFIG ---------------- #
EMAIL_SENDER = 'jerlinjohn2805@gmail.com'
EMAIL_PASSWORD = 'jbbq veeq cavc inoe'
EMAIL_RECEIVER = 'jerl1nn2805@gmail.com'
ALERT_INTERVAL = 60  # one minute

last_alert_time = 0  # track email + screenshot

# 📁 Create violations folder if missing
os.makedirs("violations", exist_ok=True)

# ---------------- FUNCTIONS ---------------- #

def log_violation(v_type, img_path):
    log_path = "violations.csv"
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    entry = [now, v_type, img_path]

    file_exists = os.path.isfile(log_path)
    with open(log_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(["time", "type", "image"])
        writer.writerow(entry)

def send_helmet_alert(image_path):
    msg = EmailMessage()
    msg['Subject'] = '🚨 Helmet Violation Detected'
    msg['From'] = EMAIL_SENDER
    msg['To'] = EMAIL_RECEIVER
    msg.set_content('A person was detected without a helmet. Screenshot is attached.')

    with open(image_path, 'rb') as f:
        img_data = f.read()
        msg.add_attachment(img_data, maintype='image', subtype='jpeg', filename=os.path.basename(image_path))

    try:
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
            smtp.login(EMAIL_SENDER, EMAIL_PASSWORD)
            smtp.send_message(msg)
            print("✅ Email with screenshot sent!")
    except Exception as e:
        print("❌ Email failed:", e)

# ---------------- PPE DETECTION ---------------- #

model = YOLO("runs/detect/train/weights/best.pt")
CLASSES = model.names
print("CLASSES:", CLASSES)

PERSON_CLASS = next((i for i, name in CLASSES.items() if "person" in name.lower() or "human" in name.lower()), -1)
HELMET_CLASS = next((i for i, name in CLASSES.items() if "helmet" in name.lower() or "hardhat" in name.lower()), -1)
VEST_CLASS = next((i for i, name in CLASSES.items() if "vest" in name.lower()), -1)

if -1 in (PERSON_CLASS, HELMET_CLASS, VEST_CLASS):
    print("❌ Error: Model must detect 'person' (or 'human'), 'helmet', and 'vest'")
    exit()

cap = cv2.VideoCapture(0)
cv2.namedWindow("PPE Detection", cv2.WINDOW_NORMAL)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)[0]

    person_boxes = []
    helmet_boxes = []
    vest_boxes = []

    for det in results.boxes:
        cls_id = int(det.cls.item())
        xyxy = det.xyxy[0].cpu().numpy().astype(int)

        if cls_id == PERSON_CLASS:
            person_boxes.append(xyxy)
        elif cls_id == HELMET_CLASS:
            helmet_boxes.append(xyxy)
        elif cls_id == VEST_CLASS:
            vest_boxes.append(xyxy)

    annotated_frame = frame.copy()

    for person_box in person_boxes:
        px1, py1, px2, py2 = person_box

        has_helmet = any(hx1 > px1 and hy1 > py1 and hx2 < px2 and hy2 < py2 for hx1, hy1, hx2, hy2 in helmet_boxes)
        has_vest = any(vx1 > px1 and vy1 > py1 and vx2 < px2 and vy2 < py2 for vx1, vy1, vx2, vy2 in vest_boxes)

        if not has_helmet:
            color = (0, 0, 255)
            label = "No Helmet!"
            print("⚠️ ALERT: Person not wearing a helmet!")

            current_time = time.time()
            if current_time - last_alert_time > ALERT_INTERVAL:
                timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                image_path = f"violations/helmet_violation_{timestamp}.jpg"
                cv2.imwrite(image_path, annotated_frame)
                log_violation("No Helmet", image_path)
                send_helmet_alert(image_path)
                last_alert_time = current_time

        elif not has_vest:
            color = (0, 165, 255)
            label = "No Vest!"
        else:
            color = (0, 255, 0)
            label = "PPE OK"

        cv2.rectangle(annotated_frame, (px1, py1), (px2, py2), color, 2)
        cv2.putText(annotated_frame, label, (px1, py1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    cv2.imshow("PPE Detection", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
