import cv2
import easyocr
from gtts import gTTS
import os
from playsound import playsound
import re
import time
from matplotlib import pyplot as plt
from pythainlp.tokenize import word_tokenize
from pythainlp.spell import correct
from inference_sdk import InferenceHTTPClient  # Roboflow SDK

# เรียกใช้งาน OCR
reader = easyocr.Reader(['th', 'en'])

# สร้าง client ของ Roboflow
CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="uCVdI95U28toSEOyVXe6"  # แทนที่ด้วย API จริง
)

def split_text_by_language(text):
    thai = []
    english = []
    for line in text.splitlines():
        if re.search(r'[ก-๙]', line):
            thai.append(line)
        else:
            english.append(line)
    return '\n'.join(thai), '\n'.join(english)

def clean_text(text):
    text = re.sub(r'[^\w\sก-๙]', '', text)
    words = word_tokenize(text, engine='newmm')
    corrected_words = [correct(w) for w in words]
    cleaned_text = " ".join(corrected_words)
    return cleaned_text

def speak_text(thai_text, eng_text):
    if thai_text and thai_text.strip():
        cleaned_th = clean_text(thai_text)
        if cleaned_th.strip():
            tts_th = gTTS(text=cleaned_th, lang='th')
            tts_th.save("temp_th.mp3")
            playsound("temp_th.mp3")
            os.remove("temp_th.mp3")

    if eng_text and eng_text.strip():
        if eng_text.strip():
            tts_en = gTTS(text=eng_text, lang='en')
            tts_en.save("temp_en.mp3")
            playsound("temp_en.mp3")
            os.remove("temp_en.mp3")

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ไม่สามารถเปิดกล้องได้")
        return

    print("เริ่ม OCR อัตโนมัติทุก 2 วินาที (กด Ctrl+C เพื่อหยุด)")

    last_text = ""

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("ไม่สามารถรับภาพจากกล้องได้")
                break

            # บันทึกภาพจากกล้อง
            cv2.imwrite("frame.jpg", frame)

            # เรียกใช้ Roboflow เพื่อตรวจวัตถุ
            try:
                roboflow_result = CLIENT.infer("frame.jpg", model_id="football-players-detection-3zvbc/12")
                print("🎯 ผลลัพธ์จาก Roboflow:")
                print(roboflow_result)
            except Exception as e:
                print("❌ เกิดข้อผิดพลาดจาก Roboflow:", e)

            # ใช้ EasyOCR ตรวจข้อความ
            result = reader.readtext("frame.jpg")
            full_text = '\n'.join([d[1] for d in result])

            if full_text.strip() and full_text != last_text:
                print("📄 ข้อความใหม่ที่ตรวจพบ:\n", full_text)
                last_text = full_text
                thai, eng = split_text_by_language(full_text)
                speak_text(thai, eng)
            else:
                print("⏸️ ข้อความเดิม หรือไม่มีข้อความใหม่")

            time.sleep(2)

    except KeyboardInterrupt:
        print("🛑 หยุดโปรแกรมด้วย Ctrl+C")

    cap.release()

if __name__ == "__main__":
    main()
