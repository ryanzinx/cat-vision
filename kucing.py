import cv2
import numpy as np

def apply_cat_vision(frame):
    # Konversi frame ke grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Tingkatkan kontras
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    contrast_enhanced = clahe.apply(gray)
    
    # Terapkan efek blur untuk mensimulasikan penglihatan yang kurang tajam
    blurred = cv2.GaussianBlur(contrast_enhanced, (5, 5), 0)
    
    # Terapkan threshold adaptif untuk mensimulasikan penglihatan dalam gelap
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    
    # Gabungkan gambar asli dengan hasil threshold
    result = cv2.addWeighted(frame, 0.7, cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR), 0.3, 0)
    
    return result

# Buka kamera
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Terapkan efek penglihatan mata kucing
    cat_vision = apply_cat_vision(frame)
    
    # Tampilkan hasil
    cv2.imshow('Cat Vision', cat_vision)
    
    # Tekan 'q' untuk keluar
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Lepaskan kamera dan tutup jendela
cap.release()
cv2.destroyAllWindows()