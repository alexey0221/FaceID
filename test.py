from pathlib import Path
import cv2
import os

# pathlib сам разберется с разделителями
image_path = "Люди/1/Abbington.jpg"
# path = Path(image_path)  # Преобразует в правильный формат
image = cv2.imread(image_path)  # ✅ Работае
print(image)