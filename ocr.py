from PIL import Image
import pytesseract

# Load the image
image_path = './invoice1.png'
import easyocr

# Initialize the reader
reader = easyocr.Reader(['nl'])

# Perform OCR on the image
result = reader.readtext(image_path)

# Print the results
for detection in result:
    print(detection)
