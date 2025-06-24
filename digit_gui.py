import tkinter as tk
from PIL import Image, ImageDraw, ImageOps
import numpy as np
import tensorflow as tf

# Load trained model
model = tf.keras.models.load_model("digit_cnn_model.h5")

# Create GUI window
window = tk.Tk()
window.title("Digit Recognizer")
canvas_width = 280
canvas_height = 280

# Create canvas
canvas = tk.Canvas(window, width=canvas_width, height=canvas_height, bg='white')
canvas.pack()

# PIL image for drawing
image1 = Image.new("L", (canvas_width, canvas_height), 'white')
draw = ImageDraw.Draw(image1)

# Drawing function
def draw_lines(event):
    x, y = event.x, event.y
    r = 4  # Thinner stroke
    canvas.create_oval(x - r, y - r, x + r, y + r, fill='black')
    draw.ellipse([x - r, y - r, x + r, y + r], fill='black')

canvas.bind("<B1-Motion>", draw_lines)

# Prediction function
def predict_digit():
    # Resize and invert the canvas image
    img = image1.resize((28, 28))
    img = ImageOps.invert(img)

    # Convert to numpy array
    img_array = np.array(img)

    # Find bounding box
    bbox = Image.fromarray(img_array).getbbox()
    if bbox:
        img_array = Image.fromarray(img_array).crop(bbox).resize((20, 20), Image.LANCZOS)

    # Paste into 28x28 image centered
    new_img = Image.new('L', (28, 28), 0)
    new_img.paste(img_array, (4, 4))

    # Normalize and reshape
    img_array = np.array(new_img).astype('float32') / 255.0
    img_array = img_array.reshape(1, 28, 28, 1)

    # Predict
    prediction = model.predict(img_array)
    digit = np.argmax(prediction)

    result_label.config(text=f"Predicted Digit: {digit}")

# Clear canvas
def clear_canvas():
    canvas.delete("all")
    draw.rectangle([0, 0, canvas_width, canvas_height], fill='white')
    result_label.config(text="")

# Buttons and labels
predict_button = tk.Button(window, text="Predict", command=predict_digit)
predict_button.pack()

clear_button = tk.Button(window, text="Clear", command=clear_canvas)
clear_button.pack()

result_label = tk.Label(window, text="", font=("Helvetica", 20))
result_label.pack()

# Run the app
window.mainloop()
