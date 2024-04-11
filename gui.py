import  tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
import torch
import torchvision.transforms as transforms
from model import CNN  # model.py containing the CNN model

# Define the transform for preprocessing the image
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize pixel values to [-1, 1]
])

# Load the trained model
model = CNN()
model.load_state_dict(torch.load('traffic_sign_model.pth'))  # Load the trained model file

# Define the class labels
class_labels = ['bump','children_crossing','keep_left','keep_right','parking', 'speed_limit_20','speed_limit_80','stop','turn_left','turn_right']

def classify_image(image_path):
    # Load and preprocess the image
    image = Image.open(image_path)
    image = image.convert("RGB")  # Ensure that the image has 3 channels (RGB)
    image = transform(image).unsqueeze(0)  # Add batch dimension

    try:
        # Perform inference
        with torch.no_grad():
            model.eval()
            outputs = model(image)
            _, predicted = torch.max(outputs, 1)
            class_idx = predicted.item()
            print(class_idx,class_labels[class_idx])

        # Display the result
        result_label.config(text=f'Predicted Class: {class_labels[class_idx]}')
    except Exception as e:
        result_label.config(text=f"Error: {e}")

def browse_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        # Display the selected image
        image = Image.open(file_path)
        image = image.resize((300, 300))  # Resize image for display
        photo = ImageTk.PhotoImage(image)
        image_label.config(image=photo)
        image_label.image = photo  # Keep a reference to the image to prevent garbage collection
        
        classify_image(file_path)

# Create the main application window
root = tk.Tk()
root.title('Traffic Sign Classification')
root.geometry("600x400")

# Create widgets
tk.Label(root,text="Traffic Sign recognition",height=3,font=("poppins",14,"bold"),anchor='center').grid(column=0,row=0,columnspan=2)
browse_button = tk.Button(root, text='Browse Image',font=("times",12), command=browse_image)
image_label = tk.Label(root)
result_label = tk.Label(root, text='Predicted Class:',font=("times",13,"bold italic"))

browse_button.grid(row=1, column=0, padx=10, pady=10)
image_label.grid(row=2, column=0, padx=10, pady=10)
result_label.grid(row=1, column=1, padx=10, pady=10)

# Run the application
root.mainloop()
