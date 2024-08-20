
    # The code segment allows users to upload an image, perform object detection using DETR-ResNet-50
    # model, draw bounding boxes around detected objects, generate descriptions and summaries for the
    # objects, and display the results in a summary table.
    
    # :param filename: The `filename` parameter in the code refers to the name of the file that is being
    # processed or queried. It is used as an input to functions like `query(filename)` and
    # `text(filename)` where the file is read and processed accordingly. In the Streamlit application
    # context, the `filename`
    # :return: The code provided is a Streamlit application for image segmentation and object detection
    # using the DETR model from Hugging Face and optical character recognition (OCR) using the T-ROCR
    # model from Microsoft.
import streamlit as st
from PIL import Image
import cv2
import requests
from dotenv import load_dotenv
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
import os
import pandas as pd

st.title("image segmentation")

load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

model = ChatGoogleGenerativeAI(model="gemini-pro",temperature=0.3)

API_URL = "https://api-inference.huggingface.co/models/facebook/detr-resnet-50"
headers = {"Authorization": "Bearer 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"}

def query(filename):
    with open(filename, "rb") as f:
        data = f.read()
        response = requests.post(API_URL, headers=headers, data=data)
    return response.json()

API_URL2 = "https://api-inference.huggingface.co/models/microsoft/trocr-base-handwritten"

def text(filename):
    with open(filename, "rb") as f:
        data = f.read()
    response = requests.post(API_URL2, headers=headers, data=data)
    return response.json()


    # The function `get_object_descriptions` takes a list of objects, generates a short description for
    # each object using a model, and returns a list of these descriptions.


def get_object_descriptions(objects_list):

  descriptions = []
  for obj in objects_list:
    text=f"Provide a very short description of {obj}."
    res = model.invoke(text)
    descriptions.append(res.content)

  return descriptions

def get_object_summary(objects_list):

    # This function takes a list of objects, generates a short description for each object, and returns a
    # list of these descriptions.

  descriptions = []
  for obj in objects_list:
    text=f"Provide a very short nature and design of {obj} in just two lines."
    res = model.invoke(text)
    descriptions.append(res.content)

  return descriptions



# The line `uploaded_file = st.file_uploader("Choose an image")` in the provided Python code segment is creating a file uploader widget using Streamlit. This widget allows users to select and upload an image file from their local system. The text "Choose an image" is displayed as the label for the file uploader, prompting the user to select an image file for processing within the Streamlit application.

uploaded_file = st.file_uploader("Choose an image")

if uploaded_file is not None:
    image_data = uploaded_file.read()
    st.image(image_data)
    st.write("file uploaded")
    image = Image.open(uploaded_file)
    # Specify the file path to save the image
    filepath = "./uploaded_image.jpg" 
    # Save the image
    image.save(filepath)
    st.success(f"Image saved successfully at {filepath}")
         

    output = query("uploaded_image.jpg")
    image=cv2.imread("uploaded_image.jpg")
     # The code snippet `res=text("uploaded_image.jpg")` is calling the `text` function with the uploaded image file as a parameter. This function is responsible for processing the uploaded image using the OCR (Optical Character Recognition) model from Microsoft. It extracts text content from the image.

    if output is None:
       res=text("uploaded_image.jpg")
       st.write("Image only contain text")
       st.write(res[0]['generated_text'])
    else:
    
        # Draw bounding boxes on the image
        # The function `draw_bounding_boxes` takes an image and a list of detections, draws bounding boxes around the detected objects, and adds labels and scores to the image.
        def draw_bounding_boxes(image, detections):
            for detection in detections:
                xmin, ymin, xmax, ymax = detection['box']['xmin'], detection['box']['ymin'], detection['box']['xmax'], detection['box']['ymax']
                label = detection['label']
                score = detection['score']

                # Draw the bounding box
                cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

                # Add the label and score
                text = f"{label}: {score:.2f}"
                cv2.putText(image, text, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            return image

        # if output is not None:

            # The line `image_with_boxes = draw_bounding_boxes(image, output)` is calling the `draw_bounding_boxes` function with the `image` and `output` as parameters. This function takes an image and a list of detections as input, then it draws bounding boxes around the detected objects on the image and adds labels and scores to those bounding boxes.
        image_with_boxes = draw_bounding_boxes(image, output)

        # Display the image with bounding boxes
        cv2.imwrite("saved_image.jpg", image_with_boxes)
        st.image("saved_image.jpg")
        obj=[]
        for i in output:
            obj.append(i["label"])
        obj1=list(set(obj))

    # The code segment you provided is responsible for generating object descriptions and summaries based on the detected objects in the uploaded image. Here is a breakdown of what each part of the code is doing:
        desc=get_object_descriptions(obj1)
        summ=get_object_summary(obj1)
        df = pd.DataFrame({'OBJECT': obj1, 'DESCRIPTION': desc, 'SUMMARY': summ})
        res=text("uploaded_image.jpg")

    # The code snippet `st.title("Summary Table")` is setting the title of a section in the Streamlit application to "Summary Table". This title will be displayed at the top of the section to provide context or information to the user about the content that follows.
        st.title("Summary Table")
        st.dataframe(df)

