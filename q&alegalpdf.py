# -*- coding: utf-8 -*-
"""Q&ALegalPDF.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1F55mVLqICZomWVRZlcSblrIhMd5dod6o
"""

pip install openai==0.28

# prompt: mount my drive

from google.colab import drive
drive.mount('/content/drive')

pip install PyMuPDF

import openai

# Enter your OpenAI API key here
openai.api_key = ''

# Enter the path to your PDF file here
pdf_path = '/content/drive/MyDrive/LegalData/pdf_chunk_summaries_output_wrapped.pdf'

#Enter the path of the output file
output_path = '/content/drive/MyDrive/LegalData/qa_pairs.jsonl'

#Enter the name of the model
model_name = 'gpt-4o-mini'

import openai
import fitz  # PyMuPDF
import time
import json
import re

# Function to extract text from a PDF using PyMuPDF
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text += page.get_text()
    return text

# Function to chunk text into smaller sizes (aiming for less than 1000 tokens per chunk)
def chunk_text(text, max_tokens=10000):
    chunks = []
    current_chunk = ""
    for paragraph in text.split('\n\n'):  # Assuming paragraphs are separated by double newlines
        if len(current_chunk.split()) + len(paragraph.split()) < max_tokens:  # Ensure chunk is less than max_tokens
            current_chunk += paragraph + "\n\n"
        else:
            chunks.append(current_chunk)
            current_chunk = paragraph + "\n\n"
    if current_chunk:
        chunks.append(current_chunk)
    return chunks

# Function to generate Q&A pairs from a chunk of text
def generate_qa_pairs(chunk):
    response = openai.ChatCompletion.create(
        model=model_name,  # Changed model as requested
        messages=[
            {"role": "system", "content": "You are a helpful Legal Chatbot Assistant."},
            {"role": "user", "content": f"Generate 100 Q&A pairs from the following text:\n\n{chunk}"}
        ],
        max_tokens=15000,  # Set max_tokens to 5000 as requested
        temperature=0.7
    )

    # Log the raw API response for debugging
    print("\nRaw API Response:\n", response['choices'][0]['message']['content'])

    return response['choices'][0]['message']['content'].strip()

# Function to process the generated response and remove ** characters
def process_qa_response(qa_pair_text):
    qa_pairs = []

    # Remove ** characters from the text
    qa_pair_text_cleaned = qa_pair_text.replace("**", "").strip()

    # Extract Q&A pairs using a simple split
    qa_list = qa_pair_text_cleaned.split("\n\n")

    for qa in qa_list:
        if "Q:" in qa and "A:" in qa:
            question = qa.split("Q:", 1)[1].split("A:", 1)[0].strip()  # Extract question
            answer = qa.split("A:", 1)[1].strip()  # Extract answer
            qa_pairs.append((question, answer))

    return qa_pairs

# Function to handle API rate limiting by adding a delay and retry if needed
def generate_qa_with_rate_limit(chunks):
    qa_pairs = []
    for chunk in chunks:
        try:
            response = generate_qa_pairs(chunk)
            qa_pairs.extend(process_qa_response(response))
            time.sleep(1)  # Add a delay to prevent rate limiting
        except openai.error.RateLimitError:
            print("Rate limit reached, waiting for 60 seconds...")
            time.sleep(60)  # Wait for 60 seconds if rate limit is hit
            response = generate_qa_pairs(chunk)  # Retry after waiting
            qa_pairs.extend(process_qa_response(response))
        except openai.error.APIError as e:
            print(f"Error occurred: {e}. Retrying...")
            time.sleep(5)  # Short wait before retrying
            response = generate_qa_pairs(chunk)
            qa_pairs.extend(process_qa_response(response))
    return qa_pairs

# Extract text from the PDF
pdf_text = extract_text_from_pdf(pdf_path)

# Chunk the extracted text
chunks = chunk_text(pdf_text)

# Generate Q&A pairs for each chunk
qa_pairs = generate_qa_with_rate_limit(chunks)

# Check if Q&A pairs are generated
if len(qa_pairs) == 0:
    print("No Q&A pairs generated. Please check the input and API responses.")
else:
    print(f"Generated {len(qa_pairs)} Q&A pairs.")

# Save the Q&A pairs in the required JSONL format
#output_path = '/content/drive/MyDrive/LegalData/qa_pairs.jsonl'

# Write the Q&A pairs to the JSONL file
try:
    with open(output_path, 'w') as f:
        for question, answer in qa_pairs:
            json_obj = {
                "messages": [
                    {"role": "system", "content": "You are a helpful Legal Chatbot Assistant."},
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": answer}
                ]
            }
            f.write(json.dumps(json_obj) + "\n")  # Writing each entry to a new line
    print(f"Q&A pairs have been successfully saved in JSONL format to {output_path}")
except Exception as e:
    print(f"Error writing to file: {e}")

# Print the Q&A pairs
print("\nGenerated Q&A Pairs:\n")
for question, answer in qa_pairs:
    print(f"Q: {question}")
    print(f"A: {answer}")
    print("\n" + "-"*50 + "\n")
