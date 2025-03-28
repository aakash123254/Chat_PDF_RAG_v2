Overview

Chat PDF RAG is an AI-powered chatbot that allows users to upload PDF documents and ask questions about their content. The system utilizes Retrieval-Augmented Generation (RAG) with FAISS for efficient document retrieval and Google Gemini embeddings to enhance accuracy. The application is built using Flask, and the frontend is designed for an improved user experience.

Features

Upload PDF Documents: Users can upload PDFs for processing.

Intelligent Question Answering: The chatbot extracts relevant information from the uploaded document using FAISS retrieval.

Google Gemini Embeddings: Enhances search relevance and response accuracy.

Flask API Backend: A robust API serves requests efficiently.

Improved UI: A user-friendly interface enhances interaction with the chatbot.

Installation & Setup

Prerequisites

Ensure you have the following installed:

Python 3.8+

pip (Python package manager)

Git

Clone the Repository

 git clone https://github.com/aakash123254/Chat_PDF_RAG_v2.git
 cd Chat_PDF_RAG_v2

Create a Virtual Environment (Optional but Recommended)

 python -m venv venv
 source venv/bin/activate  # On Windows use `venv\Scripts\activate`

Install Dependencies

 pip install -r requirements.txt

Usage

Start the Flask Server

 stramlit run chatpdf1.py

The server will start at Local URL: http://localhost:8501

Upload a PDF & Ask Questions

Open the web UI in a browser.

Upload a PDF document.

Enter your question in the chat input field.

The chatbot will retrieve relevant information and generate responses.

API Endpoints

Upload PDF

POST /upload

Request:

File: multipart/form-data (PDF file)

Response:

JSON object with file processing status

Ask Question

POST /ask

Request:

JSON { "question": "Your question here" }

Response:

JSON { "answer": "Generated answer" }

Technologies Used

Flask (Backend API)

FAISS (Efficient Vector Search)

Google Gemini Embeddings (Semantic Search)

Python (Core Development)

HTML/CSS/JavaScript (Frontend UI)

AWS (Deployment)

Screenshots




Contributing

We welcome contributions! To contribute:

Fork the repository.

Create a new branch (git checkout -b feature-branch)

Commit changes (git commit -m "Added new feature")

Push to GitHub (git push origin feature-branch)

Open a pull request.

License

This project is licensed under the MIT License. See LICENSE for details.

Contact

For questions or support, contact:

Developer: Aakash

Email: aakashharwani06@gmail.com

GitHub: Aakash123254

Happy Coding! 🚀