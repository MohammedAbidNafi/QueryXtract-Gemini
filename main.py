import asyncio
import fitz
import numpy as np
import openai
import gradio as gr
from transformers import pipeline
import os


def extract_pdf_text(pdf_path: str) -> str:
    """Extract text from the PDF."""
    with fitz.open(pdf_path) as doc:
        text = ""
        for page in doc:
            text += page.get_text()
    return text

def chunk_text(text: str, max_tokens: int = 1000) -> list:
    """Chunk text into manageable parts."""
    words = text.split()
    chunks = []
    chunk = ""

    for word in words:
        if len(chunk) + len(word) + 1 <= max_tokens:
            chunk += f" {word}"
        else:
            chunks.append(chunk.strip())
            chunk = word
    if chunk:
        chunks.append(chunk.strip())

    return chunks


def load_model():
    """Load the feature-extraction pipeline from transformers."""
    return pipeline("feature-extraction", model="sentence-transformers/all-MiniLM-L6-v2")


def get_embeddings(model, text: str) -> np.ndarray:
    """Generate embeddings for a given text."""
    embeddings = model(text)
    return np.array(embeddings[0][0]) 


def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """Calculate cosine similarity between two vectors."""
    dot_product = np.dot(vec_a, vec_b)
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    return dot_product / (norm_a * norm_b) if norm_a and norm_b else 0.0


async def find_best_chunk(question: str, chunks: list, model) -> dict:
    """Find the best context chunk based on similarity to the question."""
    question_embedding = get_embeddings(model, question)

    best_chunk = ""
    best_similarity = -1

    for chunk in chunks:
        chunk_embedding = get_embeddings(model, chunk)
        similarity = cosine_similarity(question_embedding, chunk_embedding)

        if similarity > best_similarity:
            best_similarity = similarity
            best_chunk = chunk

    return {'best_chunk': best_chunk, 'similarity': best_similarity}



client = openai.OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)
async def get_answer_from_openai(question: str, best_chunk: str) -> str:
    """Use OpenAI to generate an answer based on the best chunk."""
   

    try:
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant that answers questions based on the provided context."
                },
                {
                    "role": "user",
                    "content": f"Context: {best_chunk}\nQuestion: {question}"
                }
            ]
        )
        return completion.choices[0].message.content if completion.choices else "No answer found."
    except Exception as e:
        print(f"Error with OpenAI API: {e}")
        return "Error occurred while fetching the answer."

# Step 8: Main function to answer the question from the PDF
async def answer_question_from_pdf(pdf_file, question: str) -> str:
    """Answer a question based on the content of a PDF file."""
    print("Answering question from PDF...")
    text = extract_pdf_text(pdf_file.name)

    # Step 2: Chunk the text into manageable parts
    chunks = chunk_text(text, 1000)

    # Load model
    model = load_model()

    # Step 3: Find the most relevant chunk based on the question
    result = await find_best_chunk(question, chunks, model)
    best_chunk = result['best_chunk']
    similarity = result['similarity']

    # Step 4: If similarity is too low, return a no-answer response
    if similarity < 0.5:
        return "No relevant information found."

    # Step 5: Use OpenAI to generate an answer based on the best chunk
    return await get_answer_from_openai(question, best_chunk)

# Gradio interface
def handle_gradio(pdf_file, question):
    """Handle the Gradio interface for question-answering based on a PDF file."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    answer = loop.run_until_complete(answer_question_from_pdf(pdf_file, question))
    return answer

# Create Gradio UI
interface = gr.Interface(
    fn=handle_gradio,
    inputs=[
        gr.File(label="Upload PDF", file_types=['.pdf']),  # Restrict file type to PDF
        gr.Textbox(label="Enter your question")
    ],
    outputs="text",
    title="QueryXtract",
    description="Upload a PDF file and ask a question to extract answers based on the content.",
)

if __name__ == "__main__":
    interface.launch(share=True)
