from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from groq import Groq
from fastapi.middleware.cors import CORSMiddleware

# Initialize FastAPI app
app = FastAPI()
import logging

logging.basicConfig(level=logging.INFO)

# Allow specific origins (replace with the actual URL of your Angular frontend)
origins = ["*"]
# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # Origins allowed to make requests
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)

# Initialize Groq client
client = Groq(api_key="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")


# Models
class ChatRequest(BaseModel):
    messages: list
    model: str


class TranslationRequest(BaseModel):
    text: str
    target_language: str


@app.post("/api/chat-completion")
async def chat_completion(request: ChatRequest):
    """
    Create a chat completion using the Groq API.
    """
    try:
        # Perform the chat completion query
        chat_completion = client.chat.completions.create(
            messages=request.messages, model=request.model
        )

        # Return the result
        return {"response": chat_completion.choices[0].message.content}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/cultural-tips/")
async def cultural_tips(location: str):
    """Generate cultural tips based on location."""
    try:
        prompt = f"What are some cultural tips to avoid faux pas in {location}?"

        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}], model="llama3-8b-8192"
        )

        return {"hidden_gems": chat_completion.choices[0].message.content}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/translate/")
async def translate_text(request: TranslationRequest):
    try:
        query = f"""
        Translate the following text from {request.text} to {request.target_language}:
        "{request.text}"

        Please only return the translated text. If the source language is the same as the destination language, return the original text without any other information.
        """
        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": query}], model="llama3-8b-8192"
        )
        return {"translate_text": chat_completion.choices[0].message.content}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/hidden-gems")
async def hidden_gems(location: str):
    """
    Fetch hidden gems based on the location using Groq.
    """
    try:
        query = f"""
        Fetch hidden gems for {location}. Provide recommendations for lesser-known attractions, restaurants, 
        or experiences that are unique to this destination.
        """
        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": query}], model="llama3-8b-8192"
        )
        return {"hidden_gems": chat_completion.choices[0].message.content}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/destination-activities")
async def destination_activities(destination: str):
    """
    Fetch activities for a specific destination using Groq.
    """
    try:
        query = f"""
        Provide a list of activities for travelers visiting {destination}. Include outdoor adventures, 
        cultural experiences, and family-friendly options.
        """
        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": query}], model="llama3-8b-8192"
        )
        return {"activities": chat_completion.choices[0].message.content}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
