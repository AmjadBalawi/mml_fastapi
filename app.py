from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from groq import Groq
from fastapi.middleware.cors import CORSMiddleware
import logging
import requests
import datetime
from typing import Optional
# Initialize FastAPI app
app = FastAPI()
page_views = []
# Configure logging
logging.basicConfig(level=logging.INFO)

# Allow specific origins (replace with your actual front-end URL)
origins = ["*"]

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Origins allowed to make requests
    allow_credentials=True,
    allow_methods=["*"],     # Allow all HTTP methods
    allow_headers=["*"],     # Allow all headers
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


@app.get("/cultural-tips/")
async def cultural_tips(location: str):
    """Generate cultural tips based on location."""
    try:
        prompt = f"What are some cultural tips to consider in {location}?"
        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}], model="llama3-8b-8192"
        )
        return {"cultural_tips": chat_completion.choices[0].message.content}
    except Exception as e:
        logging.error(f"Cultural tips error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/translate/")
async def translate_text(request: TranslationRequest):
    """Translate text from one language to another."""
    try:
        query = f"""
        Translate the following text from {request.text} to {request.target_language}:
        "{request.text}"

        Please only return the translated text. If the source language is the same as the destination language,
        return the original text without any other information.
        """
        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": query}], model="llama3-8b-8192"
        )
        return {"translate_text": chat_completion.choices[0].message.content}
    except IndexError:
        logging.error("Translation response was empty.")
        raise HTTPException(status_code=500, detail="Translation error: No choices returned.")
    except Exception as e:
        logging.error(f"Translation error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/hidden-gems/")
async def hidden_gems(location: str):
    """Fetch hidden gems based on the location using Groq."""
    try:
        query = f"""
        Fetch hidden gems for {location}. Provide recommendations
        for lesser-known attractions, restaurants, or experiences
        that are unique to this destination.
        """
        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": query}], model="llama3-8b-8192"
        )
        return {"hidden_gems": chat_completion.choices[0].message.content}
    except Exception as e:
        logging.error(f"Hidden gems error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/destination-activities/")
async def destination_activities(destination: str):
    """Fetch activities for a specific destination using Groq."""
    try:
        query = f"""
        Provide a list of activities for travelers visiting {destination}.
        Include outdoor adventures, cultural experiences, and family-friendly options.
        """
        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": query}], model="llama3-8b-8192"
        )
        return {"activities": chat_completion.choices[0].message.content}
    except Exception as e:
        logging.error(f"Destination activities error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


class WeatherResponse(BaseModel):
    location: str
    latitude: float
    longitude: float
    timezone: str
    description: str
    current_temperature: float
    humidity: float
    feels_like: float
    wind_speed: float
    precipitation: Optional[float]
    forecast: float


def fahrenheit_to_celsius(fahrenheit: float) -> float:
    """Convert Fahrenheit to Celsius."""
    return (fahrenheit - 32) * 5 / 9


@app.get("/weather/{city}", response_model=WeatherResponse)
async def get_weather(city: str):
    api_key = "LABN5XZPTZ3RGLZCG4BGBMFHJ"
    url = f"https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/{city}?key={api_key}"

    response = requests.get(url)
    data = response.json()

    # Extract necessary data and convert temperatures to Celsius
    current_temperature_f = data["currentConditions"]["temp"]
    feels_like_f = data["currentConditions"]["feelslike"]
    forecast_f = data["days"][0]["tempmax"]  # Assuming you want the max temperature for the day

    # Convert Fahrenheit to Celsius
    current_temperature_c = fahrenheit_to_celsius(current_temperature_f)
    feels_like_c = fahrenheit_to_celsius(feels_like_f)
    forecast_c = fahrenheit_to_celsius(forecast_f)

    return WeatherResponse(
        location=data["resolvedAddress"],
        latitude=data["latitude"],
        longitude=data["longitude"],
        timezone=data["timezone"],
        description=data["currentConditions"]["conditions"],
        current_temperature=int(current_temperature_c),
        humidity=data["currentConditions"]["humidity"],
        feels_like=int(feels_like_c),
        wind_speed=data["currentConditions"]["windspeed"],
        precipitation=data["currentConditions"]["precip"],
        forecast=int(forecast_c)
    )
@app.middleware("http")
async def track_requests(request: Request, call_next):
    # Log request details (this can be customized based on what data you want to capture)
    page_view = {
        "method": request.method,
        "url": str(request.url),
        "timestamp": datetime.datetime.utcnow()
    }
    page_views.append(page_view)

    # Process the request and return the response
    response = await call_next(request)
    return response

@app.get("/page-views/", include_in_schema=False)
async def get_page_views():
    return page_views
