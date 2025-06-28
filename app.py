from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from ai21 import AI21Client
from ai21.models.chat import ChatMessage  # Import the required message model
from fastapi.middleware.cors import CORSMiddleware
import logging
import requests
import datetime
import os
from typing import Optional, List

# Initialize FastAPI app
app = FastAPI()
page_views = []

# Configure logging
logging.basicConfig(level=logging.INFO)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize AI21 client
client = AI21Client(api_key=os.getenv("AI21_API_KEY", "a020c79e-cb42-40c4-9577-363b57c82910"))


# Models
class ChatRequest(BaseModel):
    messages: List[dict]
    model: str = "jamba-large"


class TranslationRequest(BaseModel):
    text: str
    target_language: str


# AI21 Query Helper with proper message formatting
async def query_ai21(prompt: str, model: str = "jamba-large") -> str:
    """Query AI21 Jamba model with proper message formatting"""
    try:
        # Create proper ChatMessage objects
        messages = [
            ChatMessage(role="user", content=prompt)
        ]

        response = client.chat.completions.create(
            messages=messages,
            model=model,
            max_tokens=500,
            temperature=0.4,
            top_p=0.9
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logging.error(f"AI21 API error: {str(e)}")
        raise HTTPException(status_code=500, detail="AI service unavailable")


@app.get("/cultural-tips/")
async def cultural_tips(location: str):
    """Generate cultural tips based on location."""
    try:
        prompt = f"Provide 5 concise cultural tips for visitors to {location} in bullet point format"
        tips = await query_ai21(prompt, "jamba-large")
        return {"cultural_tips": tips}
    except Exception as e:
        logging.error(f"Cultural tips error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/translate/")
async def translate_text(request: TranslationRequest):
    """Translate text using Jamba model"""
    try:
        prompt = (
            f"Translate ONLY the text between quotes to {request.target_language}. "
            f"Text: \"{request.text}\""
        )
        translation = await query_ai21(prompt, "jamba-mini")
        return {"translate_text": translation}
    except Exception as e:
        logging.error(f"Translation error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/hidden-gems/")
async def hidden_gems(location: str):
    """Fetch hidden gems based on location"""
    try:
        prompt = f"List 5 hidden gems in {location} with brief descriptions in numbered format"
        gems = await query_ai21(prompt)
        return {"hidden_gems": gems}
    except Exception as e:
        logging.error(f"Hidden gems error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/destination-activities/")
async def destination_activities(destination: str):
    """Fetch activities for a destination"""
    try:
        prompt = (
            f"Suggest 5 activities in {destination} categorized as: "
            "- Outdoor\n- Cultural\n- Family-friendly"
        )
        activities = await query_ai21(prompt)
        return {"activities": activities}
    except Exception as e:
        logging.error(f"Destination activities error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# Weather endpoint remains unchanged
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

    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()

        current_temperature_f = data["currentConditions"]["temp"]
        feels_like_f = data["currentConditions"]["feelslike"]
        forecast_f = data["days"][0]["tempmax"]

        current_temperature_c = fahrenheit_to_celsius(current_temperature_f)
        feels_like_c = fahrenheit_to_celsius(feels_like_f)
        forecast_c = fahrenheit_to_celsius(forecast_f)

        return WeatherResponse(
            location=data["resolvedAddress"],
            latitude=data["latitude"],
            longitude=data["longitude"],
            timezone=data["timezone"],
            description=data["currentConditions"]["conditions"],
            current_temperature=round(current_temperature_c, 1),
            humidity=data["currentConditions"]["humidity"],
            feels_like=round(feels_like_c, 1),
            wind_speed=data["currentConditions"]["windspeed"],
            precipitation=data["currentConditions"].get("precip"),
            forecast=round(forecast_c, 1)
        )
    except Exception as e:
        logging.error(f"Weather API error: {str(e)}")
        raise HTTPException(status_code=500, detail="Weather service unavailable")


@app.middleware("http")
async def track_requests(request: Request, call_next):
    page_view = {
        "method": request.method,
        "url": str(request.url),
        "timestamp": datetime.datetime.utcnow().isoformat(),
        "client": request.client.host if request.client else "unknown"
    }
    page_views.append(page_view)
    response = await call_next(request)
    return response


@app.get("/page-views/", include_in_schema=False)
async def get_page_views():
    return {
        "count": len(page_views),
        "views": page_views[-100:]
    }
