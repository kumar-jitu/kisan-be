# main.py
import json
import logging
import os
from typing import Optional, List
import httpx
from fastapi import FastAPI, Form, Response, HTTPException, Query, Header
from pydantic import BaseModel
from twilio.twiml.messaging_response import MessagingResponse
from firebase_functions import https_fn
import uvicorn # For local testing
import requests

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI()

GEMINI_API_KEY = "AIzaSyBT-GVRbWLF8Jg3GqlbxA4MI3Z7RAyxzFA"  # Replace with your actual API key
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-pro:generateContent"

# Pydantic models for crop analysis
class CropAnalysisRequest(BaseModel):
    image: str  # Base64 encoded image

class CropAnalysisResponse(BaseModel):
    disease: str
    nameKannada: str
    confidence: float
    description: str
    symptoms: List[str]
    causes: List[str]
    treatment: List[str]
    prevention: List[str]

# Analysis prompt for crop disease detection
CROP_ANALYSIS_PROMPT = """
You are an expert agricultural scientist specializing in crop disease diagnosis. Analyze the provided crop image and provide a detailed disease diagnosis.

IMPORTANT: You must respond with ONLY a valid JSON object in the exact format specified below. Do not include any other text, explanations, or markdown formatting.

If you detect a disease, use this format:
{
    "disease": "Disease name in English",
    "nameKannada": "ರೋಗದ ಹೆಸರು ಕನ್ನಡದಲ್ಲಿ",
    "confidence": 0.85,
    "description": "Detailed description of the disease in Kannada",
    "symptoms": ["Symptom 1", "Symptom 2", "Symptom 3"],
    "causes": ["Cause 1", "Cause 2", "Cause 3"],
    "treatment": ["Treatment step 1", "Treatment step 2", "Treatment step 3"],
    "prevention": ["Prevention step 1", "Prevention step 2", "Prevention step 3"]
}

If no disease is detected (healthy plant), use this format:
{
    "disease": "Healthy Plant",
    "nameKannada": "ಸುಸ್ಥಿತಿಯಲ್ಲಿರುವ ಸಸ್ಯ",
    "confidence": 1.0,
    "description": "ಸಸ್ಯವು ಆರೋಗ್ಯಕರವಾಗಿದೆ ಮತ್ತು ಯಾವುದೇ ರೋಗದ ಚಿಹ್ನೆಗಳನ್ನು ತೋರಿಸುತ್ತಿಲ್ಲ.",
    "symptoms": [],
    "causes": [],
    "treatment": [],
    "prevention": ["Continue regular care and monitoring"]
}

Guidelines:
1. Focus on common Indian crop diseases (rice, wheat, cotton, sugarcane, pulses, etc.)
2. Provide practical, farmer-friendly advice
3. Use simple Kannada language that farmers can understand
4. Include both chemical and organic treatment options
5. Confidence should be between 0.0 and 1.0
6. If uncertain, provide the most likely diagnosis with lower confidence
7. Always respond with valid JSON only
"""

@app.post("/sms")
def handle_sms(From: str = Form(...), Body: str = Form(...)):
    """
    Handles incoming SMS messages from Twilio.
    It expects 'From' (the sender's number) and 'Body' (the message text).
    Calls Gemini API to generate a response.
    """
    print(f"Received message from {From}: {Body}")

    try:
        # Prepare the request for Gemini API
        headers = {
            "Content-Type": "application/json",
            "X-goog-api-key": GEMINI_API_KEY
        }

        payload = {
            "contents": [
                {
                    "parts": [
                        {
                            "text": f"{Body}. Give response in 160 characters"
                        }
                    ]
                }
            ]
        }

        # Call Gemini API
        response = requests.post(GEMINI_API_URL, headers=headers, json=payload)
        response.raise_for_status()  # Raise an exception for bad status codes

        # Parse the response
        gemini_response = response.json()
        ai_text = gemini_response["candidates"][0]["content"]["parts"][0]["text"]

        print(f"Gemini response: {ai_text}")

    except requests.RequestException as e:
        print(f"Error calling Gemini API: {e}")
        ai_text = "Sorry, I'm having trouble processing your message right now."
    except (KeyError, IndexError) as e:
        print(f"Error parsing Gemini response: {e}")
        ai_text = "Sorry, I received an unexpected response format."
    except Exception as e:
        print(f"Unexpected error: {e}")
        ai_text = "Sorry, something went wrong."

    # Create a TwiML response
    twiml_response = MessagingResponse()
    twiml_response.message(ai_text)

    # Return the TwiML response as XML
    return Response(content=str(twiml_response), media_type="application/xml")


@app.post("/api/crop-analysis", response_model=CropAnalysisResponse)
async def analyze_crop(request: CropAnalysisRequest):
    """Analyze crop image for disease detection using Gemini Pro."""
    try:
        logger.info("Starting crop analysis...")

        # Prepare the request for Gemini Pro API
        headers = {
            "Content-Type": "application/json",
            "X-goog-api-key": GEMINI_API_KEY
        }

        payload = {
            "contents": [
                {
                    "parts": [
                        {
                            "text": CROP_ANALYSIS_PROMPT
                        },
                        {
                            "inline_data": {
                                "mime_type": "image/jpeg",
                                "data": request.image
                            }
                        }
                    ]
                }
            ]
        }

        # Call Gemini Pro API
        response = requests.post(GEMINI_API_URL, headers=headers, json=payload)
        response.raise_for_status()

        # Parse the response
        gemini_response = response.json()
        response_text = gemini_response["candidates"][0]["content"]["parts"][0]["text"].strip()
        logger.info(f"Raw response: {response_text}")

        # Parse JSON response
        try:
            # Try to extract JSON from the response
            start_idx = response_text.find('{')
            end_idx = response_text.rfind('}') + 1

            if start_idx != -1 and end_idx != 0:
                json_str = response_text[start_idx:end_idx]
                result = json.loads(json_str)

                # Validate required fields
                required_fields = ['disease', 'nameKannada', 'confidence', 'description', 'symptoms', 'causes', 'treatment', 'prevention']
                for field in required_fields:
                    if field not in result:
                        raise ValueError(f"Missing required field: {field}")

                # Ensure confidence is a number
                result['confidence'] = float(result['confidence'])

                # Ensure lists are properly formatted
                for field in ['symptoms', 'causes', 'treatment', 'prevention']:
                    if not isinstance(result[field], list):
                        result[field] = []

                logger.info(f"Analysis completed successfully: {result['disease']}")
                return CropAnalysisResponse(**result)
            else:
                raise ValueError("No JSON found in response")

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {e}")
            logger.error(f"Raw response: {response_text}")

            # Return fallback response
            return CropAnalysisResponse(
                disease="Analysis Failed",
                nameKannada="ವಿಶ್ಲೇಷಣೆ ವಿಫಲ",
                confidence=0.0,
                description="Unable to parse analysis results. Please try again with a clearer image.",
                symptoms=[],
                causes=[],
                treatment=[],
                prevention=[]
            )

    except requests.RequestException as e:
        logger.error(f"Error calling Gemini API: {e}")
        raise HTTPException(status_code=500, detail=f"API call failed: {str(e)}")
    except Exception as e:
        logger.error(f"Error in crop analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/get-mandi-data")
async def get_agricultural_data(
        api_key: str = Query(..., alias="api-key"),
        format: str = Query(default="json"),
        limit: int = Query(default=100),
        state_keyword: Optional[str] = Query(default=None, alias="state"),
        district: Optional[str] = Query(default=None),
        commodity: Optional[str] = Query(default=None),
        accept: str = Header(default="application/json")
):
    """
    Fetch data from data.gov.in API
    Equivalent to: curl --location --globoff 'https://api.data.gov.in/resource/9ef84268-d588-465a-a308-a864a43d0070?api-key=ABC&format=json&limit=100&filters[state.keyword]=Karnataka&filters[district]=Bangalore&filters[commodity]=null' --header 'accept: application/xml'
    """

    # API endpoint
    url = "https://api.data.gov.in/resource/9ef84268-d588-465a-a308-a864a43d0070"

    # Build query parameters
    params = {
        "api-key": api_key,
        "format": format,
        "limit": limit
    }

    # Add filters
    if state_keyword:
        params["filters[state.keyword]"] = state_keyword
    if district:
        params["filters[district]"] = district
    if commodity:
        params["filters[commodity]"] = commodity

    # Headers
    headers = {"accept": accept}

    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(url, params=params, headers=headers, timeout=30.0)
            response.raise_for_status()

            # Return JSON response if possible, otherwise return text
            try:
                return response.json()
            except:
                return {"content": response.text}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "agricultural-api"}

@app.get("/")
async def root():
    """Root endpoint."""
    return {"message": "Agricultural API with SMS and Crop Analysis", "version": "1.0.0"}


# This is the entry point for Firebase Functions
@https_fn.on_request()
def twilio_handler(req: https_fn.Request) -> https_fn.Response:
    """
    Wraps the FastAPI app for Firebase Cloud Functions.
    """
    return https_fn.to_asgi_app(app)(req)

# This block allows you to run the app locally for testing
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)