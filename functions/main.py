# main.py
import json
import logging
import os
from typing import Optional, List
import httpx
import base64
import io
import tempfile
import wave
import struct
from fastapi import FastAPI, Form, Response, HTTPException, Query, Header, UploadFile, File
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from twilio.twiml.messaging_response import MessagingResponse
from firebase_functions import https_fn
import uvicorn # For local testing
import requests

# Import credentials configuration
from env_config import GEMINI_API_KEY

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI()

# API URLs
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-pro:generateContent"
GEMINI_TTS_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-tts:generateContent"

@app.on_event("startup")
async def startup_event():
    """Initialize credentials and validate API key on startup."""
    global GEMINI_API_KEY
    if GEMINI_API_KEY is None:
        logger.error("Failed to load Gemini API key. Application may not function properly.")
        raise RuntimeError("Gemini API key not available")

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

# Pydantic models for voice notes
class VoiceNoteResponse(BaseModel):
    transcription: str
    ai_response: str
    status: str = "success"

# Audio conversion utilities
def convert_pcm_to_wav(pcm_data: bytes, channels: int = 1, sample_rate: int = 24000, sample_width: int = 2) -> bytes:
    """
    Convert raw PCM audio data to WAV format.
    
    Args:
        pcm_data: Raw PCM audio bytes (L16 format from Gemini TTS)
        channels: Number of audio channels (1 for mono)
        sample_rate: Sample rate in Hz (24000 for Gemini TTS)
        sample_width: Sample width in bytes (2 for 16-bit)
    
    Returns:
        WAV format audio bytes
    """
    try:
        # Create a BytesIO buffer to write WAV data
        wav_buffer = io.BytesIO()
        
        # Create WAV file in memory
        with wave.open(wav_buffer, 'wb') as wav_file:
            wav_file.setnchannels(channels)
            wav_file.setsampwidth(sample_width)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(pcm_data)
        
        # Get the WAV data
        wav_buffer.seek(0)
        wav_data = wav_buffer.read()
        wav_buffer.close()
        
        logger.info(f"Converted PCM ({len(pcm_data)} bytes) to WAV ({len(wav_data)} bytes)")
        return wav_data
        
    except Exception as e:
        logger.error(f"Error converting PCM to WAV: {e}")
        raise

def validate_audio_data(audio_data: bytes) -> dict:
    """
    Validate and analyze audio data format.
    
    Returns:
        Dictionary with audio format information
    """
    info = {
        "size": len(audio_data),
        "format": "unknown",
        "is_wav": False,
        "is_pcm": False
    }
    
    if len(audio_data) >= 12:
        # Check for WAV header
        if audio_data[:4] == b'RIFF' and audio_data[8:12] == b'WAVE':
            info["format"] = "WAV"
            info["is_wav"] = True
        else:
            # Assume raw PCM if not WAV
            info["format"] = "PCM"
            info["is_pcm"] = True
    
    return info

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


@app.post("/api/voice-note", response_model=VoiceNoteResponse)
async def process_voice_note(audio: UploadFile = File(...)):
    """
    Process voice note: transcribe audio, generate AI response, and return text response.
    For the audio response version, use /api/voice-note-audio endpoint.
    """
    try:
        logger.info("Processing voice note...")
        
        # Read audio file
        audio_data = await audio.read()
        
        # Convert audio to base64
        audio_base64 = base64.b64encode(audio_data).decode('utf-8')
        
        # Prepare request for audio understanding
        headers = {
            "Content-Type": "application/json",
            "X-goog-api-key": GEMINI_API_KEY
        }
        
        payload = {
            "contents": [
                {
                    "parts": [
                        {
                            "text": "Please transcribe this audio and then provide a helpful agricultural response in Kannada language to what was said. Focus on providing practical farming advice, crop management tips, or answers to agricultural questions."
                        },
                        {
                            "inline_data": {
                                "mime_type": audio.content_type or "audio/wav",
                                "data": audio_base64
                            }
                        }
                    ]
                }
            ]
        }
        
        # Call Gemini API for transcription and response
        response = requests.post(GEMINI_API_URL, headers=headers, json=payload)
        response.raise_for_status()
        
        gemini_response = response.json()
        full_response = gemini_response["candidates"][0]["content"]["parts"][0]["text"]
        
        # Extract transcription and AI response from the full response
        # The model should provide both transcription and response
        lines = full_response.split('\n')
        transcription = ""
        ai_response = ""
        
        # Simple parsing - look for transcription and response sections
        current_section = ""
        for line in lines:
            line = line.strip()
            if "transcription" in line.lower() or "transcript" in line.lower():
                current_section = "transcription"
                continue
            elif "response" in line.lower() or "answer" in line.lower():
                current_section = "response"
                continue
            elif line and current_section == "transcription":
                transcription += line + " "
            elif line and current_section == "response":
                ai_response += line + " "
        
        # If parsing failed, use the full response as AI response
        if not transcription and not ai_response:
            # Try to split by common patterns
            if ":" in full_response:
                parts = full_response.split(":", 1)
                if len(parts) == 2:
                    transcription = parts[0].strip()
                    ai_response = parts[1].strip()
                else:
                    transcription = "Audio processed"
                    ai_response = full_response.strip()
            else:
                transcription = "Audio processed"
                ai_response = full_response.strip()
        
        logger.info(f"Voice note processed successfully")
        return VoiceNoteResponse(
            transcription=transcription.strip(),
            ai_response=ai_response.strip(),
            status="success"
        )
        
    except requests.RequestException as e:
        logger.error(f"Error calling Gemini API: {e}")
        raise HTTPException(status_code=500, detail=f"API call failed: {str(e)}")
    except Exception as e:
        logger.error(f"Error processing voice note: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/voice-note-audio")
async def process_voice_note_with_audio_response(audio: UploadFile = File(...)):
    """
    Process voice note and return audio response:
    1. Transcribe input audio
    2. Generate AI text response 
    3. Convert AI response to speech
    4. Return audio file
    """
    try:
        logger.info("Processing voice note with audio response...")
        
        # Read audio file
        audio_data = await audio.read()
        
        # Convert audio to base64
        audio_base64 = base64.b64encode(audio_data).decode('utf-8')
        
        # Prepare request for audio understanding
        headers = {
            "Content-Type": "application/json",
            "X-goog-api-key": GEMINI_API_KEY
        }
        
        # Step 1: Get transcription and generate text response
        payload = {
            "contents": [
                {
                    "parts": [
                        {
                            "text": "Please transcribe this audio and then provide a helpful agricultural response in Kannada language to what was said. Focus on providing practical farming advice, crop management tips, or answers to agricultural questions. Keep the response concise and conversational, suitable for text-to-speech conversion."
                        },
                        {
                            "inline_data": {
                                "mime_type": audio.content_type or "audio/wav",
                                "data": audio_base64
                            }
                        }
                    ]
                }
            ]
        }
        
        # Call Gemini API for transcription and response
        response = requests.post(GEMINI_API_URL, headers=headers, json=payload)
        response.raise_for_status()
        
        gemini_response = response.json()
        text_response = gemini_response["candidates"][0]["content"]["parts"][0]["text"]
        
        # Extract just the AI response part for TTS
        # Look for response section or use full text if parsing fails
        ai_response_text = text_response
        if "response:" in text_response.lower():
            parts = text_response.lower().split("response:", 1)
            if len(parts) == 2:
                ai_response_text = parts[1].strip()
        elif "\n" in text_response:
            # Take the last substantial line as the response
            lines = [line.strip() for line in text_response.split('\n') if line.strip()]
            if len(lines) > 1:
                ai_response_text = lines[-1]
        
        # Step 2: Convert AI response to speech using Gemini TTS
        logger.info(f"AI response text for TTS: {ai_response_text[:100]}...")
        
        tts_payload = {
            "contents": [
                {
                    "parts": [
                        {
                            "text": ai_response_text
                        }
                    ]
                }
            ],
            "generationConfig": {
                "responseModalities": ["AUDIO"],
                "speechConfig": {
                    "voiceConfig": {
                        "prebuiltVoiceConfig": {
                            "voiceName": "Kore"
                        }
                    }
                }
            }
        }
        
        logger.info(f"TTS API URL: {GEMINI_TTS_API_URL}")
        logger.info(f"TTS payload: {json.dumps(tts_payload, indent=2)}")
        
        # Call Gemini TTS API
        try:
            tts_response = requests.post(GEMINI_TTS_API_URL, headers=headers, json=tts_payload)
            logger.info(f"TTS API status code: {tts_response.status_code}")
            logger.info(f"TTS API response headers: {dict(tts_response.headers)}")
            
            if tts_response.status_code != 200:
                logger.error(f"TTS API error response: {tts_response.text}")
                raise HTTPException(status_code=500, detail=f"TTS API returned {tts_response.status_code}: {tts_response.text}")
            
            tts_result = tts_response.json()
            logger.info(f"TTS response structure: {json.dumps({k: str(type(v)) for k, v in tts_result.items()}, indent=2)}")
            
        except requests.RequestException as e:
            logger.error(f"TTS API request failed: {e}")
            raise HTTPException(status_code=500, detail=f"TTS API request failed: {str(e)}")
        
        # Extract audio data from TTS response
        logger.info("Attempting to extract audio data from TTS response...")
        
        if tts_result.get("candidates"):
            logger.info(f"Found {len(tts_result['candidates'])} candidates")
            candidate = tts_result["candidates"][0]
            
            if candidate.get("content"):
                logger.info("Found content in candidate")
                content = candidate["content"]
                
                if content.get("parts"):
                    logger.info(f"Found {len(content['parts'])} parts in content")
                    audio_part = content["parts"][0]
                    logger.info(f"Audio part keys: {list(audio_part.keys())}")
                    
                    # Check both possible field names for inline data
                    inline_data_field = audio_part.get("inlineData") or audio_part.get("inline_data")
                    if inline_data_field:
                        logger.info(f"Inline data keys: {list(inline_data_field.keys())}")
                        
                        if inline_data_field.get("data"):
                            logger.info("Found audio data, decoding...")
                            raw_audio_bytes = base64.b64decode(inline_data_field["data"])
                            logger.info(f"Decoded raw audio size: {len(raw_audio_bytes)} bytes")
                            
                            # Validate and convert audio format
                            audio_info = validate_audio_data(raw_audio_bytes)
                            logger.info(f"Audio format info: {audio_info}")
                            
                            # Convert PCM to WAV if needed
                            if audio_info["is_pcm"]:
                                logger.info("Converting PCM to WAV format...")
                                wav_audio_bytes = convert_pcm_to_wav(raw_audio_bytes)
                            elif audio_info["is_wav"]:
                                logger.info("Audio is already in WAV format")
                                wav_audio_bytes = raw_audio_bytes
                            else:
                                logger.warning("Unknown audio format, assuming PCM and converting...")
                                wav_audio_bytes = convert_pcm_to_wav(raw_audio_bytes)
                            
                            logger.info(f"Final WAV audio size: {len(wav_audio_bytes)} bytes")
                            
                            # Encode Kannada text safely for headers
                            try:
                                # Try to encode the transcription for header, fallback if it fails
                                transcription_header = text_response[:200] + "..." if len(text_response) > 200 else text_response
                                # Test if it can be encoded as Latin-1 (HTTP header requirement)
                                transcription_header.encode('latin-1')
                                safe_transcription = transcription_header
                            except UnicodeEncodeError:
                                # If Kannada text can't be encoded, use base64 or a safe fallback
                                safe_transcription = base64.b64encode(transcription_header.encode('utf-8')).decode('ascii')
                                logger.info("Encoded transcription as base64 due to Unicode characters")
                            
                            return StreamingResponse(
                                io.BytesIO(wav_audio_bytes),
                                media_type="audio/wav",
                                headers={
                                    "Content-Disposition": "attachment; filename=voice_response.wav",
                                    "X-Transcription-B64": safe_transcription,
                                    "X-Audio-Size": str(len(wav_audio_bytes)),
                                    "X-Original-Format": audio_info["format"],
                                    "X-Sample-Rate": "24000",
                                    "X-Channels": "1"
                                }
                            )
                        else:
                            logger.error("No 'data' field in inline data")
                    else:
                        logger.error("No 'inlineData' or 'inline_data' field in audio part")
                else:
                    logger.error("No 'parts' field in content")
            else:
                logger.error("No 'content' field in candidate")
        else:
            logger.error("No 'candidates' field in TTS result")
        
        # Log the full response structure for debugging
        logger.error(f"Full TTS response: {json.dumps(tts_result, indent=2)}")
        
        # Fallback if TTS failed
        raise HTTPException(status_code=500, detail="Failed to generate audio response - no audio data found in TTS response")
        
    except requests.RequestException as e:
        logger.error(f"Error calling Gemini API: {e}")
        raise HTTPException(status_code=500, detail=f"API call failed: {str(e)}")
    except UnicodeEncodeError as e:
        logger.error(f"Unicode encoding error: {e}")
        raise HTTPException(status_code=500, detail=f"Text encoding error - this may be due to Unicode characters in the response: {str(e)}")
    except Exception as e:
        logger.error(f"Error processing voice note with audio: {e}")
        logger.error(f"Exception type: {type(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "agricultural-api"}

@app.post("/getmarketprice")
async def predict_crop_price(
        crop_name: str = Form(...),
        state: str = Form(...),
        district: str = Form(...)
):
    """
    Predict crop prices using Gemini 2.0 Flash model.
    Accepts crop_name, state, and district as form parameters.
    """
    try:
        logger.info(f"Starting crop price prediction for {crop_name} in {district}, {state}")

        # Prepare the request for Gemini 2.0 Flash API
        headers = {
            "Content-Type": "application/json",
            "x-goog-api-key": GEMINI_API_KEY
        }

        # Construct the user prompt with the provided parameters
        user_prompt = f"""Crop Price Request - JSON Response Only

Instructions: Provide 7-day price prediction in JSON format ONLY for {crop_name} in {district}, {state}. Include today's vs yesterday's price comparison with percentage difference.

Input:
Crop: {crop_name}
District: {district}
State: {state}

Required JSON Response Must Include:
- Today's current mandi price 
- Yesterday's price
- Price change amount (₹Z)
- Percentage change: ((Today's Price - Yesterday's Price) / Yesterday's Price) × 100
- 7-day daily price forecast with specific dates starting from 2025-07-27
- Single recommendation (sell/wait/monitor)
- One key factor affecting price
- Data source used

IMPORTANT:
- Respond ONLY with valid JSON format
- No explanatory text before or after JSON
- Use actual price values, not placeholders
- Include thought process and data sources within JSON structure
- Focus only on {district}, {state} mandi data."""

        payload = {
            "systemInstruction": {
                "parts": [
                    {
                        "text": """Primary Objective: You are a crop price analyst. Provide ONLY JSON format responses with actual price data for specified crops in specific locations. NO explanatory text outside JSON.

Core Task:
When user provides crop name, state, and district:
1. Search web for recent mandi price data for that specific location (last 180 days minimum)
2. Find today's current price and yesterday's price for comparison
3. Calculate percentage difference: ((Today's Price - Yesterday's Price) / Yesterday's Price) × 100
4. Analyze short-term trends based on historical data
5. Predict prices for next 7 days with specific dates
6. Provide ONE selling recommendation

Response Requirements:
- ONLY valid JSON format response
- Focus on specific state and district data
- Include today's price vs yesterday's comparison with percentage
- Include 7-day forecast with actual dates
- No text before or after JSON
- Include data sources

JSON Response Format:{"current_price":"<current_price>","yesterday_changes":"<yesterday_changes>","changes":"up/down","one_day_later_price":"<one_day_later_price>","two_day_later_price":"<two_day_later_price>","three_day_later_price":"<three_day_later_price>","four_day_later_price":"<four_day_later_price>","five_day_later_price":"<five_day_later_price>"}"""
                    }
                ]
            },
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {
                            "text": user_prompt
                        }
                    ]
                }
            ],
            "generationConfig": {
                "temperature": 0.1,
                "topK": 1,
                "topP": 0.8,
                "maxOutputTokens": 2048,
                "candidateCount": 1
            }
        }

        # Call Gemini 2.0 Flash API
        response = requests.post(GEMINI_API_URL, headers=headers, json=payload)
        response.raise_for_status()

        # Parse the response
        gemini_response = response.json()
        response_text = gemini_response["candidates"][0]["content"]["parts"][0]["text"].strip()
        logger.info(f"Raw price prediction response: {response_text}")

        # Extract JSON from response
        try:
            # Find JSON in the response
            start_idx = response_text.find('{')
            end_idx = response_text.rfind('}') + 1

            if start_idx != -1 and end_idx != 0:
                json_str = response_text[start_idx:end_idx]
                result = json.loads(json_str)

                logger.info(f"Price prediction completed successfully for {crop_name}")
                return result
            else:
                raise ValueError("No JSON found in response")

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {e}")
            logger.error(f"Raw response: {response_text}")

            # Return fallback response
            return {
                "error": "Failed to parse price prediction",
                "crop": crop_name,
                "location": f"{district}, {state}",
                "message": "Unable to get current price data. Please try again.",
                "raw_response": response_text[:500]  # First 500 chars for debugging
            }

    except requests.RequestException as e:
        logger.error(f"Error calling Gemini API: {e}")
        raise HTTPException(status_code=500, detail=f"API call failed: {str(e)}")
    except Exception as e:
        logger.error(f"Error in crop price prediction: {e}")
        raise HTTPException(status_code=500, detail=str(e))


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