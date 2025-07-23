# main.py
from fastapi import FastAPI, Form, Response
from twilio.twiml.messaging_response import MessagingResponse
from firebase_functions import https_fn
import uvicorn # For local testing
import requests

# Initialize FastAPI app
app = FastAPI()

GEMINI_API_KEY = "AIzaSyDS07BoUDPV5Tqf_MbbCjRQwQv2hFQ1zzM"  # Replace with your actual API key
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"

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