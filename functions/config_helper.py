#!/usr/bin/env python3
"""
Utility script to encode credentials in base64 format.
This script helps you encode your API keys and other sensitive data.
"""

import base64
import sys

def encode_credential(credential: str) -> str:
    """
    Encode a credential string to base64.
    
    Args:
        credential: The plain text credential to encode
        
    Returns:
        Base64 encoded string
    """
    try:
        encoded_bytes = base64.b64encode(credential.encode('utf-8'))
        return encoded_bytes.decode('utf-8')
    except Exception as e:
        print(f"Error encoding credential: {e}")
        return None

def decode_credential(encoded_credential: str) -> str:
    """
    Decode a base64 encoded credential.
    
    Args:
        encoded_credential: The base64 encoded credential
        
    Returns:
        Decoded credential string
    """
    try:
        decoded_bytes = base64.b64decode(encoded_credential)
        return decoded_bytes.decode('utf-8')
    except Exception as e:
        print(f"Error decoding credential: {e}")
        return None

def main():
    """Main function to handle command line arguments."""
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python encode_credentials.py encode <credential>")
        print("  python encode_credentials.py decode <base64_credential>")
        print("  python encode_credentials.py interactive")
        return
    
    command = sys.argv[1].lower()
    
    if command == "encode" and len(sys.argv) == 3:
        credential = sys.argv[2]
        encoded = encode_credential(credential)
        if encoded:
            print(f"Base64 encoded: {encoded}")
        else:
            print("Failed to encode credential")
    
    elif command == "decode" and len(sys.argv) == 3:
        encoded_credential = sys.argv[2]
        decoded = decode_credential(encoded_credential)
        if decoded:
            print(f"Decoded: {decoded}")
        else:
            print("Failed to decode credential")
    
    elif command == "interactive":
        print("Credential Encoder/Decoder")
        print("=" * 30)
        
        while True:
            print("\nOptions:")
            print("1. Encode a credential")
            print("2. Decode a credential")
            print("3. Exit")
            
            choice = input("\nEnter your choice (1-3): ").strip()
            
            if choice == "1":
                credential = input("Enter the credential to encode: ").strip()
                if credential:
                    encoded = encode_credential(credential)
                    if encoded:
                        print(f"Base64 encoded: {encoded}")
                    else:
                        print("Failed to encode credential")
                else:
                    print("No credential provided")
            
            elif choice == "2":
                encoded_credential = input("Enter the base64 encoded credential: ").strip()
                if encoded_credential:
                    decoded = decode_credential(encoded_credential)
                    if decoded:
                        print(f"Decoded: {decoded}")
                    else:
                        print("Failed to decode credential")
                else:
                    print("No encoded credential provided")
            
            elif choice == "3":
                print("Goodbye!")
                break
            
            else:
                print("Invalid choice. Please enter 1, 2, or 3.")
    
    else:
        print("Invalid command or arguments.")
        print("Usage:")
        print("  python encode_credentials.py encode <credential>")
        print("  python encode_credentials.py decode <base64_credential>")
        print("  python encode_credentials.py interactive")

if __name__ == "__main__":
    main() 