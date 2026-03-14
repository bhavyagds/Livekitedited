#!/usr/bin/env python3
"""
Test script to validate all API keys and credentials in the environment.
This will help identify any missing or invalid credentials before testing the voice agent.
"""

import os
import sys
import asyncio
from typing import Dict, List, Tuple
from dotenv import load_dotenv

# Color codes for terminal output
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
RESET = '\033[0m'


def print_header(text: str):
    """Print a formatted header."""
    print(f"\n{BLUE}{'=' * 60}{RESET}")
    print(f"{BLUE}{text:^60}{RESET}")
    print(f"{BLUE}{'=' * 60}{RESET}\n")


def print_result(service: str, status: str, message: str):
    """Print a formatted test result."""
    status_color = GREEN if status == "✓" else RED if status == "✗" else YELLOW
    print(f"{status_color}{status}{RESET} {service:20s} {message}")


async def test_openai():
    """Test OpenAI API key."""
    api_key = os.getenv("OPENAI_API_KEY")
    
    if not api_key:
        return False, "API key not set"
    
    if not api_key.startswith("sk-"):
        return False, "Invalid API key format (should start with 'sk-')"
    
    try:
        import httpx
        # Test the API key with a simple request
        async with httpx.AsyncClient() as client:
            response = await client.get(
                "https://api.openai.com/v1/models",
                headers={"Authorization": f"Bearer {api_key}"},
                timeout=10.0
            )
        
        if response.status_code == 200:
            model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
            return True, f"Valid (model: {model})"
        elif response.status_code == 401:
            return False, "Invalid API key"
        else:
            return False, f"HTTP {response.status_code}"
    except Exception as e:
        return False, f"Error: {str(e)[:50]}"


async def test_elevenlabs():
    """Test ElevenLabs API key."""
    try:
        import httpx
        api_key = os.getenv("ELEVENLABS_API_KEY")
        
        if not api_key:
            return False, "API key not set"
        
        if not api_key.startswith("sk_"):
            return False, "Invalid API key format (should start with 'sk_')"
        
        # Test the API key by fetching user info
        async with httpx.AsyncClient() as client:
            response = await client.get(
                "https://api.elevenlabs.io/v1/user",
                headers={"xi-api-key": api_key}
            )
        
        if response.status_code == 200:
            data = response.json()
            voice_id = os.getenv("ELEVENLABS_VOICE_ID", "")
            return True, f"Valid (subscription: {data.get('subscription', {}).get('tier', 'unknown')}, voice: {voice_id[:8]}...)"
        elif response.status_code == 401:
            return False, "Invalid API key"
        else:
            return False, f"HTTP {response.status_code}"
    except Exception as e:
        return False, f"Error: {str(e)[:50]}"


async def test_shopify():
    """Test Shopify credentials."""
    try:
        import httpx
        store_url = os.getenv("SHOPIFY_STORE_URL")
        access_token = os.getenv("SHOPIFY_ACCESS_TOKEN")
        
        if not store_url:
            return False, "SHOPIFY_STORE_URL not set"
        
        if not access_token:
            return False, "SHOPIFY_ACCESS_TOKEN not set"
        
        # Add https:// if not present
        if not store_url.startswith("http"):
            store_url = f"https://{store_url}"
        
        # Test by fetching shop info
        url = f"{store_url}/admin/api/2024-01/shop.json"
        async with httpx.AsyncClient() as client:
            response = await client.get(
                url,
                headers={"X-Shopify-Access-Token": access_token},
                timeout=10.0
            )
        
        if response.status_code == 200:
            data = response.json()
            shop_name = data.get("shop", {}).get("name", "unknown")
            return True, f"Valid (store: {shop_name})"
        elif response.status_code == 401:
            return False, "Invalid access token"
        else:
            return False, f"HTTP {response.status_code}"
    except Exception as e:
        return False, f"Error: {str(e)[:50]}"


async def test_twilio():
    """Test Twilio credentials."""
    try:
        import httpx
        account_sid = os.getenv("TWILIO_ACCOUNT_SID")
        auth_token = os.getenv("TWILIO_AUTH_TOKEN")
        phone_number = os.getenv("TWILIO_PHONE_NUMBER")
        
        if not account_sid:
            return False, "TWILIO_ACCOUNT_SID not set"
        
        if not auth_token:
            return False, "TWILIO_AUTH_TOKEN not set"
        
        if not phone_number:
            return False, "TWILIO_PHONE_NUMBER not set"
        
        # Skip validation if using placeholder values
        if account_sid.startswith("your_"):
            return False, "Using placeholder credentials (not configured)"
        
        # Test by fetching account info via API
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"https://api.twilio.com/2010-04-01/Accounts/{account_sid}.json",
                auth=(account_sid, auth_token),
                timeout=10.0
            )
        
        if response.status_code == 200:
            data = response.json()
            status = data.get("status", "unknown")
            return True, f"Valid (status: {status}, phone: {phone_number})"
        elif response.status_code == 401:
            return False, "Invalid credentials"
        else:
            return False, f"HTTP {response.status_code}"
    except Exception as e:
        return False, f"Error: {str(e)[:50]}"


async def test_smtp():
    """Test SMTP credentials."""
    try:
        import smtplib
        
        smtp_host = os.getenv("SMTP_HOST")
        smtp_port = int(os.getenv("SMTP_PORT", "587"))
        smtp_user = os.getenv("SMTP_USER")
        smtp_pass = os.getenv("SMTP_PASS")
        
        if not smtp_host:
            return False, "SMTP_HOST not set"
        
        if not smtp_user:
            return False, "SMTP_USER not set"
        
        if not smtp_pass:
            return False, "SMTP_PASS not set"
        
        # Skip validation if using placeholder values
        if smtp_user.startswith("your_"):
            return False, "Using placeholder credentials (not configured)"
        
        # Test by connecting and authenticating (synchronous)
        try:
            with smtplib.SMTP(smtp_host, smtp_port, timeout=10) as smtp:
                smtp.starttls()
                smtp.login(smtp_user, smtp_pass)
            return True, f"Valid (server: {smtp_host}:{smtp_port})"
        except smtplib.SMTPAuthenticationError:
            return False, "Invalid credentials"
        except Exception as e:
            return False, f"Connection error: {str(e)[:40]}"
    except Exception as e:
        return False, f"Error: {str(e)[:50]}"


async def test_livekit():
    """Test LiveKit configuration."""
    try:
        import httpx
        livekit_url = os.getenv("LIVEKIT_PUBLIC_URL", "ws://localhost:7880")
        livekit_api_key = os.getenv("LIVEKIT_API_KEY")
        livekit_api_secret = os.getenv("LIVEKIT_API_SECRET")
        
        if not livekit_api_key:
            return False, "LIVEKIT_API_KEY not set"
        
        if not livekit_api_secret:
            return False, "LIVEKIT_API_SECRET not set"
        
        # Convert ws:// to http:// for health check
        http_url = livekit_url.replace("ws://", "http://").replace("wss://", "https://")
        
        # Test by checking server health
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{http_url}/", timeout=5.0)
        
        if response.status_code == 200 or response.status_code == 404:
            return True, f"Server reachable (url: {livekit_url})"
        else:
            return False, f"Server returned HTTP {response.status_code}"
    except Exception as e:
        return False, f"Error: {str(e)[:50]}"


async def check_environment_variables():
    """Check for required environment variables."""
    print_header("ENVIRONMENT VARIABLES CHECK")
    
    required_vars = {
        "Core": [
            "LIVEKIT_URL",
            "LIVEKIT_API_KEY",
            "LIVEKIT_API_SECRET",
        ],
        "AI Services": [
            "OPENAI_API_KEY",
            "OPENAI_MODEL",
            "ELEVENLABS_API_KEY",
            "ELEVENLABS_VOICE_ID",
            "ELEVENLABS_MODEL",
        ],
        "Integrations": [
            "SHOPIFY_STORE_URL",
            "SHOPIFY_ACCESS_TOKEN",
            "TWILIO_ACCOUNT_SID",
            "TWILIO_AUTH_TOKEN",
            "TWILIO_PHONE_NUMBER",
        ],
        "Email": [
            "SMTP_HOST",
            "SMTP_PORT",
            "SMTP_USER",
            "SMTP_PASS",
            "SUPPORT_EMAIL",
        ],
    }
    
    all_set = True
    for category, vars_list in required_vars.items():
        print(f"\n{YELLOW}{category}:{RESET}")
        for var in vars_list:
            value = os.getenv(var)
            if value:
                # Mask sensitive values
                if "KEY" in var or "SECRET" in var or "TOKEN" in var or "PASS" in var:
                    display_value = f"{value[:8]}...{value[-4:]}" if len(value) > 12 else "***"
                else:
                    display_value = value[:50] + "..." if len(value) > 50 else value
                print_result(var, "✓", f"Set ({display_value})")
            else:
                print_result(var, "✗", "NOT SET")
                all_set = False
    
    return all_set


async def test_all_services():
    """Test all external services."""
    print_header("API CREDENTIALS VALIDATION")
    
    tests = [
        ("OpenAI", test_openai),
        ("ElevenLabs", test_elevenlabs),
        ("Shopify", test_shopify),
        ("Twilio", test_twilio),
        ("SMTP", test_smtp),
        ("LiveKit", test_livekit),
    ]
    
    results = []
    for service_name, test_func in tests:
        try:
            success, message = await test_func()
            print_result(service_name, "✓" if success else "✗", message)
            results.append((service_name, success, message))
        except Exception as e:
            print_result(service_name, "✗", f"Test failed: {str(e)[:50]}")
            results.append((service_name, False, str(e)))
    
    return results


async def main():
    """Main test function."""
    print_header("MEALLION AI CALL AGENT - CREDENTIALS TEST")
    
    # Load environment variables
    if os.path.exists(".env"):
        load_dotenv()
        print(f"{GREEN}✓{RESET} Loaded .env file\n")
    else:
        print(f"{RED}✗{RESET} No .env file found\n")
        print("Please create a .env file based on env.example\n")
        return 1
    
    # Check environment variables
    env_check = await check_environment_variables()
    
    # Test all services
    test_results = await test_all_services()
    
    # Print summary
    print_header("SUMMARY")
    
    total_tests = len(test_results)
    passed_tests = sum(1 for _, success, _ in test_results if success)
    failed_tests = total_tests - passed_tests
    
    print(f"Total Tests:  {total_tests}")
    print(f"{GREEN}Passed:       {passed_tests}{RESET}")
    if failed_tests > 0:
        print(f"{RED}Failed:       {failed_tests}{RESET}")
    
    if not env_check:
        print(f"\n{YELLOW}⚠{RESET}  Some environment variables are not set")
    
    if failed_tests > 0:
        print(f"\n{RED}✗{RESET} Some credentials are invalid or services are unreachable")
        print("\nFailed services:")
        for service, success, message in test_results:
            if not success:
                print(f"  - {service}: {message}")
        return 1
    else:
        print(f"\n{GREEN}✓{RESET} All credentials are valid and services are reachable!")
        print(f"\n{GREEN}→{RESET} You can now test the voice agent at http://localhost:3000")
        return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
