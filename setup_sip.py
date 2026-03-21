#!/usr/bin/env python3
"""
LiveKit SIP Setup Script

This script configures SIP trunks and dispatch rules for phone call integration.
Run this after setting up your SIP provider (Twilio, Telnyx, etc.)

Usage:
    python scripts/setup_sip.py --provider twilio
    python scripts/setup_sip.py --provider telnyx
"""

import os
import sys
import argparse
import asyncio
from livekit import api

# Configuration
LIVEKIT_URL = os.getenv("LIVEKIT_URL", "ws://localhost:7880")
LIVEKIT_API_KEY = os.getenv("LIVEKIT_API_KEY", "devkey")
LIVEKIT_API_SECRET = os.getenv("LIVEKIT_API_SECRET", "secret")

# Your SIP credentials (set these in .env)
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID", "")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN", "")
TWILIO_PHONE_NUMBER = os.getenv("TWILIO_PHONE_NUMBER", "")

TELNYX_SIP_USERNAME = os.getenv("TELNYX_SIP_USERNAME", "")
TELNYX_SIP_PASSWORD = os.getenv("TELNYX_SIP_PASSWORD", "")

# Yuboto (Greek VoIP provider)
YUBOTO_SIP_SERVER = os.getenv("YUBOTO_SIP_SERVER", "sip.yuboto.com")
YUBOTO_SIP_USERNAME = os.getenv("YUBOTO_SIP_USERNAME", "")
YUBOTO_SIP_PASSWORD = os.getenv("YUBOTO_SIP_PASSWORD", "")
YUBOTO_PHONE_NUMBER = os.getenv("YUBOTO_PHONE_NUMBER", "")
YUBOTO_ALLOWED_IPS = os.getenv("YUBOTO_ALLOWED_IPS", "")


def _parse_allowed_ips(raw: str) -> list:
    if not raw:
        return []
    return [ip.strip() for ip in raw.split(",") if ip.strip()]


async def create_twilio_trunk():
    """Create a SIP trunk for Twilio."""
    print("Creating Twilio SIP Trunk...")
    
    # Create LiveKit API client
    lk_api = api.LiveKitAPI(
        LIVEKIT_URL.replace("ws://", "http://").replace("wss://", "https://"),
        LIVEKIT_API_KEY,
        LIVEKIT_API_SECRET,
    )
    
    # Create inbound trunk (for receiving calls)
    inbound_trunk = api.SIPInboundTrunkInfo(
        name="Twilio Inbound",
        numbers=[TWILIO_PHONE_NUMBER],
        # Twilio's SIP domain for your account
        # Format: {account_sid}.sip.twilio.com
        allowed_addresses=[f"{TWILIO_ACCOUNT_SID}.sip.twilio.com"],
        # Auth header for verification
        auth_username=TWILIO_ACCOUNT_SID,
        auth_password=TWILIO_AUTH_TOKEN,
    )
    
    try:
        result = await lk_api.sip.create_sip_inbound_trunk(inbound_trunk)
        print(f"✅ Created inbound trunk: {result.sip_trunk_id}")
        return result.sip_trunk_id
    except Exception as e:
        print(f"❌ Error creating trunk: {e}")
        return None


async def create_telnyx_trunk():
    """Create a SIP trunk for Telnyx."""
    print("Creating Telnyx SIP Trunk...")
    
    lk_api = api.LiveKitAPI(
        LIVEKIT_URL.replace("ws://", "http://").replace("wss://", "https://"),
        LIVEKIT_API_KEY,
        LIVEKIT_API_SECRET,
    )
    
    inbound_trunk = api.SIPInboundTrunkInfo(
        name="Telnyx Inbound",
        numbers=[os.getenv("TELNYX_PHONE_NUMBER", "")],
        allowed_addresses=["sip.telnyx.com"],
        auth_username=TELNYX_SIP_USERNAME,
        auth_password=TELNYX_SIP_PASSWORD,
    )
    
    try:
        result = await lk_api.sip.create_sip_inbound_trunk(inbound_trunk)
        print(f"✅ Created inbound trunk: {result.sip_trunk_id}")
        return result.sip_trunk_id
    except Exception as e:
        print(f"❌ Error creating trunk: {e}")
        return None


async def create_yuboto_trunk():
    """Create a SIP trunk for Yuboto (Greek VoIP provider)."""
    print("Creating Yuboto SIP Trunk...")
    print(f"  Server: {YUBOTO_SIP_SERVER}")
    print(f"  Phone: {YUBOTO_PHONE_NUMBER}")
    
    lk_api = api.LiveKitAPI(
        LIVEKIT_URL.replace("ws://", "http://").replace("wss://", "https://"),
        LIVEKIT_API_KEY,
        LIVEKIT_API_SECRET,
    )
    
    # Yuboto SIP trunk configuration
    inbound_trunk = api.SIPInboundTrunkInfo(
        name="Yuboto Greece",
        numbers=[YUBOTO_PHONE_NUMBER],
        # Allow traffic from Yuboto's SIP servers
        allowed_addresses=[YUBOTO_SIP_SERVER, "*.yuboto.com"] + _parse_allowed_ips(YUBOTO_ALLOWED_IPS),
        auth_username=YUBOTO_SIP_USERNAME,
        auth_password=YUBOTO_SIP_PASSWORD,
    )
    
    try:
        result = await lk_api.sip.create_sip_inbound_trunk(inbound_trunk)
        print(f"✅ Created inbound trunk: {result.sip_trunk_id}")
        return result.sip_trunk_id
    except Exception as e:
        print(f"❌ Error creating trunk: {e}")
        return None


async def create_dispatch_rule(trunk_id: str):
    """
    Create a dispatch rule to route incoming SIP calls to the voice agent.
    
    This tells LiveKit to:
    1. Create a room for each incoming call
    2. Dispatch to an agent worker
    """
    print("Creating SIP Dispatch Rule...")
    
    lk_api = api.LiveKitAPI(
        LIVEKIT_URL.replace("ws://", "http://").replace("wss://", "https://"),
        LIVEKIT_API_KEY,
        LIVEKIT_API_SECRET,
    )
    
    # Create dispatch rule - route all calls to agent
    dispatch_rule = api.SIPDispatchRuleInfo(
        name="Route to Elena Agent",
        trunk_ids=[trunk_id],
        # Create a room with the caller's number as the room name
        rule=api.SIPDispatchRuleDirect(
            room_name="sip-${caller.number}",
            pin="",  # No PIN required
        ),
        # Metadata to pass to the agent
        metadata='{"source": "phone", "type": "sip"}',
    )
    
    try:
        result = await lk_api.sip.create_sip_dispatch_rule(dispatch_rule)
        print(f"✅ Created dispatch rule: {result.sip_dispatch_rule_id}")
        return result.sip_dispatch_rule_id
    except Exception as e:
        print(f"❌ Error creating dispatch rule: {e}")
        return None


async def list_sip_config():
    """List current SIP configuration."""
    print("\n📋 Current SIP Configuration:")
    print("-" * 40)
    
    lk_api = api.LiveKitAPI(
        LIVEKIT_URL.replace("ws://", "http://").replace("wss://", "https://"),
        LIVEKIT_API_KEY,
        LIVEKIT_API_SECRET,
    )
    
    try:
        # List inbound trunks
        trunks = await lk_api.sip.list_sip_inbound_trunk()
        print(f"\nInbound Trunks ({len(trunks.items)}):")
        for trunk in trunks.items:
            print(f"  - {trunk.name} ({trunk.sip_trunk_id})")
            print(f"    Numbers: {', '.join(trunk.numbers)}")
        
        # List dispatch rules
        rules = await lk_api.sip.list_sip_dispatch_rule()
        print(f"\nDispatch Rules ({len(rules.items)}):")
        for rule in rules.items:
            print(f"  - {rule.name} ({rule.sip_dispatch_rule_id})")
            
    except Exception as e:
        print(f"❌ Error listing config: {e}")


async def main():
    parser = argparse.ArgumentParser(description="Setup LiveKit SIP")
    parser.add_argument(
        "--provider",
        choices=["twilio", "telnyx", "yuboto", "list"],
        default="list",
        help="SIP provider to configure",
    )
    args = parser.parse_args()
    
    print("=" * 50)
    print("🔧 LiveKit SIP Setup")
    print("=" * 50)
    print(f"LiveKit URL: {LIVEKIT_URL}")
    print()
    
    if args.provider == "list":
        await list_sip_config()
        return
    
    trunk_id = None
    
    if args.provider == "twilio":
        if not TWILIO_ACCOUNT_SID or not TWILIO_AUTH_TOKEN:
            print("❌ Error: TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN required")
            print("   Set them in your .env file")
            sys.exit(1)
        
        trunk_id = await create_twilio_trunk()
        
    elif args.provider == "telnyx":
        if not TELNYX_SIP_USERNAME:
            print("❌ Error: TELNYX_SIP_USERNAME required")
            sys.exit(1)
        
        trunk_id = await create_telnyx_trunk()
    
    elif args.provider == "yuboto":
        if not YUBOTO_SIP_USERNAME or not YUBOTO_SIP_PASSWORD:
            print("❌ Error: YUBOTO_SIP_USERNAME and YUBOTO_SIP_PASSWORD required")
            print("   Set them in your .env file:")
            print("   YUBOTO_SIP_SERVER=sip.yuboto.com")
            print("   YUBOTO_SIP_USERNAME=your_username")
            print("   YUBOTO_SIP_PASSWORD=your_password")
            print("   YUBOTO_PHONE_NUMBER=+302XXXXXXXXX")
            sys.exit(1)
        
        trunk_id = await create_yuboto_trunk()
    
    if trunk_id:
        await create_dispatch_rule(trunk_id)
        print("\n✅ SIP configuration complete!")
        print("\n📱 Next Steps:")
        if args.provider == "yuboto":
            print("1. Log into Yuboto panel: https://services.yuboto.com/mynumber/Index.aspx")
            print("2. Configure your number to forward to your VPS:")
            print(f"   SIP URI: sip:YOUR_VPS_IP:5060")
        else:
            print("1. Configure your SIP provider to send calls to your server")
            print(f"   SIP URI: sip:YOUR_VPS_IP:5060")
        print("3. Test by calling your phone number")
        print("4. The call will connect to Elena!")
    else:
        print("\n❌ Setup failed. Check your credentials.")


if __name__ == "__main__":
    asyncio.run(main())
