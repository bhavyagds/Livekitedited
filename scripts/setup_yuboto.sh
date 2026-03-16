#!/bin/bash
# ===========================================
# Yuboto SIP Setup for LiveKit
# Greek VoIP Provider Integration
# ===========================================

set -e

echo "=========================================="
echo "🇬🇷 Yuboto SIP Setup for Meallion"
echo "=========================================="

# Check if livekit-cli is installed
if ! command -v livekit-cli &> /dev/null; then
    echo "📦 Installing livekit-cli..."
    curl -sSL https://get.livekit.io/cli | bash
fi

# Load environment variables
if [ -f .env ]; then
    source .env
fi

# Check required variables
if [ -z "$YUBOTO_SIP_USERNAME" ]; then
    echo "❌ Error: YUBOTO_SIP_USERNAME not set in .env"
    echo ""
    echo "Please add these to your .env file:"
    echo "  YUBOTO_SIP_SERVER=sip.yuboto.com"
    echo "  YUBOTO_SIP_USERNAME=your_username"
    echo "  YUBOTO_SIP_PASSWORD=your_password"
    echo "  YUBOTO_PHONE_NUMBER=+302XXXXXXXXX"
    exit 1
fi

# Set LiveKit connection
LIVEKIT_URL=${LIVEKIT_URL:-"http://localhost:7880"}
LIVEKIT_API_KEY=${LIVEKIT_API_KEY:-"devkey"}
LIVEKIT_API_SECRET=${LIVEKIT_API_SECRET:-"secret"}

export LIVEKIT_URL
export LIVEKIT_API_KEY
export LIVEKIT_API_SECRET

echo ""
echo "📋 Configuration:"
echo "   LiveKit URL: $LIVEKIT_URL"
echo "   Yuboto Server: ${YUBOTO_SIP_SERVER:-sip.yuboto.com}"
echo "   Phone Number: $YUBOTO_PHONE_NUMBER"
echo ""

# Create inbound SIP trunk for Yuboto
echo "🔧 Creating Yuboto Inbound SIP Trunk..."

TRUNK_RESULT=$(livekit-cli sip inbound create \
    --name "Yuboto Greece" \
    --numbers "$YUBOTO_PHONE_NUMBER" \
    --allowed-addresses "sip.yuboto.com,*.yuboto.com" \
    --auth-username "$YUBOTO_SIP_USERNAME" \
    --auth-password "$YUBOTO_SIP_PASSWORD" \
    2>&1)

echo "$TRUNK_RESULT"

# Extract trunk ID
TRUNK_ID=$(echo "$TRUNK_RESULT" | grep -o 'ST_[a-zA-Z0-9]*' | head -1)

if [ -n "$TRUNK_ID" ]; then
    echo "✅ Created trunk: $TRUNK_ID"
    
    # Create dispatch rule
    echo ""
    echo "🔧 Creating Dispatch Rule..."
    
    livekit-cli sip dispatch create \
        --name "Route to Elena" \
        --rule-direct "sip-call-{caller.number}" \
        --trunk-ids "$TRUNK_ID"
    
    echo ""
    echo "✅ SIP Configuration Complete!"
    echo ""
    echo "=========================================="
    echo "📱 Next Steps:"
    echo "=========================================="
    echo ""
    echo "1. Log into Yuboto panel: https://services.yuboto.com/mynumber/Index.aspx"
    echo ""
    echo "2. Configure your number to forward calls to:"
    echo "   SIP URI: sip:YOUR_VPS_IP:5060"
    echo "   (Replace YOUR_VPS_IP with your server's public IP)"
    echo ""
    echo "3. Test by calling your Yuboto number!"
    echo ""
else
    echo "❌ Failed to create trunk. Check your credentials."
    exit 1
fi
