"""
LiveKit SIP Management Service
Handles SIP trunk and dispatch rule configuration via LiveKit API.
"""

import logging
import asyncio
import re
import socket
from typing import List, Dict, Optional, Any, Tuple

logger = logging.getLogger(__name__)


def validate_phone_number(number: str) -> Tuple[bool, str]:
    """Validate phone number format (E.164)."""
    # Remove spaces and dashes
    cleaned = re.sub(r'[\s\-\(\)]', '', number)
    
    # Check E.164 format: + followed by 1-15 digits
    if re.match(r'^\+[1-9]\d{1,14}$', cleaned):
        return True, cleaned
    
    # Allow numbers without + for local formats
    if re.match(r'^[1-9]\d{6,14}$', cleaned):
        return True, f"+{cleaned}"
    
    return False, f"Invalid phone format: {number}"


def validate_server_address(server: str) -> Tuple[bool, str]:
    """Validate SIP server address."""
    if not server or len(server) < 3:
        return False, "Server address is required"
    
    # Remove protocol if present
    server = re.sub(r'^(sip:|sips:|udp:|tcp:|tls:)', '', server.lower())
    
    # Check if it's an IP address
    ip_pattern = r'^(\d{1,3}\.){3}\d{1,3}(:\d+)?$'
    if re.match(ip_pattern, server):
        parts = server.split(':')[0].split('.')
        if all(0 <= int(p) <= 255 for p in parts):
            return True, server
        return False, "Invalid IP address"
    
    # Check if it's a valid domain
    domain_pattern = r'^[a-z0-9]([a-z0-9\-]*[a-z0-9])?(\.[a-z0-9]([a-z0-9\-]*[a-z0-9])?)*\.[a-z]{2,}(:\d+)?$'
    if re.match(domain_pattern, server):
        return True, server
    
    return False, f"Invalid server address: {server}"


def validate_credentials(username: str, password: str) -> Tuple[bool, str]:
    """Validate SIP credentials (optional for inbound-only trunks)."""
    # Credentials are optional for inbound-only trunks (IP-based auth)
    if not username and not password:
        return True, "OK (no auth - IP-based)"
    
    # If one is provided, both should be provided
    if username and not password:
        return False, "Password is required when username is provided"
    if password and not username:
        return False, "Username is required when password is provided"

    # Check for common invalid characters
    if re.search(r'[\x00-\x1f]', (username or "") + (password or "")):
        return False, "Credentials contain invalid characters"
    
    return True, "OK"


class LiveKitSIPService:
    """Service to manage LiveKit SIP configuration without restarts."""
    
    def __init__(self):
        self._api = None
        self._initialized = False
    
    async def _get_api(self):
        """Get or create LiveKit API client."""
        if self._api is None:
            try:
                from livekit import api
                from src.config import settings
                
                # Convert ws:// to http:// for API calls
                api_url = settings.livekit_url.replace("ws://", "http://").replace("wss://", "https://")
                
                self._api = api.LiveKitAPI(
                    api_url,
                    settings.livekit_api_key,
                    settings.livekit_api_secret,
                )
                self._initialized = True
                logger.info(f"LiveKit API client initialized: {api_url}")
            except Exception as e:
                logger.error(f"Failed to initialize LiveKit API: {e}")
                raise
        return self._api
    
    async def list_inbound_trunks(self) -> List[Dict]:
        """List all SIP inbound trunks."""
        try:
            from livekit import api
            
            lk_api = await self._get_api()
            
            # Create proper request object
            request = api.ListSIPInboundTrunkRequest()
            response = await lk_api.sip.list_sip_inbound_trunk(request)
            
            trunks = []
            for t in response.items:
                trunks.append({
                    "id": t.sip_trunk_id,
                    "name": t.name,
                    "numbers": list(t.numbers) if t.numbers else [],
                    "allowed_addresses": list(t.allowed_addresses) if t.allowed_addresses else [],
                    "metadata": t.metadata,
                })
            return trunks
        except Exception as e:
            logger.error(f"Failed to list inbound trunks: {e}")
            return []
    
    async def create_inbound_trunk(
        self,
        name: str,
        numbers: List[str],
        allowed_addresses: List[str],
        auth_username: Optional[str] = None,
        auth_password: Optional[str] = None,
        metadata: Optional[str] = None,
    ) -> Optional[Dict]:
        """Create a new SIP inbound trunk."""
        try:
            from livekit import api
            
            lk_api = await self._get_api()
            
            # Create trunk info
            trunk_info = api.SIPInboundTrunkInfo(
                name=name,
                numbers=numbers,
                allowed_addresses=allowed_addresses,
                auth_username=auth_username or "",
                auth_password=auth_password or "",
                metadata=metadata or "",
            )
            
            # Create request with trunk
            request = api.CreateSIPInboundTrunkRequest(trunk=trunk_info)
            result = await lk_api.sip.create_sip_inbound_trunk(request)
            
            logger.info(f"Created SIP inbound trunk: {result.sip_trunk_id}")
            
            return {
                "id": result.sip_trunk_id,
                "name": result.name,
                "numbers": list(result.numbers) if result.numbers else [],
            }
        except Exception as e:
            logger.error(f"Failed to create inbound trunk: {e}")
            return None
    
    async def delete_inbound_trunk(self, trunk_id: str) -> bool:
        """Delete a SIP inbound trunk."""
        try:
            from livekit import api
            
            lk_api = await self._get_api()
            request = api.DeleteSIPTrunkRequest(sip_trunk_id=trunk_id)
            await lk_api.sip.delete_sip_trunk(request)
            logger.info(f"Deleted SIP trunk: {trunk_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete trunk {trunk_id}: {e}")
            return False
    
    async def list_dispatch_rules(self) -> List[Dict]:
        """List all SIP dispatch rules."""
        try:
            from livekit import api
            
            lk_api = await self._get_api()
            
            # Create proper request object
            request = api.ListSIPDispatchRuleRequest()
            response = await lk_api.sip.list_sip_dispatch_rule(request)
            
            rules = []
            for r in response.items:
                rules.append({
                    "id": r.sip_dispatch_rule_id,
                    "name": r.name,
                    "trunk_ids": list(r.trunk_ids) if r.trunk_ids else [],
                    "metadata": r.metadata,
                })
            return rules
        except Exception as e:
            logger.error(f"Failed to list dispatch rules: {e}")
            return []
    
    async def create_dispatch_rule(
        self,
        name: str,
        room_name_template: str = "sip-${caller.number}",
        trunk_ids: Optional[List[str]] = None,
        metadata: Optional[str] = None,
    ) -> Optional[Dict]:
        """Create a SIP dispatch rule to route calls to rooms."""
        try:
            from livekit import api
            
            lk_api = await self._get_api()
            
            # Create the dispatch rule wrapper with direct rule inside
            dispatch_rule = api.SIPDispatchRule(
                dispatch_rule_direct=api.SIPDispatchRuleDirect(
                    room_name=room_name_template,
                    pin="",  # No PIN required
                )
            )
            
            # Create request directly with fields (not using SIPDispatchRuleInfo wrapper)
            request = api.CreateSIPDispatchRuleRequest(
                rule=dispatch_rule,
                trunk_ids=trunk_ids or [],
                name=name,
                metadata=metadata or '{"source": "phone", "type": "sip"}',
            )
            result = await lk_api.sip.create_sip_dispatch_rule(request)
            
            logger.info(f"Created dispatch rule: {result.sip_dispatch_rule_id}")
            
            return {
                "id": result.sip_dispatch_rule_id,
                "name": result.name,
            }
        except Exception as e:
            logger.error(f"Failed to create dispatch rule: {e}")
            return None
    
    async def delete_dispatch_rule(self, rule_id: str) -> bool:
        """Delete a SIP dispatch rule."""
        try:
            from livekit import api
            
            lk_api = await self._get_api()
            request = api.DeleteSIPDispatchRuleRequest(sip_dispatch_rule_id=rule_id)
            await lk_api.sip.delete_sip_dispatch_rule(request)
            logger.info(f"Deleted dispatch rule: {rule_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete dispatch rule {rule_id}: {e}")
            return False
    
    async def get_sip_status(self) -> Dict:
        """Get current SIP configuration status."""
        trunks = await self.list_inbound_trunks()
        rules = await self.list_dispatch_rules()
        
        return {
            "trunks_count": len(trunks),
            "trunks": trunks,
            "rules_count": len(rules),
            "rules": rules,
            "status": "configured" if trunks else "not_configured",
        }
    
    async def validate_provider_config(
        self,
        provider_name: str,
        server: str,
        username: str,
        password: str,
        phone_numbers: List[str],
    ) -> Dict:
        """Validate SIP provider configuration before creating."""
        errors = []
        warnings = []
        validated_numbers = []
        
        # Validate provider name
        if not provider_name or len(provider_name) < 2:
            errors.append("Provider name must be at least 2 characters")
        
        # Validate server
        server_valid, server_result = validate_server_address(server)
        if not server_valid:
            errors.append(server_result)
        
        # Validate credentials
        creds_valid, creds_result = validate_credentials(username, password)
        if not creds_valid:
            errors.append(creds_result)
        
        # Validate phone numbers
        if not phone_numbers or len(phone_numbers) == 0:
            warnings.append("No phone numbers specified - trunk will accept calls from any number")
        else:
            for num in phone_numbers:
                if not num.strip():
                    continue
                valid, result = validate_phone_number(num.strip())
                if valid:
                    validated_numbers.append(result)
                else:
                    errors.append(result)
        
        # Check for duplicate numbers in existing trunks
        if validated_numbers:
            existing_trunks = await self.list_inbound_trunks()
            for trunk in existing_trunks:
                for existing_num in trunk.get("numbers", []):
                    if existing_num in validated_numbers:
                        errors.append(f"Phone number {existing_num} is already configured in trunk '{trunk['name']}'")
        
        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings,
            "validated_numbers": validated_numbers,
            "validated_server": server_result if server_valid else None,
        }
    
    async def test_livekit_connection(self) -> Dict:
        """Test connection to LiveKit SIP service."""
        try:
            lk_api = await self._get_api()
            
            # Try to list trunks as a connection test
            from livekit import api
            request = api.ListSIPInboundTrunkRequest()
            await lk_api.sip.list_sip_inbound_trunk(request)
            
            return {
                "connected": True,
                "message": "Successfully connected to LiveKit SIP service",
            }
        except Exception as e:
            return {
                "connected": False,
                "message": f"Failed to connect to LiveKit: {str(e)}",
            }
    
    async def configure_provider(
        self,
        provider_name: str,
        server: str,
        username: str,
        password: str,
        phone_numbers: List[str],
        allowed_ips: List[str] = None,
        skip_validation: bool = False,
    ) -> Dict:
        """Configure a SIP provider (creates trunk and dispatch rule)."""
        try:
            # Validate configuration first
            if not skip_validation:
                validation = await self.validate_provider_config(
                    provider_name, server, username, password, phone_numbers
                )
                if not validation["valid"]:
                    return {
                        "success": False,
                        "error": "Validation failed",
                        "validation_errors": validation["errors"],
                        "warnings": validation.get("warnings", []),
                    }
                
                # Use validated values
                phone_numbers = validation["validated_numbers"]
                server = validation["validated_server"] or server
            
            # Build allowed addresses from provided list or use defaults
            allowed_addresses = [server]
            server_parts = server.split('.')
            if len(server_parts) >= 2:
                # Add wildcard for subdomain matching
                allowed_addresses.append(f"*.{server_parts[-2]}.{server_parts[-1]}")
            
            # Use provided allowed_ips if available
            if allowed_ips and len(allowed_ips) > 0:
                allowed_addresses.extend(allowed_ips)
            else:
                # Default: allow all IPs
                allowed_addresses.append("0.0.0.0/0")
            
            # Create inbound trunk
            # For inbound-only trunks (like Twilio Origination), we typically don't need auth
            # The provider sends calls FROM their IPs, we accept based on allowed_addresses
            # Only set auth if username is provided AND it's not a placeholder
            use_auth = username and password and username.lower() not in ('none', 'skip', '', 'na', 'n/a')
            
            trunk = await self.create_inbound_trunk(
                name=f"{provider_name} Trunk",
                numbers=phone_numbers,
                allowed_addresses=allowed_addresses,
                auth_username=username if use_auth else None,
                auth_password=password if use_auth else None,
                metadata=f'{{"provider": "{provider_name}"}}',
            )
            
            if not trunk:
                return {"success": False, "error": "Failed to create trunk"}
            
            # Create dispatch rule
            rule = await self.create_dispatch_rule(
                name=f"{provider_name} Dispatch",
                room_name_template="sip-call-${caller.number}",
                trunk_ids=[trunk["id"]],
                metadata=f'{{"provider": "{provider_name}", "source": "phone"}}',
            )
            
            if not rule:
                # Rollback trunk creation
                await self.delete_inbound_trunk(trunk["id"])
                return {"success": False, "error": "Failed to create dispatch rule"}
            
            logger.info(f"Configured SIP provider: {provider_name}")
            
            return {
                "success": True,
                "trunk_id": trunk["id"],
                "rule_id": rule["id"],
                "message": f"Provider {provider_name} configured successfully",
            }
            
        except Exception as e:
            logger.error(f"Failed to configure provider {provider_name}: {e}")
            return {"success": False, "error": str(e)}


    async def sync_providers_from_db(self) -> Dict:
        """
        Sync all SIP providers from database to LiveKit.
        Called on API startup to restore configuration after restarts.
        """
        from src.services.database import get_database_service
        db = get_database_service()
        
        results = {
            "synced": 0,
            "failed": 0,
            "errors": [],
        }
        
        try:
            # Get all active providers from database
            providers = await db.get_all_sip_providers_with_credentials()
            logger.info(f"Found {len(providers)} SIP providers to sync")
            
            if not providers:
                return results
            
            # Get existing trunks to avoid duplicates
            existing_trunks = await self.list_inbound_trunks()
            existing_trunk_names = {t["name"] for t in existing_trunks}
            
            for provider in providers:
                provider_name = provider["name"]
                trunk_name = f"{provider_name} Trunk"
                
                try:
                    # Check if trunk already exists
                    if trunk_name in existing_trunk_names:
                        logger.info(f"Trunk '{trunk_name}' already exists, skipping")
                        await db.update_sip_provider_sync(
                            provider["id"],
                            sync_status="synced",
                            sync_error=None,
                        )
                        results["synced"] += 1
                        continue
                    
                    # Create trunk and rule
                    result = await self.configure_provider(
                        provider_name=provider_name,
                        server=provider["server"],
                        username=provider["username"],
                        password=provider["password"],
                        phone_numbers=provider["phone_numbers"],
                        allowed_ips=provider.get("allowed_ips", []),
                        skip_validation=True,  # Skip validation since it was validated on creation
                    )
                    
                    if result.get("success"):
                        await db.update_sip_provider_sync(
                            provider["id"],
                            livekit_trunk_id=result.get("trunk_id"),
                            livekit_rule_id=result.get("rule_id"),
                            sync_status="synced",
                            sync_error=None,
                        )
                        results["synced"] += 1
                        logger.info(f"Synced SIP provider: {provider_name}")
                    else:
                        error = result.get("error", "Unknown error")
                        await db.update_sip_provider_sync(
                            provider["id"],
                            sync_status="failed",
                            sync_error=error,
                        )
                        results["failed"] += 1
                        results["errors"].append(f"{provider_name}: {error}")
                        logger.error(f"Failed to sync SIP provider {provider_name}: {error}")
                        
                except Exception as e:
                    error = str(e)
                    await db.update_sip_provider_sync(
                        provider["id"],
                        sync_status="failed",
                        sync_error=error,
                    )
                    results["failed"] += 1
                    results["errors"].append(f"{provider_name}: {error}")
                    logger.error(f"Error syncing SIP provider {provider_name}: {e}")
            
            logger.info(f"SIP sync complete: {results['synced']} synced, {results['failed']} failed")
            return results
            
        except Exception as e:
            logger.error(f"Error during SIP provider sync: {e}")
            results["errors"].append(str(e))
            return results


# Global instance
_sip_service: Optional[LiveKitSIPService] = None


def get_sip_service() -> LiveKitSIPService:
    """Get the global SIP service instance."""
    global _sip_service
    if _sip_service is None:
        _sip_service = LiveKitSIPService()
    return _sip_service


async def sync_sip_providers_on_startup() -> Dict:
    """
    Called on application startup to sync SIP providers from DB to LiveKit.
    """
    try:
        sip_service = get_sip_service()
        result = await sip_service.sync_providers_from_db()
        return result
    except Exception as e:
        logger.error(f"Failed to sync SIP providers on startup: {e}")
        return {"synced": 0, "failed": 0, "errors": [str(e)]}
