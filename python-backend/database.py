import os
import logging
from typing import Optional, List, Dict, Any
from supabase import create_client, Client
from datetime import datetime

logger = logging.getLogger(__name__)

class DatabaseClient:
    def __init__(self):
        """Initialize Supabase client with environment variables."""
        self.supabase_url = os.getenv("SUPABASE_URL")
        self.supabase_key = os.getenv("SUPABASE_ANON_KEY")
        
        if not self.supabase_url or not self.supabase_key:
            logger.error("SUPABASE_URL and SUPABASE_ANON_KEY environment variables must be set")
            raise ValueError("Missing Supabase configuration")
        
        self.client: Client = create_client(self.supabase_url, self.supabase_key)
        logger.info("Supabase client initialized successfully")

    async def get_customer_by_account_number(self, account_number: str) -> Optional[Dict[str, Any]]:
        """Get customer details by account number."""
        try:
            response = self.client.table("customers").select("*").eq("account_number", account_number).execute()
            
            if response.data and len(response.data) > 0:
                customer = response.data[0]
                logger.info(f"Found customer: {customer['name']} with account {account_number}")
                return customer
            else:
                logger.warning(f"No customer found with account number: {account_number}")
                return None
                
        except Exception as e:
            logger.error(f"Error fetching customer by account number {account_number}: {e}")
            return None

    async def get_bookings_by_customer_id(self, customer_id: str) -> List[Dict[str, Any]]:
        """Get all bookings for a customer with flight details."""
        try:
            response = self.client.table("bookings").select("""
                *,
                flights (
                    flight_number,
                    origin,
                    destination,
                    scheduled_departure,
                    scheduled_arrival,
                    current_status,
                    gate,
                    terminal,
                    delay_minutes
                )
            """).eq("customer_id", customer_id).order("id", desc=True).execute()
            
            if response.data:
                logger.info(f"Found {len(response.data)} bookings for customer {customer_id}")
                return response.data
            else:
                logger.info(f"No bookings found for customer {customer_id}")
                return []
                
        except Exception as e:
            logger.error(f"Error fetching bookings for customer {customer_id}: {e}")
            return []

    async def get_flight_status(self, flight_number: str) -> Optional[Dict[str, Any]]:
        """Get flight status by flight number."""
        try:
            response = self.client.table("flights").select("*").eq("flight_number", flight_number).execute()
            
            if response.data and len(response.data) > 0:
                flight = response.data[0]
                logger.info(f"Found flight {flight_number} with status: {flight['current_status']}")
                return flight
            else:
                logger.warning(f"No flight found with number: {flight_number}")
                return None
                
        except Exception as e:
            logger.error(f"Error fetching flight status for {flight_number}: {e}")
            return None

    async def update_seat_number(self, confirmation_number: str, new_seat: str) -> bool:
        """Update seat number for a booking."""
        try:
            response = self.client.table("bookings").update({
                "seat_number": new_seat
            }).eq("confirmation_number", confirmation_number).execute()
            
            if response.data and len(response.data) > 0:
                logger.info(f"Updated seat to {new_seat} for confirmation {confirmation_number}")
                return True
            else:
                logger.warning(f"No booking found with confirmation number: {confirmation_number}")
                return False
                
        except Exception as e:
            logger.error(f"Error updating seat for confirmation {confirmation_number}: {e}")
            return False

    async def cancel_booking(self, confirmation_number: str) -> bool:
        """Cancel a booking by updating its status."""
        try:
            response = self.client.table("bookings").update({
                "booking_status": "Cancelled"
            }).eq("confirmation_number", confirmation_number).execute()
            
            if response.data and len(response.data) > 0:
                logger.info(f"Cancelled booking with confirmation {confirmation_number}")
                return True
            else:
                logger.warning(f"No booking found with confirmation number: {confirmation_number}")
                return False
                
        except Exception as e:
            logger.error(f"Error cancelling booking {confirmation_number}: {e}")
            return False

    async def save_conversation(self, session_id: str, history: List[Dict[str, Any]], 
                              context: Dict[str, Any], current_agent: str) -> bool:
        """Save conversation state to Supabase."""
        try:
            # Check if conversation exists
            existing = self.client.table("conversations").select("session_id").eq("session_id", session_id).execute()
            
            conversation_data = {
                "session_id": session_id,
                "history": history,
                "context": context,
                "current_agent": current_agent,
                "last_updated": datetime.now().isoformat()
            }
            
            if existing.data and len(existing.data) > 0:
                # Update existing conversation
                response = self.client.table("conversations").update(conversation_data).eq("session_id", session_id).execute()
            else:
                # Insert new conversation
                response = self.client.table("conversations").insert(conversation_data).execute()
            
            logger.info(f"Saved conversation {session_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving conversation {session_id}: {e}")
            return False

    async def load_conversation(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Load conversation state from Supabase."""
        try:
            response = self.client.table("conversations").select("*").eq("session_id", session_id).execute()
            
            if response.data and len(response.data) > 0:
                conversation = response.data[0]
                logger.info(f"Loaded conversation {session_id}")
                return conversation
            else:
                logger.info(f"No conversation found with session_id: {session_id}")
                return None
                
        except Exception as e:
            logger.error(f"Error loading conversation {session_id}: {e}")
            return None

# Create a global instance
db_client = DatabaseClient()