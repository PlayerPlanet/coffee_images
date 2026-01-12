"""
Bot Poller for Coffee Images Time-Series Collection

This script polls @StufeBot every 5-10 minutes with the /status command
to collect coffee pot images from the FK guildroom RasPi camera.
Each /status command triggers a new photo capture.
"""

import asyncio
import json
import logging
import os
import signal
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import dotenv
from telethon import TelegramClient
from telethon.errors import TimeoutError as TelethonTimeoutError

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Load environment variables
dotenv.load_dotenv()

# Configuration
API_ID = os.getenv("api_id")
API_HASH = os.getenv("api_hash")
BOT_TOKEN = os.getenv("BOT_TOKEN")  # Bot token for bot authentication (alternative to user auth)
BOT_USERNAME = os.getenv("BOT_USERNAME", "@StufeBot")
POLL_INTERVAL_SECONDS = int(os.getenv("POLL_INTERVAL_SECONDS", "300"))  # Default 5 minutes
DATA_DIR = Path(os.getenv("DATA_DIR", "data"))
IMAGES_DIR = DATA_DIR / "raw_img"
METADATA_FILE = DATA_DIR / "image_metadata.json"
SESSION_DIR = Path(os.getenv("SESSION_DIR", ".telethon"))
RESPONSE_TIMEOUT = int(os.getenv("RESPONSE_TIMEOUT", "30"))  # 30 seconds for bot to respond
POLL_START_HOUR = int(os.getenv("POLL_START_HOUR", "8"))  # Start polling at 8:00 AM
POLL_END_HOUR = int(os.getenv("POLL_END_HOUR", "18"))  # Stop polling at 6:00 PM

# Global flag for graceful shutdown
shutdown_flag = False


def is_within_polling_hours() -> bool:
    """Check if current local time is within configured polling hours"""
    now = datetime.now()
    current_hour = now.hour
    return POLL_START_HOUR <= current_hour < POLL_END_HOUR


def seconds_until_next_poll_window() -> int:
    """Calculate seconds until the next polling window starts"""
    now = datetime.now()
    current_hour = now.hour
    
    if current_hour < POLL_START_HOUR:
        # Before polling hours - wait until start
        target = now.replace(hour=POLL_START_HOUR, minute=0, second=0, microsecond=0)
        return int((target - now).total_seconds())
    else:
        # After polling hours - wait until next day's start
        next_day = now.replace(hour=POLL_START_HOUR, minute=0, second=0, microsecond=0)
        next_day = next_day.replace(day=now.day + 1)
        return int((next_day - now).total_seconds())


class MetadataManager:
    """Manages the single JSON metadata file with atomic writes"""
    
    def __init__(self, metadata_path: Path):
        self.metadata_path = metadata_path
        self._ensure_metadata_exists()
    
    def _ensure_metadata_exists(self):
        """Initialize metadata file if it doesn't exist"""
        if not self.metadata_path.exists():
            self.metadata_path.parent.mkdir(parents=True, exist_ok=True)
            initial_data = {
                "images": [],
                "last_poll_time": None,
                "total_images_collected": 0
            }
            self._atomic_write(initial_data)
            logger.info(f"Created metadata file: {self.metadata_path}")
    
    def _atomic_write(self, data: dict):
        """Write metadata atomically using temp file + rename"""
        temp_path = self.metadata_path.with_suffix('.tmp')
        with open(temp_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        temp_path.replace(self.metadata_path)
    
    def load(self) -> dict:
        """Load metadata from file"""
        with open(self.metadata_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def add_image(self, image_entry: dict):
        """Add a new image entry to metadata"""
        data = self.load()
        data["images"].append(image_entry)
        data["total_images_collected"] = len(data["images"])
        data["last_poll_time"] = datetime.now(timezone.utc).isoformat()
        self._atomic_write(data)
        logger.info(f"Added image to metadata: {image_entry['filename']}")
    
    def update_last_poll(self):
        """Update last poll time without adding an image"""
        data = self.load()
        data["last_poll_time"] = datetime.now(timezone.utc).isoformat()
        self._atomic_write(data)


class CoffeeBotPoller:
    """Polls @StufeBot for coffee pot images"""
    
    def __init__(self):
        # Check authentication method
        self.use_bot_token = bool(BOT_TOKEN)
        
        if self.use_bot_token:
            logger.info("Using bot token authentication")
            # Bot token authentication - simpler, no interactive auth needed
            if not API_ID or not API_HASH:
                raise ValueError("api_id and api_hash must be set in .env file")
            
            SESSION_DIR.mkdir(parents=True, exist_ok=True)
            self.session_file = SESSION_DIR / 'bot_token_session.session'
            self.client = TelegramClient(
                str(SESSION_DIR / 'bot_token_session'),
                API_ID,
                API_HASH
            )
        else:
            logger.info("Using user authentication")
            # User authentication - requires interactive phone auth
            if not API_ID or not API_HASH:
                raise ValueError("api_id and api_hash must be set in .env file")
            
            SESSION_DIR.mkdir(parents=True, exist_ok=True)
            self.session_file = SESSION_DIR / 'bot_poller_session.session'
            self.client = TelegramClient(
                str(SESSION_DIR / 'bot_poller_session'),
                API_ID,
                API_HASH
            )
        
        self.metadata_manager = MetadataManager(METADATA_FILE)
        IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    
    def check_session_exists(self) -> bool:
        """Check if Telegram session file exists"""
        return self.session_file.exists()
    
    async def request_status(self) -> Optional[dict]:
        """
        Send /status command to bot and download the response image.
        Returns image metadata dict or None if failed.
        """
        try:
            logger.info(f"Sending /status command to {BOT_USERNAME}")
            
            # Use conversation API for clean request-response
            async with self.client.conversation(BOT_USERNAME, timeout=RESPONSE_TIMEOUT) as conv:
                # Send /status command
                await conv.send_message('/status')
                
                # Wait for response with image
                response = await conv.get_response()
                
                if not response.media:
                    logger.warning(f"Bot response has no media. Text: {response.text}")
                    return None
                
                # Generate filename with timestamp
                timestamp = datetime.now(timezone.utc)
                filename = f"coffee_{timestamp.strftime('%Y-%m-%d_%H-%M-%S')}.jpg"
                file_path = IMAGES_DIR / filename
                
                # Download image
                downloaded_path = await response.download_media(file=str(file_path))
                logger.info(f"Downloaded image: {downloaded_path}")
                
                # Create metadata entry
                metadata_entry = {
                    "filename": filename,
                    "message_id": response.id,
                    "telegram_date": response.date.isoformat() if response.date else None,
                    "download_date": timestamp.isoformat(),
                    "bot_command": "/status",
                    "caption": response.text or "",
                    "labeled": False,
                    "mask_filename": None
                }
                
                return metadata_entry
                
        except TelethonTimeoutError:
            logger.error(f"Timeout waiting for response from {BOT_USERNAME}")
            return None
        except Exception as e:
            logger.error(f"Error requesting status: {e}", exc_info=True)
            return None
    
    async def poll_once(self):
        """Execute one polling cycle"""
        try:
            logger.info("Starting poll cycle")
            
            # Request image from bot
            metadata_entry = await self.request_status()
            
            if metadata_entry:
                # Save metadata
                self.metadata_manager.add_image(metadata_entry)
                logger.info(f"Successfully collected image: {metadata_entry['filename']}")
            else:
                # Update last poll time even if failed
                self.metadata_manager.update_last_poll()
                logger.warning("Poll cycle completed but no image was collected")
                
        except Exception as e:
            logger.error(f"Error in poll cycle: {e}", exc_info=True)
            self.metadata_manager.update_last_poll()
    
    async def run(self):
        """Main polling loop"""
        logger.info(f"Starting Coffee Bot Poller")
        logger.info(f"Bot: {BOT_USERNAME}")
        logger.info(f"Poll interval: {POLL_INTERVAL_SECONDS} seconds")
        logger.info(f"Images directory: {IMAGES_DIR}")
        logger.info(f"Metadata file: {METADATA_FILE}")
        
        # Check if session exists (only for user auth, not needed for bot token)
        if not self.use_bot_token and not self.check_session_exists():
            logger.error("="*60)
            logger.error("AUTHENTICATION REQUIRED")
            logger.error("No Telegram session found. You must authenticate first.")
            logger.error("")
            logger.error("Run this command to authenticate interactively:")
            logger.error("  docker compose run --rm coffee-poller")
            logger.error("")
            logger.error("Then start the service normally:")
            logger.error("  docker compose up -d")
            logger.error("")
            logger.error("Or use bot token authentication (set BOT_TOKEN in .env)")
            logger.error("="*60)
            raise RuntimeError("Authentication required - no session file found")
        
        # Start client with appropriate authentication
        if self.use_bot_token:
            await self.client.start(bot_token=BOT_TOKEN)
            logger.info("Telegram client connected using bot token")
        else:
            await self.client.start()
            logger.info("Telegram client connected using user authentication")
        
        # Load existing metadata
        metadata = self.metadata_manager.load()
        logger.info(f"Loaded metadata: {metadata['total_images_collected']} images collected so far")
        logger.info(f"Polling hours: {POLL_START_HOUR:02d}:00 - {POLL_END_HOUR:02d}:00 (local time)")
        
        poll_count = 0
        while not shutdown_flag:
            # Check if we're within polling hours
            if not is_within_polling_hours():
                wait_seconds = seconds_until_next_poll_window()
                now = datetime.now()
                logger.info(f"Outside polling hours (current time: {now.strftime('%H:%M:%S')})")
                logger.info(f"Waiting {wait_seconds} seconds ({wait_seconds/3600:.1f} hours) until next poll window")
                
                # Sleep in smaller chunks to allow graceful shutdown
                while wait_seconds > 0 and not shutdown_flag:
                    sleep_time = min(60, wait_seconds)  # Check every minute
                    await asyncio.sleep(sleep_time)
                    wait_seconds -= sleep_time
                
                if shutdown_flag:
                    break
                
                logger.info("Entering polling hours window")
            
            poll_count += 1
            logger.info(f"=== Poll #{poll_count} ===")
            
            await self.poll_once()
            
            if not shutdown_flag:
                logger.info(f"Sleeping for {POLL_INTERVAL_SECONDS} seconds until next poll")
                await asyncio.sleep(POLL_INTERVAL_SECONDS)
        
        logger.info("Shutdown flag detected, stopping poller")
        await self.client.disconnect()


def signal_handler(signum, frame):
    """Handle shutdown signals"""
    global shutdown_flag
    logger.info(f"Received signal {signum}, initiating graceful shutdown")
    shutdown_flag = True


async def main():
    """Entry point"""
    # Register signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        poller = CoffeeBotPoller()
        await poller.run()
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)
    
    logger.info("Coffee Bot Poller stopped")


if __name__ == "__main__":
    asyncio.run(main())
