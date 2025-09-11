from telethon import TelegramClient
import os
from tqdm.asyncio import tqdm  # pip install tqdm
import asyncio
import dotenv
# --- Fill in your values ---
dotenv.load_dotenv()
api_id = dotenv.get_key("api_id")
api_hash = dotenv.get_key("api_hash")
client = TelegramClient('testsession', api_id, api_hash)

channel_username = "@fklors"   # e.g. "mychannel" or "https://t.me/..."
target_username = "@TsufeBot"         # the sender’s @username
download_dir = "/raw_img"

semaphore = asyncio.Semaphore(100)  # Limit to 10 concurrent downloads

async def download_with_semaphore(msg, download_dir, pbar):
    async with semaphore:
        await msg.download_media(file=download_dir)
        pbar.update(1)

async def main():
    if not os.path.exists(download_dir):
        os.makedirs(download_dir)

    target = await client.get_entity(target_username)

    # First pass → count how many messages match
    messages = []
    async for message in tqdm(client.iter_messages(channel_username, from_user=target), desc="Collecting messages"):
        messages.append(message)
    print(f"Found {len(messages)} images from {target_username}")

    # Second pass → download with progress bar
    with tqdm(total=len(messages), desc="Downloading") as pbar:
        tasks = [download_with_semaphore(msg, download_dir, pbar) for msg in messages]
        await asyncio.gather(*tasks)

with client:
    print("starting...")
    client.loop.run_until_complete(main())