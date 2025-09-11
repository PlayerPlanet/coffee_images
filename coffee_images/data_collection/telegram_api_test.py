from telethon import TelegramClient
import dotenv
import dotenv
# --- Fill in your values ---
dotenv.load_dotenv()
api_id = dotenv.get_key("api_id")
api_hash = dotenv.get_key("api_hash")
client = TelegramClient('testsession', api_id, api_hash)

async def main():
    print(await client.get_me())

with client:
    client.loop.run_until_complete(main())