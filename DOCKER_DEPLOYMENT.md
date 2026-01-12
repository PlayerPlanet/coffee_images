# Coffee Bot Poller - Docker Automation

Automated time-series collection of coffee pot images from @StufeBot using Docker.

## Overview

This Docker service polls @StufeBot every 5-10 minutes with the `/status` command to collect fresh coffee pot images from the FK guildroom RasPi camera. Each image is stored with comprehensive metadata in a single JSON database.

## Prerequisites

1. **Telegram API Credentials**
   - Visit https://my.telegram.org/apps
   - Create a new application
   - Note your `api_id` and `api_hash`

2. **Docker & Docker Compose**
   - Docker Engine 20.10+
   - Docker Compose v2+

## Setup

1. **Clone repository and navigate to project:**
   ```bash
   cd coffee_images
   ```

2. **Create `.env` file:**
   ```bash
   cp .env.example .env
   ```

3. **Edit `.env` file with your credentials:**
   ```env
   api_id=YOUR_TELEGRAM_API_ID
   api_hash=YOUR_TELEGRAM_API_HASH
   BOT_USERNAME=@StufeBot
   POLL_INTERVAL_SECONDS=300
   ```

4. **First-time authentication:**
   
   On first run, Telegram will require phone authentication. Run interactively:
   ```bash
   docker compose run --rm coffee-poller
   ```
   
   Enter your phone number and verification code when prompted. The session will be saved to `.telethon/` directory.

5. **Start the service:**
   ```bash
   docker compose up -d
   ```

## Usage

### Start the poller
```bash
docker compose up -d
```

### View logs
```bash
docker compose logs -f coffee-poller
```

### Stop the poller
```bash
docker compose down
```

### Restart the poller
```bash
docker compose restart coffee-poller
```

### Check collected data
```bash
# View metadata
cat data/image_metadata.json

# List collected images
ls data/raw_img/
```

## Data Structure

### Directory Layout
```
coffee_images/
├── data/
│   ├── raw_img/                    # Collected coffee pot images
│   │   ├── coffee_2026-01-12_14-30-45.jpg
│   │   └── coffee_2026-01-12_14-35-50.jpg
│   └── image_metadata.json         # Single JSON metadata database
├── .telethon/                      # Telegram session (gitignored)
│   └── bot_poller_session.session
└── .env                            # Your credentials (gitignored)
```

### Metadata Schema

`data/image_metadata.json`:
```json
{
  "images": [
    {
      "filename": "coffee_2026-01-12_14-30-45.jpg",
      "message_id": 123456,
      "telegram_date": "2026-01-12T14:30:45+00:00",
      "download_date": "2026-01-12T14:30:48+00:00",
      "bot_command": "/status",
      "caption": "Kahvia pannu puolillaan",
      "labeled": false,
      "mask_filename": null
    }
  ],
  "last_poll_time": "2026-01-12T14:30:48+00:00",
  "total_images_collected": 1
}
```

### Metadata Fields

- **filename**: Timestamped filename of the saved image
- **message_id**: Telegram message ID (unique identifier)
- **telegram_date**: When the bot sent the image
- **download_date**: When we downloaded the image
- **bot_command**: Command used (`/status`)
- **caption**: Text from the bot (e.g., coffee level description)
- **labeled**: Whether a segmentation mask has been created (default: false)
- **mask_filename**: Name of the corresponding mask file (default: null)

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `api_id` | *required* | Telegram API ID from my.telegram.org |
| `api_hash` | *required* | Telegram API hash from my.telegram.org |
| `BOT_USERNAME` | `@StufeBot` | Telegram bot username |
| `POLL_INTERVAL_SECONDS` | `300` | Time between polls (300 = 5 min) |
| `RESPONSE_TIMEOUT` | `30` | Timeout for bot response (seconds) |

### Adjusting Poll Interval

Edit `.env` file:
```env
# Poll every 5 minutes (default)
POLL_INTERVAL_SECONDS=300

# Poll every 10 minutes
POLL_INTERVAL_SECONDS=600

# Poll every 2 minutes (more frequent)
POLL_INTERVAL_SECONDS=120
```

Then restart:
```bash
docker compose restart coffee-poller
```

## Integration with Training Pipeline

The metadata is kept **separate** from the existing `image_label_status.json` to avoid conflicts with the manual labeling workflow.

When preparing training data:
1. Filter `data/image_metadata.json` for images where `labeled: true`
2. Use `mask_filename` to locate corresponding masks
3. Copy to training directories as needed

## Troubleshooting

### Authentication Issues
```bash
# Remove old session and re-authenticate
rm -rf .telethon/*
docker compose run --rm coffee-poller
```

### Check if bot is responding
```bash
# View recent logs
docker compose logs --tail=50 coffee-poller

# Look for "Downloaded image" success messages
```

### Bot not sending images
- Verify bot username is correct: `@StufeBot`
- Check if bot is online in Telegram
- Try sending `/status` manually to the bot

### Permission Issues
```bash
# Ensure data directories exist and are writable
mkdir -p data/raw_img .telethon
chmod 755 data .telethon
```

## Development

### Run without Docker
```bash
# Install dependencies
poetry install

# Set environment variables
export $(cat .env | xargs)

# Run poller
poetry run python -m coffee_images.data_collection.bot_poller
```

### Build Docker image
```bash
docker compose build
```

## Maintenance

### Backup collected data
```bash
tar -czf coffee_backup_$(date +%Y%m%d).tar.gz data/
```

### Clean up old images (optional)
```bash
# Remove images older than 90 days
find data/raw_img/ -name "coffee_*.jpg" -mtime +90 -delete
```

### View statistics
```bash
# Count collected images
jq '.total_images_collected' data/image_metadata.json

# View last poll time
jq '.last_poll_time' data/image_metadata.json

# Count labeled vs unlabeled
jq '[.images[] | select(.labeled == true)] | length' data/image_metadata.json
```

## Notes

- Each `/status` command triggers a **new photo** from the RasPi camera
- Images are stored with UTC timestamps in filenames
- The Telegram session is persisted to avoid re-authentication
- Graceful shutdown on Docker stop (SIGTERM handling)
- Automatic retry on network failures
- JSON metadata is written atomically to prevent corruption
