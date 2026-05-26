---
title: Bridge Game Postmortem Chatbot
emoji: 🥸
colorFrom: indigo
colorTo: yellow
sdk: streamlit
sdk_version: 1.26.0
app_file: app.py
pinned: false
license: mit
---

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference

# Bridge Game Postmortem Chatbot

Project to provide high-level postmortem game information using a chat interface. User can either use vetted prompts, such as via the Summarize button, or custom prompts. Games are limited to ACBL club pair matchpoint games which used a Mitchell movement and not shuffled.

Try it live at: https://huggingface.co/spaces/bsalita/Bridge_Game_Postmortem_Chatbot

## Related Projects and Documents
For a list of related projects and documents see: https://github.com/BSalita/BridgeStats

## Shareable URLs

The sidebar options are mirrored into the URL query string so any view is
shareable / bookmarkable. Supported parameters:

| Param            | Maps to sidebar option         | Example value |
|------------------|--------------------------------|---------------|
| `player_id`      | ACBL player number             | `2663279`     |
| `session_id`     | Club game or tournament session| `6534522`     |
| `show_sql_query` | Developer Settings → Show SQL  | `1` / `0`     |
| `sd_samples`     | Single Dummy Samples Count     | `10`          |

Example: `https://acbl.postmortem.chat/?player_id=2663279&session_id=6534522`
loads that exact game directly.

## Automation: email a daily PDF report

`acbl_postmortem_generator.py` drives the live app in a headless browser,
captures the generated PDF, and emails it. Combined with the included
PowerShell helper this becomes a daily "did the player play today? if so,
mail me the report" job.

One-time setup:

```powershell
pip install -r requirements.txt
playwright install chromium
# put SMTP creds in a .env next to app.py:
#   SMTP_HOST=smtp.gmail.com
#   SMTP_PORT=587
#   SMTP_USER=you@example.com
#   SMTP_PASSWORD=<app password>
#   SMTP_FROM=you@example.com
```

Register a daily 6 PM scheduled task (Windows):

```powershell
.\schedule_daily_report.ps1 `
    -Url  "https://acbl.postmortem.chat/?player_id=2663279" `
    -Email coach@example.com `
    -Time 18:00
```

What the scheduled run does:

1. Purges any cached PDFs in `report_cache\` whose mtime is older than the
   TTL (default 168 h / 1 week). Note the TTL is a sliding window: every
   cache hit refreshes the file's mtime, so an entry survives indefinitely
   as long as the player hasn't played a newer game.
2. Opens the URL in headless Chromium and waits for the report metadata.
3. Always loads the player's most recent game (no date filter).
4. Skips silently if `report_cache\{player_id}_{game_date}.pdf` already
   exists -- the cache assumes the report was already emailed for that game.
5. Otherwise downloads the PDF, stores it in the cache, and emails it.

Net effect: each game produces exactly one email, no matter how often the
scheduler runs in between, and no matter how many days pass between games.

Other useful flags:

```powershell
# One-off run, save the PDF locally too, no email:
python acbl_postmortem_generator.py --url "..." --email me@x.com --output report.pdf --no-email

# Force re-send even though the cache says we already did:
python acbl_postmortem_generator.py --url "..." --email me@x.com --force-email

# Remove the scheduled task:
.\schedule_daily_report.ps1 -Unregister
```

"# Bridge_Game_Postmortem_Chatbot" 
