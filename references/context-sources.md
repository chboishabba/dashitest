# Context Sources

## Canonical chat archive (preferred)
- Path: `/home/c/Documents/code/ITIR-suite/chat-export-structurer/my_archive.sqlite`
- Schema: `messages(message_id, canonical_thread_id, platform, account_id, ts, role, text, title, source_id)`

## Live cache (fallback)
- Path: `/home/c/.chatgpt_history.sqlite3`
- Tables: `conversations`, `messages`

## Session token (for live sync scripts)
- Path: `/home/c/.chatgpt_session` (first line)
