<p align="left">
  <a href="https://flare.network/" target="blank"><img src="https://content.flare.network/Flare-2.svg" width="410" height="106" alt="Flare Logo" /></a>
</p>

# fsp-observer

A python tool to observe
[`flare-systems-deployment`](https://github.com/flare-foundation/flare-systems-deployment)
and
[`fdc-suite-deployment`](https://github.com/flare-foundation/fdc-suite-deployment).
It does so without keeping any state. This allows it to only require the
identity address of the observed entity and an observer node rpc. It can send
error messages to:

- discord via webhook
- slack via webhook
- telegram bot via sendMessage method
- generic rest api via http post method

## Setting up and using fsp-observer

The easiest way to run is via docker. The only required env variables are
`RPC_BASE_URL` and `IDENTITY_ADDRESS`. Others can be set as desired.

```bash
docker run \
    -e RPC_BASE_URL="https://flare-api.flare.network" \
    -e IDENTITY_ADDRESS="0x0000000000000000000000000000000000000000" \
    -e NOTIFICATION_DISCORD_WEBHOOK="https://discord.com/api/webhooks/secret/secret" \
    -e NOTIFICATION_TELEGRAM_BOT_TOKEN="secret" \
    -e NOTIFICATION_TELEGRAM_CHAT_ID="secret" \
    -e NOTIFICATION_SLACK_WEBHOOK="https://hooks.slack.com/services/secret/secret/secret" \
    -e NOTIFICATION_GENERIC_WEBHOOK="http://host:port/path" \
    -e METRICS_ENABLED="true" \
    -e METRICS_PORT="8000" \
    -e LOG_LEVEL="INFO" \
    -e MAX_BLOCK_RANGE="1000" \
    ghcr.io/flare-foundation/fsp-observer:main
```

Alternatively [uv](https://docs.astral.sh/uv/) can be used to run:

```bash
uv sync
# optionally create .env file
RPC_BASE_URL="https://flare-api.flare.network" \
  IDENTITY_ADDRESS="0x0000000000000000000000000000000000000000" \
  uv run python main.py
```

## Environment variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `RPC_BASE_URL` | yes | - | RPC base URL without `/ext/bc/C/rpc` suffix |
| `IDENTITY_ADDRESS` | yes | - | Identity address of the observed entity |
| `FEE_THRESHOLD` | no | `25` | Balance threshold in FLR to trigger low balance warning |
| `NOTIFICATION_DISCORD_WEBHOOK` | no | - | Discord webhook URL (comma-separated for multiple) |
| `NOTIFICATION_DISCORD_EMBED_WEBHOOK` | no | - | Discord embed webhook URL |
| `NOTIFICATION_SLACK_WEBHOOK` | no | - | Slack webhook URL |
| `NOTIFICATION_SLACK_EMBED_WEBHOOK` | no | - | Slack embed webhook URL |
| `NOTIFICATION_TELEGRAM_BOT_TOKEN` | no | - | Telegram bot token (comma-separated for multiple) |
| `NOTIFICATION_TELEGRAM_CHAT_ID` | no | - | Telegram chat ID (comma-separated, paired with bot tokens) |
| `NOTIFICATION_GENERIC_WEBHOOK` | no | - | Generic HTTP POST webhook URL |
| `METRICS_ENABLED` | no | `false` | Enable Prometheus metrics endpoint |
| `METRICS_PORT` | no | `8000` | Prometheus metrics server port |
| `METRICS_ADDRESS` | no | `0.0.0.0` | Prometheus metrics server bind address |
| `LOG_LEVEL` | no | `INFO` | Logging level (`DEBUG`, `INFO`, `WARNING`, `ERROR`) |
| `MAX_BLOCK_RANGE` | no | `1000` | Max number of blocks per `get_logs` request (lower it if the RPC caps the block range) |
| `FALSE_POSITIVE_THRESHOLD` | no | `100` | Max false positive probability (in %) to report a missed fast update as critical; `100` reports on any false positive |

## Prometheus metrics

When `METRICS_ENABLED=true`, metrics are exposed at `http://host:METRICS_PORT/metrics`.

| Metric | Type | Description |
|---|---|---|
| `flare_fsp_submit_ok_total` | Counter | Successful submissions per protocol/phase |
| `flare_fsp_submit_late_total` | Counter | Late submissions per protocol/phase |
| `flare_fsp_submit_early_total` | Counter | Early submissions per protocol/phase |
| `flare_fsp_submit_missing_total` | Counter | Missing submissions per protocol/phase |
| `flare_fsp_address_balance_wei` | Gauge | Address balance in wei per role |
| `flare_fsp_registered_current_epoch` | Gauge | 1 if registered in current reward epoch |
| `flare_fsp_registered_next_epoch` | Gauge | 1 if registered for next reward epoch |
| `flare_fsp_voting_round` | Gauge | Current voting round ID |
| `flare_fsp_reward_epoch` | Gauge | Current reward epoch ID |
| `flare_fsp_node_uptime_ratio` | Gauge | Node uptime ratio per node ID |
| `flare_fsp_fast_update_blocks_since_last` | Gauge | Blocks since last fast update submission |
| `flare_fsp_ftso_anchor_feeds_success_rate_bips` | Gauge | FTSO anchor feeds success rate in bips |
| `flare_fsp_fdc_participation_rate_bips` | Gauge | FDC participation rate in bips |
| `flare_fsp_reveal_offence_total` | Counter | Reveal offences per protocol |
| `flare_fsp_signature_grace_period_missed_total` | Counter | Signature submissions past grace period |
| `flare_fsp_signature_mismatch_total` | Counter | Signature mismatches per protocol |
| `flare_fsp_contract_address_wrong_total` | Counter | Wrong contract address detections |
| `flare_fsp_unclaimed_rewards_wei` | Gauge | Unclaimed reward amount in wei |
