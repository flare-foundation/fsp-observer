<p align="left">
  <a href="https://flare.network/" target="blank"><img src="https://content.flare.network/Flare-2.svg" width="410" height="106" alt="Flare Logo" /></a>
</p>

# fsp-observer

A python tool to observe
[`flare-systems-deployment`](https://github.com/flare-foundation/flare-systems-deployment)
and
[`fdc-suite-deployment`](https://github.com/flare-foundation/fdc-suite-deployment).
It does so without keeping any state. This allows it to only require the identity address of
the observed entity and an observer node rpc. It can send error messages to:
- discord via webhook
- slack via webhook
- telegram bot via sendMessage method
- generic rest api via http post method

## Setting up and using fsp-observer

The easiest way to run is via docker. The only required env variables are `RPC_BASE_URL` and `IDENTITY_ADDRESS`. Others can be set as desired.

```bash
docker run \
    -e RPC_BASE_URL="https://flare-api.flare.network" \
    -e IDENTITY_ADDRESS="0x0000000000000000000000000000000000000000" \
    -e NOTIFICATION_DISCORD_WEBHOOK="https://discord.com/api/webhooks/secret/secret" \
    -e NOTIFICATION_TELEGRAM_BOT_TOKEN="secret" \
    -e NOTIFICATION_TELEGRAM_CHAT_ID="secret" \
    -e NOTIFICATION_SLACK_WEBHOOK="https://hooks.slack.com/services/secret/secret/secret" \
    -e NOTIFICATION_GENERIC_WEBHOOK="http://host:port/path" \
    ghcr.io/flare-foundation/fsp-observer:main
```

Alternatively python can be used to run:
```bash
python -m venv venv
pip install -r requirements.txt
# optionally create .env file
RPC_BASE_URL="https://flare-api.flare.network" \
  IDENTITY_ADDRESS="0x0000000000000000000000000000000000000000" \
  python main.py
```
