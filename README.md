<p align="left">
  <a href="https://flare.network/" target="blank"><img src="https://content.flare.network/Flare-2.svg" width="410" height="106" alt="Flare Logo" /></a>
</p>

# flare-observer

deploy:

currently we support push notification over:
- discord webhook
- slack webhook
- telegram bot sendMessage method
- generic url post

All of these can be configured via env variables. All of these have a `NOTIFICATION` prefix and 
aren't a required value for the observer to start. All example values below

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

todos:

- more checks:
    - general/fsp:
        - [x] check if addresses (submit, signature, sign) have enough tokens for gas
            - [ ] collect fast updates addresses and check them too
        - [ ] check for unclaimed rewards
        - [ ] check for registration:
            - [ ] aggresively report failure to register after X minutes of registration window
            - [ ] include preregistration as well
        - [ ] check if transactions are being made against correct contracts (eg.: what if relay contract switches)
        - [ ] check if transactions are being made but were sent too early or too late
    - staking:
        - [ ] check node uptime
    - ftso:
        - [x] better ftso value analysis 
            - if you are meeting minimal conditions
            - weird value (not just None but also 0.1 all the time, or just wildly different to median)
            - parse events to be able to tell feeds by names not by indices
        - [x] check submit signatures signature against finalization 
    - fdc:
        - [x] sample minimal conditions
        - [x] correct bitvote length (submit2 fdc)
        - [x] check submit signatures signature against finalization 
    - fast updates:
        - [ ] recover signature from fast updates and check if updates are being made 
        - [ ] check if length of update is correct
        - [ ] sample minimal conditions
- push notification scheme:
    - we need a general-ish extnesible framework to add more notification plugins
    - notification plugins
        - [x] stdout logging
        - [x] discord
        - [x] slack
        - [x] telegram
        - [ ] pager duty
        - [x] generic post
