# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

# \[[v0.2.0](<>) (Upcoming)\]

### Changed

- **BREAKING CHANGE:** `RPC_URL` environment variable was renamed to
  `RPC_BASE_URL` and now only expects base rpc url (without `/ext/bc/C/rpc` at
  the end)

### Added

- fast updates addresses are periodically checked for sufficient balance
- periodically check for unclaimed rewards
- check for registration and preregistration
- check if transactions are made against correct contracts
- check if transactions are sent too early or too late (before they were just
  reported as missing)
- check uptime for validator nodes
- validate fast updates participation
- sample minimal conditions for ftso, fast updates, fdc and validator nodes
- add a global try except and report when there are issues with the rpc

### Fixed

- fixed bug where the observer loop would crash at the beggining of next

# \[[v0.1.0](https://github.com/flare-foundation/fsp-observer/tree/v0.1.0)\] - 2025-06-16

### Added

- fdc round validation
- ftso round validation
- entity addresses are periodically checked for sufficient balance
- notifications plugins for discord, slack, telegram, generic http post
