# Public hardware information

- Keep the owner's physical GPU model, UUID, serial number and machine identity out of public documentation, changelogs, commit messages, validation summaries and uploaded logs. Describe hardware by architecture, such as Ampere or Blackwell.
- GPU catalogs, supported profile IDs and synthetic compatibility tests may retain model names. Do not describe those entries as hardware owned by the maintainer or identify them as the maintainer's test devices.
- Documentation examples that select a physical test device should use a user-supplied `TARGET_GPU_PROFILE`, not a maintainer-specific default.
- Keep original hardware-identifying experiment records private. Publish explicitly redacted review copies and preserve the distinction between original digests and public-copy digests. Architecture-only summaries do not establish calibration compatibility across an entire architecture.
