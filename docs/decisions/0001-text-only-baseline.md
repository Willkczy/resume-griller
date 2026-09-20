# M0 decision: text-only baseline

Date: 2026-09-12. Status: implemented; live voice remains experimental.

The Python default enabled voice while `.env.example` disabled it and the locked
runtime omitted ElevenLabs. Choose the smallest supported contract: voice defaults
off in `Settings`, without adding a provider dependency or promising live support.
Existing explicit `VOICE_ENABLED=true` configuration is preserved, not overwritten.

Disabled voice status reports enabled/STT/TTS availability false, without creating
clients. Voice HTTP operations return 503; WS TTS returns no audio and WS audio
input rejects the request. Text UI is the default; recording appears only after
status reports available STT. Explicit video links require both STT/TTS capability,
otherwise render chat. Capability lookup failure also keeps text available. Existing
wire schemas are unchanged; the existing TypeScript `VoiceStatus` contract is reused.

The status endpoint checks client construction, not network/provider health. An
explicitly enabled installation may still fail at call time. ElevenLabs remains
absent from the supported root dependency set, and credentials are never required
for the text-only baseline. Proper optional voice packaging, live-call quality,
provider-error semantics, and the video's placeholder read-aloud behavior remain
unverified/unsupported work; this decision adds no streaming or new provider.

Tests exercise actual voice HTTP routes with provider factories forbidden, settings
with dotenv disabled, and the WS TTS guard. No live services or real audio/resumes
are used. The default regression failed before the change and passed afterward.
These checks establish voice isolation, not correctness of session startup (M1).
