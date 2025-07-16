# Checklist - OpenRouter Authorization Fix

- [x] Analyze existing use of OpenAI library in `lib/textGeneration.ts` for OpenRouter requests.
- [x] Replace OpenAI client usage with direct `fetch` implementation ensuring `Authorization`, `HTTP-Referer`, and `X-Title` headers are included.
- [x] Support streaming responses by processing `ReadableStream` chunks.
- [x] Update `docs/architecture/architecture.mmd` to include `OpenRouter` and `Anthropic` providers.
- [x] Run `npm install`.
- [x] Run `npm run build` to verify build succeeds.
