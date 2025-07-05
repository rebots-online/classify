# Checklist - Enhancements

- [x] Update `parseJSON` in `lib/parse.ts` to handle stray text around JSON.
- [x] Extend `TextGenerationInteraction` type in `lib/textGeneration.ts` to include `TOKEN` and add optional streaming.
- [x] Update `generateText` in `lib/textGeneration.ts` to support streaming via `generateContentStream` and emit `TOKEN` interactions.
- [x] Update `LlmLogPanel` in `components/LlmLogPanel.tsx` to display streaming tokens in real time.
- [x] Change default `showHowToUseModal` state in `App.tsx` to `false`.
- [x] Add "Get Started!" button to `HowToUseModal` to close the modal.
- [x] Add Save/Load project capability in `components/ContentContainer.tsx` using `localStorage`.
- [x] Add "Save Project" and "Load Project" buttons to action bar in `ContentContainer`.
- [x] Ensure loaded projects bypass regeneration and populate spec, code, and materials.
- [x] Update architecture diagram in `docs/architecture/architecture.mmd` after modifications.
✅ Run `npm install` and `npm run build` to verify build succeeds.
