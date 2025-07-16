# Checklist - OpenRouter Key Sanitization

- [x] Review OpenRouter authentication failure logs showing 401 errors.
- [x] Sanitize API key in `lib/textGeneration.ts` by trimming whitespace and avoiding duplicate `Bearer` prefix.
- [x] Ensure Authorization header uses sanitized key when calling OpenRouter.
- [x] Run `npm install`.
- [x] Run `npm run build` to confirm successful compilation.
