# Video to Learning App

This is an experimental web application that generates interactive learning apps from YouTube videos or topics using AI models. It supports multiple AI providers and models for text generation, with features like educational material generation (lesson plans, handouts, quizzes) and app refinement.

**All rights reserved.** © 2025 Robin L. M. Cheung, MBA. This project is proprietary and not open for public contributions or usage without permission.

## Features
- Generate interactive apps from YouTube URLs or text topics.
- Support for multiple AI models via OpenRouter or native providers (Google Gemini, Anthropic Claude).
- Persist model selections and API keys in localStorage.
- Embed and deploy generated apps.
- Optional educational materials: lesson plans, handouts, and quizzes.

## Run Locally

**Prerequisites:** Node.js (v16+), npm, and API keys for selected providers (e.g., Gemini API key, OpenRouter key).

1. Clone the repository:
   ```
   git clone https://github.com/rebots-online/classify.git
   cd classify
   ```

2. Install dependencies:
   ```
   npm install
   ```

3. Configure environment:
   - Copy `.env.local.example` to `.env.local` and set your API keys (e.g., `GEMINI_API_KEY` for native Google provider).
   - For OpenRouter, the app uses a default key, but you can override it in the UI.

4. Run the development server:
   ```
   npm run dev
   ```
   - Access the app at `http://localhost:56872` (port configurable in `vite.config.ts`).

5. Usage:
   - Enter a YouTube URL, topic, or app idea in the input field.
   - Select optional materials (lesson plan, handout, quiz).
   - Choose AI model and provider from the dropdowns.
   - Click "Generate App" to create the interactive learning app.
   - Use tabs to view Spec, Code, Render, and Materials.

## Deployment
- Build the app: `npm run build`.
- Deploy the `dist` folder to a static host (e.g., Netlify, Vercel).
- For generated apps, use the "Deploy to Web" button in the UI.

## Development Notes
- The app uses Vite for bundling and React for the frontend.
- Text generation is handled via `lib/textGeneration.ts` with support for streaming responses.
- Customize prompts in `lib/prompts.ts`.

For issues or feedback, contact the repository owner.


