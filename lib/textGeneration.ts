/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
*/
/* tslint:disable */

import {
  FinishReason,
  GenerateContentConfig,
  GenerateContentParameters, // Corrected import
  GenerateContentResponse, // Corrected import
  GoogleGenAI,
  HarmBlockThreshold,
  HarmCategory,
  Part,
  SafetySetting,
  GroundingMetadata, 
} from '@google/genai';

// Using fetch directly for OpenRouter calls to avoid library header issues
import { Anthropic } from '@anthropic-ai/sdk';


const GEMINI_API_KEY = process.env.API_KEY;

export interface TextGenerationInteraction {
  type: 'PROMPT' | 'RESPONSE' | 'ERROR' | 'TOKEN';
  data: any; // Raw request for PROMPT, GenerateContentResponse for RESPONSE, Error for ERROR, or string token for TOKEN
  modelName?: string;
}

export interface GenerateTextOptions {
  modelName: string;
  provider: string;
  apiKey: string;
  basePrompt: string;
  videoUrl?: string;
  additionalUserText?: string;
  temperature?: number;
  safetySettings?: SafetySetting[];
  responseMimeType?: string;
  useGoogleSearch?: boolean;
  stream?: boolean;
  onInteraction?: (interaction: TextGenerationInteraction) => void;
  onToken?: (token: string) => void;
}

export interface TextGenerationResponse { // This remains for the function's return type
  text: string;
  groundingMetadata?: GroundingMetadata;
}

/**
 * Generate text content using the Gemini API.
 *
 * @param options - Configuration options for the generation request.
 * @returns The response text and optional grounding metadata from the Gemini API.
 */
export async function generateText(
  options: GenerateTextOptions,
): Promise<TextGenerationResponse> {
  const {
    modelName,
    provider,
    apiKey,
    basePrompt,
    videoUrl,
    additionalUserText,
    temperature = 0.75,
    safetySettings,
    responseMimeType,
    useGoogleSearch = false,
    stream = false,
    onInteraction,
    onToken,
  } = options;

  if (!apiKey) {
    const error = new Error('API key is missing or empty');
    if (onInteraction) {
      onInteraction({type: 'ERROR', data: error, modelName});
    }
    throw error;
  }

  const fullPrompt = basePrompt + (additionalUserText ? `\n${additionalUserText}` : '');

  if (provider === 'openrouter') {
    if (!apiKey) {
      throw new Error('OpenRouter API key is required');
    }

    const messages = [{ role: 'user', content: fullPrompt }];

    if (videoUrl) {
      throw new Error('Video input is not supported for OpenRouter provider.');
    }

    if (useGoogleSearch) {
      messages[0].content = `Please research using available tools or knowledge: ${fullPrompt}`;
    }

    if (onInteraction) {
      onInteraction({type: 'PROMPT', data: {model: modelName, messages}, modelName});
    }

    const requestBody = {
      model: modelName,
      messages,
      temperature,
      stream: stream || Boolean(onToken),
    };

    let sanitizedKey = apiKey.trim();
    if (!sanitizedKey.toLowerCase().startsWith('bearer ')) {
      sanitizedKey = `Bearer ${sanitizedKey}`;
    }

    const headers: HeadersInit = {
      'Content-Type': 'application/json',
      'Authorization': sanitizedKey,
      'HTTP-Referer': window.location.origin,
      'X-Title': 'Video to Learning App',
    };

    const url = 'https://openrouter.ai/api/v1/chat/completions';

    try {
      let collectedText = '';
      if (requestBody.stream) {
        const resp = await fetch(url, {
          method: 'POST',
          headers,
          body: JSON.stringify(requestBody),
        });
        if (!resp.ok || !resp.body) {
          throw new Error(`${resp.status} ${resp.statusText}`);
        }
        const reader = resp.body.getReader();
        const decoder = new TextDecoder();
        let done = false;
        while (!done) {
          const { value, done: doneReading } = await reader.read();
          if (doneReading) break;
          const chunk = decoder.decode(value, { stream: true });
          const lines = chunk.split('\n');
          for (const line of lines) {
            const trimmed = line.trim();
            if (!trimmed.startsWith('data:')) continue;
            const data = trimmed.replace(/^data:\s*/, '');
            if (data === '[DONE]') { done = true; break; }
            try {
              const parsed = JSON.parse(data);
              const token = parsed.choices?.[0]?.delta?.content || '';
              collectedText += token;
              if (onToken) onToken(token);
              if (onInteraction) onInteraction({type: 'TOKEN', data: token, modelName});
            } catch (e) {
              console.error('Failed to parse stream chunk', e, data);
            }
          }
        }
      } else {
        const resp = await fetch(url, {
          method: 'POST',
          headers,
          body: JSON.stringify(requestBody),
        });
        if (!resp.ok) {
          throw new Error(`${resp.status} ${resp.statusText}`);
        }
        const json = await resp.json();
        collectedText = json.choices?.[0]?.message?.content || '';
      }

      if (onInteraction) onInteraction({type: 'RESPONSE', data: {text: collectedText}, modelName});
      return { text: collectedText };
    } catch (error) {
      if (onInteraction) onInteraction({type: 'ERROR', data: error, modelName});
      throw error;
    }
  } else if (provider === 'native') {
    if (modelName.startsWith('google/')) {
      const ai = new GoogleGenAI({apiKey});

      const parts: Part[] = [{text: fullPrompt}];

      if (videoUrl) {
        parts.push({ fileData: { mimeType: 'video/mp4', fileUri: videoUrl } });
      }

      const baseConfig: GenerateContentConfig = { temperature };

      if (responseMimeType) baseConfig.responseMimeType = responseMimeType;
      if (safetySettings) baseConfig.safetySettings = safetySettings;
      if (useGoogleSearch) baseConfig.tools = [{googleSearchRetrieval: {}}];

      const request = { model: modelName.split('/')[1], contents: [{role: 'user', parts}], config: baseConfig };

      if (onInteraction) onInteraction({type: 'PROMPT', data: request, modelName});

      try {
        let genAiResponse: GenerateContentResponse | undefined;
        let collectedText = '';
        if (stream || onToken) {
          const streamResp = await ai.models.generateContentStream(request);
          for await (const chunk of streamResp) {
            const token = chunk.text || '';
            collectedText += token;
            genAiResponse = chunk;
            if (onToken) onToken(token);
            if (onInteraction) onInteraction({type: 'TOKEN', data: token, modelName});
          }
          if (!genAiResponse) {
            throw new Error('No response received from streaming');
          }
          (genAiResponse as any).text = collectedText;
        } else {
          genAiResponse = await ai.models.generateContent(request);
          collectedText = genAiResponse.text;
        }

        if (onInteraction) onInteraction({type: 'RESPONSE', data: genAiResponse, modelName});

        if (genAiResponse.promptFeedback?.blockReason) {
          throw new Error(
            `Content generation failed: Prompt blocked (reason: ${genAiResponse.promptFeedback.blockReason})`,
          );
        }

        if (!genAiResponse.candidates || genAiResponse.candidates.length === 0) {
          if (genAiResponse.promptFeedback?.blockReason) {
            throw new Error(
              `Content generation failed: No candidates returned. Prompt feedback: ${genAiResponse.promptFeedback.blockReason}`,
            );
          }
          throw new Error('Content generation failed: No candidates returned.');
        }

        const firstCandidate = genAiResponse.candidates[0];

        if (
          firstCandidate.finishReason &&
          firstCandidate.finishReason !== FinishReason.STOP
        ) {
          if (firstCandidate.finishReason === FinishReason.SAFETY) {
            console.error('Safety ratings:', firstCandidate.safetyRatings);
            throw new Error(
              'Content generation failed: Response blocked due to safety settings.',
            );
          } else {
            throw new Error(
              `Content generation failed: Stopped due to ${firstCandidate.finishReason}.`,
            );
          }
        }

        return { text: collectedText, groundingMetadata: firstCandidate.groundingMetadata };
      } catch (error) {
        console.error(
          'An error occurred during Gemini API call or response processing:',
          error,
        );
        if (onInteraction) onInteraction({type: 'ERROR', data: error, modelName});
         if (error instanceof Error && error.message.includes("application/json") && error.message.includes("tool")) {
            throw new Error(`API Error: ${error.message}. Note: JSON response type is not supported with Google Search tool.`);
        }
        throw error;
      }
    } else if (modelName.startsWith('anthropic/')) {
      const anthropic = new Anthropic({ apiKey });

      const messages = [{ role: 'user', content: fullPrompt }];

      if (videoUrl) {
        throw new Error('Video input is not supported for Anthropic provider.');
      }

      if (useGoogleSearch) {
        messages[0].content = `Please research or use knowledge for: ${fullPrompt}`;
      }

      if (onInteraction) onInteraction({type: 'PROMPT', data: {model: modelName.split('/')[1], messages}, modelName});

      try {
        let collectedText = '';
        if (stream || onToken) {
          const streamResp = await anthropic.messages.create({
            model: modelName.split('/')[1],
            max_tokens: 4096,
            temperature,
            messages,
            stream: true,
          });

          for await (const chunk of streamResp) {
            if (chunk.type === 'content_block_delta' && chunk.delta.type === 'text_delta') {
              const token = chunk.delta.text;
              collectedText += token;
              if (onToken) onToken(token);
              if (onInteraction) onInteraction({type: 'TOKEN', data: token, modelName});
            }
          }
        } else {
          const response = await anthropic.messages.create({
            model: modelName.split('/')[1],
            max_tokens: 4096,
            temperature,
            messages,
          });
          if (response.content[0].type === 'text') {
            collectedText = response.content[0].text;
          }
        }

        if (onInteraction) onInteraction({type: 'RESPONSE', data: {text: collectedText}, modelName});

        return { text: collectedText };
      } catch (error) {
        if (onInteraction) onInteraction({type: 'ERROR', data: error, modelName});
        throw error;
      }
    } else {
      throw new Error(`Native provider not supported for model: ${modelName}`);
    }
  } else {
    throw new Error(`Unknown provider: ${provider}`);
  }
}