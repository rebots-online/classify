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

const GEMINI_API_KEY = process.env.API_KEY;

export interface TextGenerationInteraction {
  type: 'PROMPT' | 'RESPONSE' | 'ERROR' | 'TOKEN';
  data: any; // Raw request for PROMPT, GenerateContentResponse for RESPONSE, Error for ERROR, or string token for TOKEN
  modelName?: string;
}

export interface GenerateTextOptions {
  modelName: string;
  basePrompt: string;
  videoUrl?: string;
  additionalUserText?: string;
  temperature?: number;
  safetySettings?: SafetySetting[];
  responseMimeType?: string;
  useGoogleSearch?: boolean;
  stream?: boolean;
  onInteraction?: (interaction: TextGenerationInteraction) => void; // Added callback
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
    basePrompt,
    videoUrl,
    additionalUserText,
    temperature = 0.75,
    safetySettings,
    responseMimeType,
    useGoogleSearch = false,
    stream = false,
    onInteraction, // Destructure callback
    onToken,
  } = options;

  if (!GEMINI_API_KEY) {
    const error = new Error('Gemini API key is missing or empty');
    if (onInteraction) {
      onInteraction({type: 'ERROR', data: error, modelName});
    }
    throw error;
  }

  const ai = new GoogleGenAI({apiKey: GEMINI_API_KEY});

  const parts: Part[] = [{text: basePrompt}];

  if (additionalUserText) {
    parts.push({text: additionalUserText});
  }

  if (videoUrl) {
    try {
      parts.push({
        fileData: {
          mimeType: 'video/mp4', 
          fileUri: videoUrl,
        },
      });
    } catch (error) {
      console.error('Error processing video input:', error);
      const err = new Error(`Failed to process video input from URL: ${videoUrl}`);
      if (onInteraction) {
        onInteraction({type: 'ERROR', data: err, modelName});
      }
      throw err;
    }
  }
  
  const baseConfig: GenerateContentConfig = {
    temperature,
  };

  if (!useGoogleSearch && responseMimeType) {
    baseConfig.responseMimeType = responseMimeType;
  }
  
  if (safetySettings) {
    baseConfig.safetySettings = safetySettings;
  }

  if (useGoogleSearch) {
    baseConfig.tools = [{googleSearch: {}}];
  }
  
  const request: GenerateContentParameters = { 
    model: modelName,
    contents: [{role: 'user', parts}],
    config: baseConfig,
  };


  if (onInteraction) {
    onInteraction({type: 'PROMPT', data: request, modelName});
  }

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
        if (onInteraction) {
          onInteraction({type: 'TOKEN', data: token, modelName});
        }
      }
      if (!genAiResponse) {
        throw new Error('No response received from streaming');
      }
      // Overwrite text with collected text for convenience
      (genAiResponse as any).text = collectedText;
    } else {
      genAiResponse = await ai.models.generateContent(request);
    }

    if (onInteraction) {
      onInteraction({type: 'RESPONSE', data: genAiResponse, modelName});
    }

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
    
    return {
        text: genAiResponse.text, 
        groundingMetadata: firstCandidate.groundingMetadata,
    };

  } catch (error) {
    console.error(
      'An error occurred during Gemini API call or response processing:',
      error,
    );
    if (onInteraction) {
      onInteraction({type: 'ERROR', data: error, modelName});
    }
     if (error instanceof Error && error.message.includes("application/json") && error.message.includes("tool")) {
        throw new Error(`API Error: ${error.message}. Note: JSON response type is not supported with Google Search tool.`);
    }
    throw error;
  }
}