/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
*/
/* tslint:disable */

// Fix: Update parseJSON to handle markdown fences and parse the extracted string.
export const parseJSON = (str: string): any => {
  let jsonStr = str.trim();
  // Regex from guideline to extract content from ```json ... ``` or ``` ... ```
  const fenceRegex = /^```(\w*)?\s*\n?(.*?)\n?\s*```$/s;
  const match = jsonStr.match(fenceRegex);
  if (match && match[2]) {
    jsonStr = match[2].trim(); // Extracted content is in match[2]
  }
  // Attempt to parse the string (either extracted or original if no fences)
  try {
    return JSON.parse(jsonStr);
  } catch (e) {
    try {
      const firstBrace = jsonStr.indexOf('{');
      const lastBrace = jsonStr.lastIndexOf('}');
      if (firstBrace !== -1 && lastBrace !== -1 && lastBrace > firstBrace) {
        const potentialJson = jsonStr.slice(firstBrace, lastBrace + 1);
        return JSON.parse(potentialJson);
      }
    } catch {
      // ignore and fall through
    }
    console.error(
      "Failed to parse JSON content:",
      e,
      "String after fence removal:",
      jsonStr,
      "Original string:",
      str,
    );
    throw new Error(
      `Failed to parse JSON response. ${e instanceof Error ? e.message : String(e)}. Preview: ${jsonStr.substring(0, 200)}...`,
    );
  }
};

// Fix: Update parseHTML to handle markdown fences. Removed opener/closer params as they are standardized by prompt.
export const parseHTML = (str: string): string => {
  let htmlStr = str.trim();
  // Regex to extract content from ```html ... ``` or ``` ... ```
  const fenceRegex = /^```(\w*)?\s*\n?(.*?)\n?\s*```$/s; // Using 's' for dotall
  const match = htmlStr.match(fenceRegex);
  if (match && match[2]) {
    htmlStr = match[2].trim(); // Extracted content is in match[2]
    return htmlStr;
  }
  // Fallback: If no fences were found.
  // The prompt for code generation _requires_ fences.
  // If they are not there, it's a deviation. Return the original trimmed string as a best effort.
  // console.warn("HTML content was not wrapped in ```...``` as expected. Using raw content as fallback.");
  return htmlStr;
};
