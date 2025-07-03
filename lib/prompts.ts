/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
*/
/* tslint:disable */

export const SPEC_FROM_VIDEO_PROMPT = `You are a pedagogist and product designer with deep expertise in crafting engaging learning experiences via interactive web apps.

Examine the contents of the attached video. Then, write a detailed and carefully considered spec for an interactive web app designed to complement the video and reinforce its key idea or ideas. The recipient of the spec does not have access to the video, so the spec must be thorough and self-contained (the spec must not mention that it is based on a video). Here is an example of a spec written in response to a video about functional harmony:

"In music, chords create expectations of movement toward certain other chords and resolution towards a tonal center. This is called functional harmony.

Build me an interactive web app to help a learner understand the concept of functional harmony.

SPECIFICATIONS:
1. The app must feature an interactive keyboard.
2. The app must showcase all 7 diatonic triads that can be created in a major key (i.e., tonic, supertonic, mediant, subdominant, dominant, submediant, leading chord).
3. The app must somehow describe the function of each of the diatonic triads, and state which other chords each triad tends to lead to.
4. The app must provide a way for users to play different chords in sequence and see the results.
[etc.]"

The goal of the app that is to be built based on the spec is to enhance understanding through simple and playful design. The provided spec should not be overly complex, i.e., a junior web developer should be able to implement it in a single html file (with all styles and scripts inline). Most importantly, the spec must clearly outline the core mechanics of the app, and those mechanics must be highly effective in reinforcing the given video's key idea(s).

If the user provides additional details, consider them carefully when crafting the spec.

Provide the result as a JSON object containing a single field called "spec". The value of the "spec" field must be a string. Ensure this string value is valid JSON string content: any double quotes, backslashes, or control characters (like newlines) within the specification text itself must be properly escaped (e.g., \\", \\\\, \\n).
For example: {"spec": "Build me an app with a spec string that is carefully escaped.\\nIt should include details like this: [\\"example\\", \\"list\\"]..."}`;


export const SPEC_FROM_TOPIC_PROMPT_TEMPLATE = `You are a pedagogist and product designer with deep expertise in crafting engaging learning experiences via interactive web apps.

Your task is to research the following topic using available tools (like Google Search) and then write a detailed and carefully considered spec for an interactive web app designed to teach or explore this topic.

Topic: "[USER_TOPIC_HERE]"

The spec must be thorough and self-contained. Here is an example of a spec written in response to a video about functional harmony (adapt this format for the topic):

"In music, chords create expectations of movement toward certain other chords and resolution towards a tonal center. This is called functional harmony.

Build me an interactive web app to help a learner understand the concept of functional harmony.

SPECIFICATIONS:
1. The app must feature an interactive keyboard.
2. The app must showcase all 7 diatonic triads that can be created in a major key (i.e., tonic, supertonic, mediant, subdominant, dominant, submediant, leading chord).
3. The app must somehow describe the function of each of the diatonic triads, and state which other chords each triad tends to lead to.
4. The app must provide a way for users to play different chords in sequence and see the results.
[etc.]"

The goal of the app to be built based on your spec is to enhance understanding through simple and playful design. The spec should not be overly complex, i.e., a junior web developer should be able to implement it in a single html file (with all styles and scripts inline). Most importantly, the spec must clearly outline the core mechanics of the app, and those mechanics must be highly effective for learning about the given topic.

Provide your response as a text string formatted as a JSON object, containing a single field called "spec". The value of the "spec" field must be a string. Ensure this string value is valid JSON string content: any double quotes, backslashes, or control characters (like newlines) within the specification text itself must be properly escaped (e.g., \\", \\\\, \\n).
For example: {"spec": "Build me an app about [USER_TOPIC_HERE] with a spec string that is carefully escaped.\\nIt should include features like this: [\\"item1\\", \\"item2\\"]..."}`;

export const REFINE_SPEC_PROMPT_TEMPLATE = `You are an expert web app designer and pedagogist. You will be given an existing specification for an interactive learning web app, and a set of user instructions for how to refine or change that specification.

Your task is to carefully consider the user's refinement instructions and rewrite the original specification to incorporate these changes. Ensure the new specification remains clear, detailed, and suitable for a junior web developer to implement as a single HTML file with inline styles and scripts. The core goal of the app (to enhance understanding through simple and playful design) should be preserved or enhanced by the refinements.

If the user's instructions are unclear or contradictory, try your best to interpret them in a way that improves the app. If the instructions request removal of core functionality that would render the app useless, you may point this out and suggest an alternative, but prioritize fulfilling the user's request if possible.

Original Specification:
---
[EXISTING_SPEC_HERE]
---

User's Refinement Instructions:
---
[USER_REFINEMENT_INSTRUCTIONS_HERE]
---

Provide the **new, complete specification** as a JSON object containing a single field called "spec". The value of the "spec" field must be a string. Ensure this string value is valid JSON string content: any double quotes, backslashes, or control characters (like newlines) within the specification text itself must be properly escaped (e.g., \\", \\\\, \\n).
For example: {"spec": "Updated spec based on user feedback, ensuring all special characters like \\"quotes\\" and newlines (\\n) are escaped."}`;

export const LESSON_PLAN_PROMPT_TEMPLATE = `You are an experienced educator. Based on the following specification for an interactive learning web app, create a concise lesson plan for a teacher to use this app in a classroom setting (e.g., for a 30-45 minute session).

The lesson plan should include:
1.  **Learning Objectives:** What students should be able to do or understand after using the app and the lesson.
2.  **Materials:** List the app itself and any other simple materials needed (e.g., paper, pencils, whiteboard).
3.  **Procedure:** A step-by-step guide for the teacher, including:
    *   Introduction/Engagement (hooking students).
    *   Guided exploration of the app.
    *   Key discussion points related to the app's content.
    *   A simple activity or assessment to check understanding.
4.  **Differentiation (Optional):** Brief suggestions for supporting diverse learners.

App Specification:
---
[APP_SPECIFICATION_HERE]
---

Provide the lesson plan as a well-formatted text response. Use markdown for clarity (headings, lists).`;

export const HANDOUT_PROMPT_TEMPLATE = `You are an instructional designer. Based on the following specification for an interactive learning web app, create a one-page student handout.

The handout should:
1.  Briefly introduce the topic the app covers.
2.  Explain the main goal of using the interactive app.
3.  List 2-3 key concepts or vocabulary terms students will encounter or learn through the app.
4.  Provide 2-3 simple questions or prompts for students to think about or answer while (or after) using the app to guide their learning.
5.  (Optional) Include a small space for notes.

The tone should be student-friendly and encouraging.

App Specification:
---
[APP_SPECIFICATION_HERE]
---

Provide the handout content as a well-formatted text response. Use markdown for clarity.`;

export const QUIZ_PROMPT_TEMPLATE = `You are an expert quiz designer. Based on the following specification for an interactive learning web app, create a short review quiz with 3-5 questions.

For each question:
1.  Provide the question text.
2.  Provide 3-4 multiple-choice options.
3.  Clearly indicate the correct answer.
4.  (Optional but encouraged) Provide a brief explanation for why the correct answer is right, or why the other options are incorrect.

The questions should directly assess understanding of the core concepts presented or interacted with in the app.

App Specification:
---
[APP_SPECIFICATION_HERE]
---

Provide the quiz as a JSON object. The main JSON object should have a key "quiz" which is an array of question objects. Each question object should have keys like "question", "options" (an array of strings), "correctAnswer" (a string matching one of the options), and "explanation" (a string).
Ensure the JSON is strictly valid, paying close attention to commas, quotes, and closing brackets for arrays and objects. For arrays of strings, ensure each string is correctly quoted and separated by commas, and the array is properly terminated with a square bracket.
Example:
{
  "quiz": [
    {
      "question": "What is the primary color of the sun?",
      "options": ["Blue", "Red", "Yellow", "Green"],
      "correctAnswer": "Yellow",
      "explanation": "The sun emits white light, but appears yellow to us due to atmospheric scattering."
    }
  ]
}`;


export const CODE_REGION_OPENER = '```';
export const CODE_REGION_CLOSER = '```';

export const SPEC_ADDENDUM = `\n\nThe app must be fully responsive and function properly on both desktop and mobile. Provide the code as a single, self-contained HTML document. All styles and scripts must be inline. In the result, encase the code between "${CODE_REGION_OPENER}html" and "${CODE_REGION_CLOSER}" for easy parsing. Make sure the HTML is complete, starting with <!DOCTYPE html> and ending with </html>.`;