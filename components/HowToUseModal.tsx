/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
*/
/* tslint:disable */

import React from 'react';

interface HowToUseModalProps {
  isOpen: boolean;
  onClose: () => void;
}

const HowToUseModal: React.FC<HowToUseModalProps> = ({ isOpen, onClose }) => {
  if (!isOpen) {
    return null;
  }

  return (
    <div className="modal-overlay" onClick={onClose} role="dialog" aria-modal="true" aria-labelledby="howToUseTitle">
      <div className="modal-content" onClick={(e) => e.stopPropagation()}>
        <button className="modal-close-button" onClick={onClose} aria-label="Close manual">
          <span className="material-symbols-outlined">close</span>
        </button>
        <h2 id="howToUseTitle" className="modal-title">How to Use This App</h2>
        
        <div className="modal-section">
          <h3>Introduction</h3>
          <p>This app uses AI to turn YouTube videos or your topic ideas into simple, interactive learning apps for fun and engaging concept exploration.</p>
        </div>

        <div className="modal-section">
          <h3>Getting Started</h3>
          <ol>
            <li>
              <strong>Input Your Content Idea:</strong>
              <ul>
                <li>Use the single input box to type or paste:
                    <ul>
                        <li>A YouTube URL (e.g., <code>https://www.youtube.com/watch?v=...</code>).</li>
                        <li>A topic or detailed app idea (e.g., "An interactive quiz about the planets").</li>
                        <li>A combination of both (e.g., "<code>https://www.youtube.com/watch?v=xyz</code> Explain the main concepts in this video about photosynthesis and make an app for 7th graders.").</li>
                    </ul>
                </li>
                <li><em>Tip:</em> If a YouTube URL is included, the AI uses it as a primary source. Any other text serves as additional context or instructions. If no URL, the entire text is the basis for the app idea.</li>
              </ul>
            </li>
             <li>
                <strong>Request Materials (Optional):</strong> Check boxes for Lesson Plan, Handout, or Quiz if you want AI-generated supplements.
            </li>
            <li><strong>Generate App:</strong> Click "Generate App". (URL validation occurs if a URL is detected).</li>
          </ol>
        </div>

        <div className="modal-section">
          <h3>The AI Process</h3>
          <p>When you click "Generate App":</p>
          <ol>
            <li><strong>Input Analysis:</strong> The AI checks your input for a YouTube URL.
                <ul>
                    <li>If a URL is found, it's used as the main content source. Other text provides context.</li>
                    <li>If no URL, all text becomes the topic/idea.</li>
                </ul>
            </li>
            <li><strong>Spec Generation:</strong> AI analyzes the derived content basis. For topics without a video, it may use Google Search (sources shown in "Spec" tab). It creates a detailed "Specification" (plan) for the app.</li>
            <li><strong>Code Generation:</strong> AI writes HTML, CSS, & JavaScript for the app based on the spec.</li>
            <li><strong>Materials Generation (if requested):</strong> AI creates selected educational items using the app spec.</li>
          </ol>
          <p><em>LLM Log:</em> See AI queries/responses via the <span className="material-symbols-outlined" style={{fontSize: '1em', verticalAlign: 'middle'}}>terminal</span> icon in the header.</p>
        </div>

        <div className="modal-section">
          <h3>Interacting with Output</h3>
          <p>Generated content appears in tabs on the right:</p>
          <ul>
            <li><strong>Spec Tab:</strong>
              <ul>
                <li>View/edit AI's app plan.</li>
                <li>Click "Edit Spec", make changes, then "Save & regenerate code" to update the app & materials.</li>
                <li>If Google Search was used, source URLs are listed here.</li>
              </ul>
            </li>
            <li><strong>Code Tab:</strong> View/edit HTML/CSS/JS. Changes are live in "Render".</li>
            <li><strong>Render Tab:</strong> Interact with your learning app.</li>
            <li><strong>Lesson Plan / Handout / Quiz Tabs:</strong> View generated supplementary materials.</li>
          </ul>
        </div>
        
        <div className="modal-section">
          <h3>Refining & Embedding</h3>
          <ul>
            <li>
                <strong>Refine App:</strong>
                <ul>
                    <li>Click <span className="material-symbols-outlined" style={{fontSize: '1em', verticalAlign: 'middle'}}>auto_fix_high</span> "Refine App" button (near "Edit Spec").</li>
                    <li>In the modal, type your desired changes (e.g., "add a timer," "change colors").</li>
                    <li>AI uses your instructions and the current spec to generate a *new* spec and *new* code.</li>
                </ul>
            </li>
            <li>
                <strong>Get Embed Code:</strong>
                <ul>
                    <li>Click <span className="material-symbols-outlined" style={{fontSize: '1em', verticalAlign: 'middle'}}>integration_instructions</span> "Get Embed Code" button (above tabs).</li>
                    <li>Modal shows full HTML and an <code>&lt;iframe&gt;</code> snippet.</li>
                    <li>Copy code to embed app on your website or learning platform.</li>
                </ul>
            </li>
             <li>
                <strong>Deploy to Web:</strong>
                <ul>
                    <li>Click <span className="material-symbols-outlined" style={{fontSize: '1em', verticalAlign: 'middle'}}>publish</span> "Deploy to Web" button (above tabs).</li>
                    <li>Download the <code>index.html</code> file.</li>
                    <li>Follow instructions to deploy to a service like Netlify Drop for a live URL.</li>
                </ul>
            </li>
          </ul>
        </div>
        
        <div className="modal-section">
            <h3>Schematic Diagram</h3>
            <div className="flowchart">
                <div className="flow-node input-node">
                    <strong>User Input</strong>
                    <div className="flow-sub-item">Single Text Area: YouTube URL / Topic / App Idea (or combination)</div>
                    <div className="flow-sub-item">Optional Materials Request</div>
                </div>
                <div className="flow-arrow">&darr;</div>
                <div className="flow-node process-node">
                    <strong>AI Processing Engine</strong>
                     <div className="flow-sub-process">
                        <div className="sub-node">0. Input Parsing
                            <div className="flow-detail">(Detect URL, separate topic/details)</div>
                        </div>
                    </div>
                    <div className="flow-arrow sub-arrow">&darr;</div>
                    <div className="flow-sub-process">
                        <div className="sub-node">1. Spec Generation
                            <div className="flow-detail">(Video/Topic Analysis) &rarr; App Spec</div>
                        </div>
                         <div className="llm-link"><span>&harr;</span> LLM Log</div>
                    </div>
                    <div className="flow-arrow sub-arrow">&darr; (OR, if app exists & "Refine App" clicked)</div>
                     <div className="flow-sub-process">
                        <div className="sub-node">1b. Refine Spec
                            <div className="flow-detail">(Current Spec + User Changes) &rarr; New App Spec</div>
                        </div>
                         <div className="llm-link right"><span>&harr;</span> LLM Log</div>
                    </div>
                    <div className="flow-arrow sub-arrow">&darr;</div>
                    <div className="flow-sub-process">
                        <div className="sub-node">2. Code Generation
                            <div className="flow-detail">(Input: App Spec) &rarr; App Code (HTML/CSS/JS)</div>
                        </div>
                         <div className="llm-link"><span>&harr;</span> LLM Log</div>
                    </div>
                     <div className="flow-arrow sub-arrow">&darr; (If requested)</div>
                    <div className="flow-sub-process">
                        <div className="sub-node">3. Educational Materials
                            <div className="flow-detail">(Input: App Spec) &rarr; Lesson Plan, etc.</div>
                        </div>
                         <div className="llm-link right"><span>&harr;</span> LLM Log</div>
                    </div>
                </div>
                <div className="flow-arrow">&darr;</div>
                <div className="flow-node output-node">
                    <strong>Application Output / Interaction</strong>
                    <div className="flow-sub-item">Spec Tab (View/Edit)</div>
                    <div className="flow-sub-item">Code Tab (View/Edit)</div>
                    <div className="flow-sub-item">Render Tab (Interact)</div>
                    <div className="flow-sub-item">Materials Tabs (View)</div>
                    <div className="flow-sub-item">"Refine App" / "Get Embed Code" / "Deploy to Web" Buttons</div>
                </div>
                 <div className="flow-node llm-panel-node">
                    <strong>LLM Log Panel</strong>
                    <div>(View AI Prompts & Responses)</div>
                </div>
            </div>
        </div>

        <div className="modal-section">
          <h3>Tips for Best Results</h3>
          <ul>
            <li><strong>Be Specific:</strong> More detail helps AI create better apps. If providing a URL and text, make the text clearly state how it relates to the video or what extra you want.</li>
            <li><strong>Clear URLs:</strong> Ensure YouTube URLs are correct and publicly accessible.</li>
            <li><strong>Iterate & Experiment:</strong> Use "Edit Spec" and "Refine App" to customize.</li>
            <li><strong>Check the Log:</strong> The LLM Log can provide clues if something seems off.</li>
          </ul>
        </div>

        <div className="modal-actions" style={{textAlign: 'center', marginTop: '1rem'}}>
          <button onClick={onClose} className="button-primary">Get Started!</button>
        </div>

      </div>
    </div>
  );
};

export default HowToUseModal;
