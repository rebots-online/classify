/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
*/
/* tslint:disable */

import ContentContainer from '@/components/ContentContainer';
import Header from '@/components/Header';
import Footer from '@/components/Footer';
import LlmLogPanel from '@/components/LlmLogPanel';
import HowToUseModal from '@/components/HowToUseModal';

import {DataContext} from '@/context';
import {TextGenerationInteraction} from '@/lib/textGeneration';
import {
  getYoutubeEmbedUrl,
  validateYoutubeUrl,
  getYouTubeVideoId, // Import for URL detection
} from '@/lib/youtube';
import {useContext, useEffect, useRef, useState, useCallback} from 'react'; 
import { v4 as uuidv4 } from 'uuid';

const VALIDATE_INPUT_URL = true;

export interface ContentBasis {
  videoUrl?: string;
  topicOrDetails?: string;
}

export interface LlmInteraction {
  id: string;
  timestamp: string;
  type: 'PROMPT' | 'RESPONSE' | 'ERROR' | 'TOKEN';
  model?: string;
  data: any;
}

export interface EducationalMaterialRequest {
  lessonPlan: boolean;
  handout: boolean;
  quiz: boolean;
}

// Regex to find a YouTube URL in a string
const YOUTUBE_URL_REGEX = /(https?:\/\/(?:www\.)?(?:youtube\.com\/(?:watch\?v=|embed\/|shorts\/)|youtu\.be\/)[\w\-]+)/i;


export default function App() {
  const {} = useContext(DataContext); 

  const [contentBasis, setContentBasis] = useState<ContentBasis | null>(null);
  const [inputProcessing, setInputProcessing] = useState(false); // Combines urlValidating and initial part of contentLoading
  const [contentLoading, setContentLoading] = useState(false); // Specifically for LLM generation phase
  const [isOutputExpanded, setIsOutputExpanded] = useState(false); 

  const contentContainerRef = useRef<{
    getSpec: () => string;
    getCode: () => string;
  } | null>(null);

  const [reloadCounter, setReloadCounter] = useState(0);
  const contentInputRef = useRef<HTMLTextAreaElement>(null); // Single input ref

  const [llmInteractions, setLlmInteractions] = useState<LlmInteraction[]>([]);
  const [showLlmLogPanel, setShowLlmLogPanel] = useState(true); 
  const [showHowToUseModal, setShowHowToUseModal] = useState(false);

  const [requestedMaterials, setRequestedMaterials] = useState<EducationalMaterialRequest>({
    lessonPlan: false,
    handout: false,
    quiz: false,
  });

  const handleLlmInteraction = useCallback((interaction: TextGenerationInteraction) => {
    setLlmInteractions((prevInteractions) => [
      ...prevInteractions,
      {
        id: uuidv4(),
        timestamp: new Date().toISOString(),
        type: interaction.type,
        model: interaction.modelName,
        data: interaction.data,
      },
    ]);
  }, []); 

  const handleToggleLlmLogPanel = () => {
    setShowLlmLogPanel(prev => !prev);
  };

  const handleClearLlmLog = () => {
    setLlmInteractions([]);
  };

  const toggleHowToUseModal = () => {
    setShowHowToUseModal(prev => !prev);
  };

  const handleToggleExpandOutput = () => {
    setIsOutputExpanded(prev => !prev);
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey && !inputProcessing && !contentLoading) {
      e.preventDefault();
      handleSubmit();
    }
  };

  const handleSubmit = async () => {
    const inputValue = contentInputRef.current?.value.trim() || '';

    if (!inputValue) {
      if (contentInputRef.current) contentInputRef.current.focus();
      return;
    }

    if (inputProcessing || contentLoading) return;

    setInputProcessing(true);
    setContentBasis(null);
    // setContentLoading(false); // This will be set by onLoadingStateChange

    const urlMatch = inputValue.match(YOUTUBE_URL_REGEX);
    let videoUrlToProcess: string | undefined = undefined;
    let topicOrDetailsToProcess: string | undefined = inputValue;

    if (urlMatch && urlMatch[0]) {
      const potentialUrl = urlMatch[0];
      // Basic check if it's a valid ID format, actual validation below
      if (getYouTubeVideoId(potentialUrl)) {
        videoUrlToProcess = potentialUrl;
        topicOrDetailsToProcess = inputValue.replace(potentialUrl, '').trim();
        if (topicOrDetailsToProcess === '') {
          topicOrDetailsToProcess = undefined; // Avoid sending empty string as topic
        }
      }
    }
    
    if (videoUrlToProcess && VALIDATE_INPUT_URL) {
      const validationResult = await validateYoutubeUrl(videoUrlToProcess);
      if (validationResult.isValid) {
        proceedWithContent(videoUrlToProcess, topicOrDetailsToProcess);
      } else {
        alert(validationResult.error || 'Invalid YouTube URL detected in input.');
        setInputProcessing(false);
        contentInputRef.current?.focus();
      }
    } else if (videoUrlToProcess) { // URL detected but no validation needed or passed
      proceedWithContent(videoUrlToProcess, topicOrDetailsToProcess);
    } 
    else { // No valid URL detected, use full input as topic
      proceedWithContent(undefined, inputValue);
    }
  };

  const proceedWithContent = (url?: string, topic?: string) => {
    setContentBasis({ videoUrl: url, topicOrDetails: topic });
    setReloadCounter((c) => c + 1);
    setInputProcessing(false); // Finished input processing, LLM loading will be handled by ContentContainer
  };

  const handleContentLoadingStateChange = (isLoading: boolean) => {
    setContentLoading(isLoading);
     if (!isLoading) { // If LLM loading finishes (or errors out), input processing is also done
        setInputProcessing(false);
    }
  };
  
  const isFormDisabled = inputProcessing || contentLoading;
  const currentVideoUrl = contentBasis?.videoUrl || ''; 

  const handleMaterialRequestChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const { name, checked } = event.target;
    setRequestedMaterials(prev => ({ ...prev, [name]: checked }));
  };

  return (
    <>
      <div className="app-wrapper">
        <Header 
          siteTitle="Video to Learning App"
          subTitle="Generate interactive learning apps from YouTube content or topics"
          showLlmLogPanel={showLlmLogPanel}
          onToggleLlmLogPanel={handleToggleLlmLogPanel}
          onToggleHowToUse={toggleHowToUseModal}
        />
        <div className="content-pusher">
          <main className={`main-container ${isOutputExpanded ? 'output-expanded' : ''}`}>
            <div className="left-side">
              <div className="input-section">
                <div className="input-container">
                  <label htmlFor="content-input" className="input-label">
                    Enter YouTube URL, Topic, or App Idea:
                  </label>
                  <textarea
                    ref={contentInputRef}
                    id="content-input"
                    className="content-input" // New class for combined input
                    rows={4} // Adjusted rows
                    placeholder="e.g., 'https://www.youtube.com/watch?v=xyz' or 'Quantum physics for beginners' or combine both!"
                    disabled={isFormDisabled}
                    onKeyDown={handleKeyDown}
                    onChange={() => {
                      setContentBasis(null);
                    }}
                  />
                </div>

                <div className="educational-materials-request">
                  <h4>Additional Materials (Optional):</h4>
                  <div>
                    <input type="checkbox" id="lessonPlan" name="lessonPlan" className="educational-material-checkbox" onChange={handleMaterialRequestChange} disabled={isFormDisabled} />
                    <label htmlFor="lessonPlan">Lesson Plan</label>
                  </div>
                  <div>
                    <input type="checkbox" id="handout" name="handout" className="educational-material-checkbox" onChange={handleMaterialRequestChange} disabled={isFormDisabled} />
                    <label htmlFor="handout">Student Handout</label>
                  </div>
                  <div>
                    <input type="checkbox" id="quiz" name="quiz" className="educational-material-checkbox" onChange={handleMaterialRequestChange} disabled={isFormDisabled} />
                    <label htmlFor="quiz">Review Quiz</label>
                  </div>
                </div>

                <div className="button-container">
                  <button
                    onClick={handleSubmit}
                    className="button-primary submit-button"
                    disabled={isFormDisabled}
                  >
                    {inputProcessing && !contentLoading
                      ? 'Processing Input...'
                      : contentLoading 
                        ? 'Generating App...'
                        : 'Generate App'}
                  </button>
                </div>
              </div>

              <div className="video-container">
                {currentVideoUrl ? (
                  <iframe
                    className="video-iframe"
                    src={getYoutubeEmbedUrl(currentVideoUrl)}
                    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
                    allowFullScreen></iframe>
                ) : contentBasis?.topicOrDetails && !contentBasis?.videoUrl ? (
                    <div className="video-placeholder">Generating app from topic: "{contentBasis.topicOrDetails.substring(0,60)}{contentBasis.topicOrDetails.length > 60 ? '...' : ''}"</div>
                ) : (
                  <div className="video-placeholder">Video appears here if URL is provided</div>
                )}
              </div>
            </div>

            <div className={`right-side ${showLlmLogPanel ? 'llm-log-visible' : ''}`}>
              <div className="content-area-header">
                 {contentBasis && (
                   <button
                    onClick={handleToggleExpandOutput}
                    className="button-secondary expand-toggle-button"
                    title={isOutputExpanded ? "Collapse Output" : "Expand Output"}
                    aria-pressed={isOutputExpanded}
                  >
                    <span className="material-symbols-outlined">
                      {isOutputExpanded ? 'contract_content' : 'expand_content'}
                    </span>
                  </button>
                 )}
              </div>
              <div className="content-area">
                {contentBasis ? (
                  <ContentContainer
                    key={reloadCounter}
                    contentBasisInput={contentBasis}
                    onLoadingStateChange={handleContentLoadingStateChange}
                    onLlmInteraction={handleLlmInteraction}
                    requestedMaterials={requestedMaterials}
                    ref={contentContainerRef}
                    isOutputExpanded={isOutputExpanded}
                  />
                ) : (
                  <div className="content-placeholder">
                    <p>
                      {inputProcessing
                        ? 'Processing input...'
                        : 'Enter a YouTube URL, topic, or app idea to begin'}
                    </p>
                  </div>
                )}
              </div>
               {showLlmLogPanel && (
                <LlmLogPanel
                  interactions={llmInteractions}
                  onClearLog={handleClearLlmLog}
                  isOutputExpanded={isOutputExpanded}
                />
              )}
            </div>
          </main>
        </div>
        <Footer 
          attributionText="An experiment by <strong>Robin L. M. Cheung, MBA</strong>"
        />
      </div>
      {showHowToUseModal && (
        <HowToUseModal isOpen={showHowToUseModal} onClose={toggleHowToUseModal} />
      )}
    </>
  );
}
