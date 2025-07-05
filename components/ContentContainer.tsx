/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
*/
/* tslint:disable */

import Editor from '@monaco-editor/react';
import React, {
  forwardRef,
  useEffect,
  useImperativeHandle,
  useState,
  useCallback,
} from 'react';
import {Tab, TabList, TabPanel, Tabs} from 'react-tabs';
import { ContentBasis, EducationalMaterialRequest } from '@/App';
import RefineAppModal from '@/components/RefineAppModal';
import EmbedCodeModal from '@/components/EmbedCodeModal';
import DeployModal from '@/components/DeployModal'; 

import {parseHTML, parseJSON} from '@/lib/parse';
import {
  SPEC_ADDENDUM,
  SPEC_FROM_VIDEO_PROMPT,
  SPEC_FROM_TOPIC_PROMPT_TEMPLATE,
  REFINE_SPEC_PROMPT_TEMPLATE,
  LESSON_PLAN_PROMPT_TEMPLATE,
  HANDOUT_PROMPT_TEMPLATE,
  QUIZ_PROMPT_TEMPLATE,
} from '@/lib/prompts';
import {generateText, TextGenerationResponse, TextGenerationInteraction} from '@/lib/textGeneration';
import { HarmCategory, HarmBlockThreshold, SafetySetting, GroundingMetadata } from '@google/genai';

interface ContentContainerProps {
  contentBasisInput: ContentBasis;
  // preSeededSpec and preSeededCode props are removed
  onLoadingStateChange?: (isLoading: boolean) => void;
  onLlmInteraction: (interaction: TextGenerationInteraction) => void;
  requestedMaterials: EducationalMaterialRequest;
  isOutputExpanded: boolean;
}

type LoadingState = 
  | 'idle'
  | 'loading-spec' 
  | 'loading-code' 
  | 'refining-spec'
  | 'loading-lesson-plan'
  | 'loading-handout'
  | 'loading-quiz'
  | 'ready' 
  | 'error';

const defaultSafetySettings: SafetySetting[] = [
  { category: HarmCategory.HARM_CATEGORY_HARASSMENT, threshold: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE },
  { category: HarmCategory.HARM_CATEGORY_HATE_SPEECH, threshold: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE },
  { category: HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT, threshold: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE },
  { category: HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT, threshold: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE },
];


export default forwardRef(function ContentContainer(
  {
    contentBasisInput,
    // preSeededSpec, preSeededCode removed from props
    onLoadingStateChange,
    onLlmInteraction,
    requestedMaterials,
    isOutputExpanded,
  }: ContentContainerProps,
  ref,
) {
  const [spec, setSpec] = useState<string>(''); // Initialize as empty
  const [code, setCode] = useState<string>(''); // Initialize as empty
  const [lessonPlan, setLessonPlan] = useState<string | null>(null);
  const [handout, setHandout] = useState<string | null>(null);
  const [quiz, setQuiz] = useState<any | null>(null); 

  const [iframeKey, setIframeKey] = useState(0);
  const [saveMessage, setSaveMessage] = useState('');
  // Initial loading state set to 'loading-spec' as we no longer pre-seed
  const [loadingState, setLoadingState] = useState<LoadingState>('loading-spec');
  const [loadingMessage, setLoadingMessage] = useState<string>('Generating app spec & code...');
  const [error, setError] = useState<string | null>(null);
  const [isEditingSpec, setIsEditingSpec] = useState(false);
  const [editedSpec, setEditedSpec] = useState('');
  const [activeTabIndex, setActiveTabIndex] = useState(0);
  const [groundingMetadata, setGroundingMetadata] = useState<GroundingMetadata | null>(null);

  const [showRefineModal, setShowRefineModal] = useState(false);
  const [showEmbedModal, setShowEmbedModal] = useState(false);
  const [showDeployModal, setShowDeployModal] = useState(false); 


  useImperativeHandle(ref, () => ({
    getSpec: () => spec,
    getCode: () => code,
  }));

  const generateEducationalMaterials = useCallback(async (currentSpec: string) => {
    if (requestedMaterials.lessonPlan) {
      setLoadingState('loading-lesson-plan');
      setLoadingMessage('Generating lesson plan...');
      try {
        const { text } = await generateText({
          modelName: 'gemini-2.5-flash-preview-04-17',
          basePrompt: LESSON_PLAN_PROMPT_TEMPLATE.replace('[APP_SPECIFICATION_HERE]', currentSpec),
          safetySettings: defaultSafetySettings,
          onInteraction: onLlmInteraction,
          stream: true,
        });
        setLessonPlan(text);
      } catch (err) {
        console.error('Error generating lesson plan:', err);
        setLessonPlan(`Error generating lesson plan: ${err instanceof Error ? err.message : 'Unknown error'}`);
      }
    }
    if (requestedMaterials.handout) {
      setLoadingState('loading-handout');
      setLoadingMessage('Generating student handout...');
      try {
        const { text } = await generateText({
          modelName: 'gemini-2.5-flash-preview-04-17',
          basePrompt: HANDOUT_PROMPT_TEMPLATE.replace('[APP_SPECIFICATION_HERE]', currentSpec),
          safetySettings: defaultSafetySettings,
          onInteraction: onLlmInteraction,
          stream: true,
        });
        setHandout(text);
      } catch (err) {
        console.error('Error generating handout:', err);
        setHandout(`Error generating handout: ${err instanceof Error ? err.message : 'Unknown error'}`);
      }
    }
    if (requestedMaterials.quiz) {
      setLoadingState('loading-quiz');
      setLoadingMessage('Generating review quiz...');
      try {
        const { text } = await generateText({
          modelName: 'gemini-2.5-flash-preview-04-17',
          basePrompt: QUIZ_PROMPT_TEMPLATE.replace('[APP_SPECIFICATION_HERE]', currentSpec),
          safetySettings: defaultSafetySettings,
          responseMimeType: "application/json",
          onInteraction: onLlmInteraction,
          stream: true,
        });
        const parsedQuiz = parseJSON(text);
        setQuiz(parsedQuiz.quiz || parsedQuiz); 
      } catch (err) {
        console.error('Error generating quiz:', err);
        setQuiz({ error: `Error generating quiz: ${err instanceof Error ? err.message : 'Unknown error'}` });
      }
    }
  }, [requestedMaterials, onLlmInteraction]);


  const generateSpecAndCode = useCallback(async (videoUrl?: string, topicOrDetails?: string): Promise<{generatedSpec: string, generatedCode: string, searchResults?: GroundingMetadata}> => {
    let specResponse: TextGenerationResponse;
    setLoadingMessage('Generating app spec & code...');
    if (videoUrl) {
      specResponse = await generateText({
        modelName: 'gemini-2.5-flash-preview-04-17',
        basePrompt: SPEC_FROM_VIDEO_PROMPT,
        additionalUserText: topicOrDetails ? `User-provided details to consider: ${topicOrDetails}` : undefined,
        videoUrl: videoUrl,
        responseMimeType: "application/json",
        safetySettings: defaultSafetySettings,
        onInteraction: onLlmInteraction,
        stream: true,
      });
    } else if (topicOrDetails) {
      specResponse = await generateText({
        modelName: 'gemini-2.5-flash-preview-04-17',
        basePrompt: SPEC_FROM_TOPIC_PROMPT_TEMPLATE.replace('[USER_TOPIC_HERE]', topicOrDetails),
        safetySettings: defaultSafetySettings,
        useGoogleSearch: true,
        onInteraction: onLlmInteraction,
        stream: true,
      });
    } else {
      throw new Error("No content basis (video URL or topic) provided for spec generation.");
    }

    const parsedData = parseJSON(specResponse.text);
    let generatedSpecText = parsedData.spec;

    if (typeof generatedSpecText !== 'string') {
        console.error("Parsed spec is not a string:", parsedData);
        throw new Error("The 'spec' field in the JSON response was not a string.");
    }
    generatedSpecText += SPEC_ADDENDUM;
    
    setLoadingMessage('Generating app code from spec...');
    const { text: codeResponseText } = await generateText({
      modelName: 'gemini-2.5-flash-preview-04-17',
      basePrompt: generatedSpecText,
      safetySettings: defaultSafetySettings,
      onInteraction: onLlmInteraction,
      stream: true,
    });
    const generatedCode = parseHTML(codeResponseText);

    return { generatedSpec: generatedSpecText, generatedCode, searchResults: specResponse.groundingMetadata };
  }, [onLlmInteraction]);
  
  const regenerateCodeFromSpec = useCallback(async (currentSpec: string): Promise<string> => {
    setLoadingMessage('Regenerating code from spec...');
    const { text: codeResponseText } = await generateText({
      modelName: 'gemini-2.5-flash-preview-04-17',
      basePrompt: currentSpec,
      safetySettings: defaultSafetySettings,
      onInteraction: onLlmInteraction,
      stream: true,
    });
    const generatedCode = parseHTML(codeResponseText);
    return generatedCode;
  }, [onLlmInteraction]);


  useEffect(() => {
    if (onLoadingStateChange) {
      const isLoading = !['ready', 'error', 'idle'].includes(loadingState);
      onLoadingStateChange(isLoading);
    }
  }, [loadingState, onLoadingStateChange]);

  useEffect(() => {
    async function generateContentFlow() {
      // Logic for preSeededSpec and preSeededCode is removed.
      // The component will always proceed to generate content if contentBasisInput is present.

      const { videoUrl, topicOrDetails } = contentBasisInput;
      if (!videoUrl && !topicOrDetails) {
         // This case should ideally be handled by App.tsx before rendering ContentContainer
         // but as a fallback:
         setError("No video URL or topic provided.");
         setLoadingState('error');
         setActiveTabIndex(0); // Default to spec tab on error
         return;
      }

      try {
        setLoadingState('loading-spec');
        setActiveTabIndex(0);
        setError(null);
        setSpec(''); setCode(''); setGroundingMetadata(null);
        setLessonPlan(null); setHandout(null); setQuiz(null); 

        const { generatedSpec, generatedCode, searchResults } = await generateSpecAndCode(videoUrl, topicOrDetails);
        
        setSpec(generatedSpec);
        setCode(generatedCode);
        if (searchResults) setGroundingMetadata(searchResults);
        
        await generateEducationalMaterials(generatedSpec);

        setLoadingState('ready');
        setActiveTabIndex(2); // Switch to render tab after generation
      } catch (err) {
        console.error('An error occurred while attempting to generate content:', err);
        setError(err instanceof Error ? err.message : 'An unknown error occurred');
        setLoadingState('error');
        if (!spec && !code) setActiveTabIndex(0);
        else if (spec && !code) setActiveTabIndex(1);
      }
    }

    if (contentBasisInput) { // Only run if there's input
        generateContentFlow();
    } else {
        // If no contentBasisInput (e.g., initial load before user interaction), set to idle or placeholder state
        setLoadingState('idle');
        setActiveTabIndex(0);
        setError(null);
        setSpec(''); setCode(''); setGroundingMetadata(null);
        setLessonPlan(null); setHandout(null); setQuiz(null);
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [contentBasisInput, generateSpecAndCode, generateEducationalMaterials]); // preSeeded props removed from dependencies

  useEffect(() => {
    if (code) setIframeKey((prev) => prev + 1);
  }, [code]);

  useEffect(() => {
    if (saveMessage) {
      const timer = setTimeout(() => setSaveMessage(''), 2000);
      return () => clearTimeout(timer);
    }
  }, [saveMessage]);

  const handleCodeChange = (value: string | undefined) => {
    setCode(value || '');
    setSaveMessage('HTML updated. Changes will appear in the Render tab.');
  };

  const handleSpecEdit = () => {
    setEditedSpec(spec);
    setIsEditingSpec(true);
  };

  const handleSpecSave = async () => {
    const trimmedEditedSpec = editedSpec.trim();
    if (trimmedEditedSpec === spec) {
      setIsEditingSpec(false);
      setEditedSpec('');
      return;
    }

    try {
      setLoadingState('loading-code');
      setError(null);
      setSpec(trimmedEditedSpec);
      setIsEditingSpec(false);
      setActiveTabIndex(1);

      const generatedCode = await regenerateCodeFromSpec(trimmedEditedSpec);
      setCode(generatedCode);
      
      setLessonPlan(null); setHandout(null); setQuiz(null); 
      await generateEducationalMaterials(trimmedEditedSpec);

      setLoadingState('ready');
      setActiveTabIndex(2);
    } catch (err) {
      console.error('Error generating code from edited spec:', err);
      setError(err instanceof Error ? err.message : 'An unknown error occurred');
      setLoadingState('error');
      setActiveTabIndex(1);
    }
  };

  const handleSpecCancel = () => {
    setIsEditingSpec(false);
    setEditedSpec('');
  };

  const handleRefineOpen = () => setShowRefineModal(true);
  const handleRefineClose = () => setShowRefineModal(false);

  const handleRefineSubmit = async (refinementInstructions: string) => {
    if (!spec) {
      alert("Cannot refine: No current specification available.");
      return;
    }
    setShowRefineModal(false);
    try {
      setLoadingState('refining-spec');
      setLoadingMessage('Refining app specification...');
      setError(null);
      setActiveTabIndex(0); 

      const { text: refinedSpecJson } = await generateText({
        modelName: 'gemini-2.5-flash-preview-04-17',
        basePrompt: REFINE_SPEC_PROMPT_TEMPLATE.replace('[EXISTING_SPEC_HERE]', spec).replace('[USER_REFINEMENT_INSTRUCTIONS_HERE]', refinementInstructions),
        safetySettings: defaultSafetySettings,
        responseMimeType: "application/json",
        onInteraction: onLlmInteraction,
      });

      const parsedRefinedData = parseJSON(refinedSpecJson);
      let refinedSpecText = parsedRefinedData.spec;
      if (typeof refinedSpecText !== 'string') {
        throw new Error("The 'spec' field in the refined JSON response was not a string.");
      }
      refinedSpecText += SPEC_ADDENDUM;
      setSpec(refinedSpecText);

      setLoadingState('loading-code');
      setLoadingMessage('Generating code from refined spec...');
      const newCode = await regenerateCodeFromSpec(refinedSpecText);
      setCode(newCode);

      setLessonPlan(null); setHandout(null); setQuiz(null); 
      await generateEducationalMaterials(refinedSpecText);
      
      setLoadingState('ready');
      setActiveTabIndex(2); 
    } catch (err) {
      console.error('Error during app refinement:', err);
      setError(err instanceof Error ? err.message : 'An unknown error occurred during refinement');
      setLoadingState('error');
      setActiveTabIndex(0); 
    }
  };
  
  const handleEmbedModalOpen = () => setShowEmbedModal(true);
  const handleEmbedModalClose = () => setShowEmbedModal(false);

  const handleDeployModalOpen = () => setShowDeployModal(true);
  const handleDeployModalClose = () => setShowDeployModal(false);

  const handleSaveProject = () => {
    const name = prompt('Project name?');
    if (!name) return;
    const data = {
      spec,
      code,
      lessonPlan,
      handout,
      quiz,
    };
    localStorage.setItem(`project_${name}`, JSON.stringify(data));
    alert('Project saved');
  };

  const handleLoadProject = () => {
    const keys = Object.keys(localStorage).filter(k => k.startsWith('project_'));
    if (keys.length === 0) { alert('No saved projects'); return; }
    const name = prompt(`Load which project?\n${keys.map(k=>k.replace('project_','')).join(', ')}`);
    if (!name) return;
    const raw = localStorage.getItem(`project_${name}`);
    if (!raw) { alert('Project not found'); return; }
    try {
      const data = JSON.parse(raw);
      setSpec(data.spec || '');
      setCode(data.code || '');
      setLessonPlan(data.lessonPlan || null);
      setHandout(data.handout || null);
      setQuiz(data.quiz || null);
      setLoadingState('ready');
      setActiveTabIndex(2);
    } catch (e) {
      alert('Failed to load project');
    }
  };

  const renderLoadingSpinner = () => (
    <div className="loading-container">
      <div className="loading-spinner"></div>
      <p className="loading-text">{loadingMessage}</p>
    </div>
  );

  const renderErrorState = () => (
    <div className="error-container">
      <div className="error-icon">error</div>
      <h3 className="error-title">Error</h3>
      <p>{error || 'Something went wrong'}</p>
      {contentBasisInput?.videoUrl && !(contentBasisInput.videoUrl.startsWith('http://') || contentBasisInput.videoUrl.startsWith('https://')) && (
        <p className="error-note">
          (<strong>NOTE:</strong> URL must begin with http:// or https://)
        </p>
      )}
    </div>
  );

  const tabListStyle = { };
  const tabStyle = { };

  const renderSpecContent = () => {
    if ((loadingState === 'error' && !spec) || loadingState === 'loading-spec' || loadingState === 'refining-spec') {
      return loadingState === 'error' ? renderErrorState() : renderLoadingSpinner();
    }

    if (isEditingSpec) {
      return (
        <div style={{height: '100%', display: 'flex', flexDirection: 'column'}}>
          <Editor
            height="calc(100% - 50px)"
            defaultLanguage="text"
            value={editedSpec}
            onChange={(value) => setEditedSpec(value || '')}
            theme="light" options={{ minimap: {enabled: false}, fontSize: 14, wordWrap: 'on', lineNumbers: 'off' }}
          />
          <div style={{display: 'flex', gap: '6px', padding: '10px 1rem', borderTop: '1px solid #eee' }}>
            <button onClick={handleSpecSave} className="button-primary">Save & regenerate code</button>
            <button onClick={handleSpecCancel} className="button-secondary">Cancel</button>
          </div>
        </div>
      );
    }

    return (
      <div style={{height: '100%', display: 'flex', flexDirection: 'column', overflowY: 'auto'}}>
        <div style={{ whiteSpace: 'pre-wrap', fontFamily: 'var(--font-technical)', lineHeight: 1.75, flex: 1, padding: '1rem 2rem', overflow: 'visible' }}>
          {spec || (loadingState === 'error' && "Error loading spec, but previous spec might be available.") || (loadingState === 'idle' && "Spec will appear here after generation.")} 
        </div>
        {groundingMetadata && groundingMetadata.groundingChunks && groundingMetadata.groundingChunks.length > 0 && (
          <div className="grounding-info">
            <h4>Sources from Google Search:</h4>
            <ul>
              {groundingMetadata.groundingChunks.map((chunk, index) => (
                (chunk.web && chunk.web.uri) && ( 
                  <li key={index}>
                    <a href={chunk.web.uri} target="_blank" rel="noopener noreferrer">
                      {chunk.web.title || chunk.web.uri}
                    </a>
                  </li>
                )
              ))}
            </ul>
          </div>
        )}
        {loadingState === 'error' && spec && renderErrorState()} 
      </div>
    );
  };

  const renderQuizContent = () => {
    if (loadingState === 'loading-quiz') return renderLoadingSpinner();
    if (!quiz || quiz.error) return <div className="content-placeholder-text">{quiz?.error || "Quiz will appear here once generated."}</div>;
    
    const quizData = Array.isArray(quiz) ? quiz : (quiz.questions || []); 
    if (!Array.isArray(quizData) || quizData.length === 0) return <div className="content-placeholder-text">No quiz questions found.</div>;

    return (
        <div className="quiz-content">
            {quizData.map((item, index) => (
                <div key={index} className="quiz-item">
                    <p className="quiz-question"><strong>{index + 1}. {item.question}</strong></p>
                    {item.options && (
                        <ul className="quiz-options">
                            {item.options.map((option: string, optIndex: number) => (
                                <li key={optIndex} className={option === item.correctAnswer ? 'correct-answer' : ''}>
                                    {option}
                                    {option === item.correctAnswer && <span className="correct-answer-indicator"> (Correct)</span>}
                                </li>
                            ))}
                        </ul>
                    )}
                    {item.explanation && <p className="quiz-explanation"><em>Explanation:</em> {item.explanation}</p>}
                </div>
            ))}
        </div>
    );
};


  const isLoading = !['ready', 'error', 'idle'].includes(loadingState);
  const codeTabDisabledCondition = ((isLoading && !code) ? true : (loadingState === 'idle'));
  const materialTabDisabledCondition = (isLoading ? true : (loadingState === 'idle'));

  return (
    <div className="content-container-wrapper">
      <div className="content-actions-bar">
          <button
            onClick={handleSpecEdit}
            disabled={!spec || isLoading || isEditingSpec || loadingState === 'idle'}
            className="button-secondary action-button">
            <span className="material-symbols-outlined">edit_document</span> Edit Spec
          </button>
          <button
            onClick={handleRefineOpen}
            disabled={!spec || isLoading || isEditingSpec || loadingState === 'idle'}
            className="button-secondary action-button">
            <span className="material-symbols-outlined">auto_fix_high</span> Refine App
          </button>
          <button
            onClick={handleEmbedModalOpen}
            disabled={!code || isLoading || loadingState === 'idle'}
            className="button-secondary action-button">
            <span className="material-symbols-outlined">integration_instructions</span> Get Embed Code
          </button>
          <button
            onClick={handleDeployModalOpen}
            disabled={!code || isLoading || loadingState === 'idle'}
            className="button-secondary action-button">
            <span className="material-symbols-outlined">publish</span> Deploy to Web
          </button>
          <button onClick={handleSaveProject} className="button-secondary action-button">
            <span className="material-symbols-outlined">save</span> Save Project
          </button>
          <button onClick={handleLoadProject} className="button-secondary action-button">
            <span className="material-symbols-outlined">folder_open</span> Load Project
          </button>
      </div>

      <Tabs
        className="content-tabs"
        selectedIndex={activeTabIndex}
        onSelect={(index) => {
          if (isEditingSpec && index !== 0) { 
            setIsEditingSpec(false);
            setEditedSpec('');
          }
          setActiveTabIndex(index);
        }}>
        <TabList style={tabListStyle}>
          <Tab style={tabStyle} selectedClassName="selected-tab" disabled={isLoading && activeTabIndex !==0}>Spec</Tab>
          <Tab style={tabStyle} selectedClassName="selected-tab" disabled={codeTabDisabledCondition && activeTabIndex !==1}>Code</Tab>
          <Tab style={tabStyle} selectedClassName="selected-tab" disabled={(loadingState !== 'ready' && loadingState !== 'idle') && activeTabIndex !== 2}>Render</Tab>
          {requestedMaterials.lessonPlan && <Tab style={tabStyle} selectedClassName="selected-tab" disabled={materialTabDisabledCondition && activeTabIndex !== 3}>Lesson Plan</Tab>}
          {requestedMaterials.handout && <Tab style={tabStyle} selectedClassName="selected-tab" disabled={materialTabDisabledCondition && activeTabIndex !== (3 + (requestedMaterials.lessonPlan ? 1:0))}>Handout</Tab>}
          {requestedMaterials.quiz && <Tab style={tabStyle} selectedClassName="selected-tab" disabled={materialTabDisabledCondition && activeTabIndex !== (3 + (requestedMaterials.lessonPlan ? 1:0) + (requestedMaterials.handout ? 1:0))}>Quiz</Tab>}
        </TabList>

        <div className="tab-panel-container">
          <TabPanel className="react-tabs__tab-panel" selectedClassName="react-tabs__tab-panel--selected full-height-panel">
            {renderSpecContent()}
          </TabPanel>
          <TabPanel className="react-tabs__tab-panel" selectedClassName="react-tabs__tab-panel--selected full-height-panel">
            {loadingState === 'error' && !code ? renderErrorState() : ((isLoading && !code) || loadingState === 'idle') ? (loadingState === 'idle' ? <div className="content-placeholder-text">Code will appear here.</div> : renderLoadingSpinner()) : (
              <div style={{height: '100%', position: 'relative'}}>
                <Editor height="100%" defaultLanguage="html" value={code} onChange={handleCodeChange} theme="vs-dark" options={{ minimap: {enabled: false}, fontSize: 14, wordWrap: 'on', formatOnPaste: true, formatOnType: true }} />
                {saveMessage && <div className="save-message">{saveMessage}</div>}
              </div>
            )}
          </TabPanel>
          <TabPanel className="react-tabs__tab-panel" selectedClassName="react-tabs__tab-panel--selected full-height-panel">
            {(() => {
              const renderIframeContent = (
                <div style={{height: '100%', width: '100%', position: 'relative'}}>
                  <iframe key={iframeKey} srcDoc={code} style={{ border: 'none', width: '100%', height: '100%' }} title="rendered-html" sandbox="allow-scripts allow-modals allow-forms allow-popups allow-same-origin" />
                </div>
              );

              if (loadingState === 'ready') {
                return renderIframeContent;
              }
              if (loadingState === 'idle') {
                return <div className="content-placeholder-text">Rendered app will appear here.</div>;
              }
              // isLoading is true if loadingState is one of 'loading-spec', 'loading-code', 'refining-spec', 'loading-lesson-plan', 'loading-handout', 'loading-quiz'
              if (isLoading) { 
                return renderLoadingSpinner();
              }
              if (loadingState === 'error') {
                if (!code) { // Error and no code was ever generated
                  return renderErrorState();
                } else { // Error, but some code exists (e.g. from previous run, or refining failed)
                  return renderIframeContent; // Show previous good state or current code despite error during refinement
                }
              }
              return null; // Should not be reached if all states are handled
            })()}
          </TabPanel>
          {requestedMaterials.lessonPlan && (
            <TabPanel className="react-tabs__tab-panel" selectedClassName="react-tabs__tab-panel--selected full-height-panel">
              {loadingState === 'loading-lesson-plan' ? renderLoadingSpinner() : lessonPlan ? <div className="educational-material-content" dangerouslySetInnerHTML={{ __html: lessonPlan.replace(/\n/g, '<br />') }}/> : <div className="content-placeholder-text">Lesson plan will appear here.</div>}
            </TabPanel>
          )}
          {requestedMaterials.handout && (
            <TabPanel className="react-tabs__tab-panel" selectedClassName="react-tabs__tab-panel--selected full-height-panel">
              {loadingState === 'loading-handout' ? renderLoadingSpinner() : handout ? <div className="educational-material-content" dangerouslySetInnerHTML={{ __html: handout.replace(/\n/g, '<br />') }}/> : <div className="content-placeholder-text">Student handout will appear here.</div>}
            </TabPanel>
          )}
          {requestedMaterials.quiz && (
            <TabPanel className="react-tabs__tab-panel" selectedClassName="react-tabs__tab-panel--selected full-height-panel">
              {renderQuizContent()}
            </TabPanel>
          )}
        </div>
      </Tabs>

      {showRefineModal && (
        <RefineAppModal
          isOpen={showRefineModal}
          onClose={handleRefineClose}
          onSubmit={handleRefineSubmit}
        />
      )}
      {showEmbedModal && (
        <EmbedCodeModal
          isOpen={showEmbedModal}
          onClose={handleEmbedModalClose}
          appCode={code}
        />
      )}
      {showDeployModal && (
        <DeployModal
          isOpen={showDeployModal}
          onClose={handleDeployModalClose}
          appCode={code}
        />
      )}

      <style>{`
        /* Styles from before, plus new/updated styles */
        .content-container-wrapper {
          border: 2px solid light-dark(#000, #fff);
          border-radius: 8px;
          box-sizing: border-box;
          display: flex;
          flex-direction: column;
          height: 100%;
          max-height: inherit;
          min-height: inherit;
          overflow: hidden;
          position: relative;
        }
        .content-actions-bar {
          display: flex;
          flex-wrap: wrap; 
          gap: 0.5rem;
          padding: 0.5rem 1rem; 
          border-bottom: 1px solid light-dark(#ccc, #555);
          background-color: light-dark(#f9f9f9, #333);
        }
        .action-button {
          padding: 0.35rem 0.7rem; 
          font-size: 0.85rem;
          display: flex;
          align-items: center;
          gap: 0.3rem;
        }
        .action-button .material-symbols-outlined {
          font-size: 1.1rem; 
        }

        .content-tabs { 
          display: flex;
          flex-direction: column;
          flex-grow: 1; 
          overflow: hidden; 
        }
        .tab-panel-container {
          flex: 1; 
          overflow: hidden;
          position: relative;
        }
        .react-tabs__tab-panel {
          border-top: 1px solid light-dark(#000, #fff);
          padding: 0; 
          box-sizing: border-box;
          height: 0; 
          overflow: hidden;
        }
        .react-tabs__tab-panel--selected.full-height-panel {
           height: 100%; 
           overflow: auto; 
           padding: 1rem; 
        }
        .react-tabs__tab-panel--selected.full-height-panel:has(.monaco-editor),
        .react-tabs__tab-panel--selected.full-height-panel:has(iframe) {
            padding: 0;
        }
        .selected-tab {
          background: light-dark(#f0f0f0, #4a4a4a); 
          color: light-dark(#000, #fff);
          font-weight: bold;
          border-bottom: 1px solid transparent !important;
        }
        .loading-container, .error-container { /* As before */ }
        .loading-spinner { /* As before */ }
        .loading-text, .error-title, .error-note { /* As before */ }
        .error-icon, .error-container p { /* As before */ }
        .save-message { /* As before */ }
        .grounding-info { /* As before */ }
        .educational-material-content { white-space: pre-wrap; font-family: var(--font-secondary); padding: 1rem; }
        .quiz-content { padding: 1rem; }
        .quiz-item { margin-bottom: 1.5rem; padding-bottom: 1rem; border-bottom: 1px solid #eee; }
        .quiz-question { font-weight: bold; margin-bottom: 0.5rem; color: var(--color-text); }
        .quiz-options { list-style-type: none; padding-left: 0; margin-bottom: 0.5rem; }
        .quiz-options li { margin-bottom: 0.25rem; padding: 0.2rem 0; }
        .quiz-options li.correct-answer { color: var(--color-positive); font-weight: bold; }
        .correct-answer-indicator { font-size: 0.9em; margin-left: 5px; }
        .quiz-explanation { font-size: 0.9em; color: var(--color-subtitle); margin-top: 0.5rem; }
        .content-placeholder-text { padding: 2rem; text-align: center; color: var(--color-subtitle); }
        @keyframes spin { to { transform: rotate(360deg); } }
      `}</style>
    </div>
  );
});