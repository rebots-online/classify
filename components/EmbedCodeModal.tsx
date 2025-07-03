/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
*/
/* tslint:disable */

import React, { useState } from 'react';

interface EmbedCodeModalProps {
  isOpen: boolean;
  onClose: () => void;
  appCode: string;
}

const EmbedCodeModal: React.FC<EmbedCodeModalProps> = ({ isOpen, onClose, appCode }) => {
  const [fullCodeCopied, setFullCodeCopied] = useState(false);
  const [iframeCopied, setIframeCopied] = useState(false);

  if (!isOpen) {
    return null;
  }

  const iframeSnippet = `<iframe \n  srcdoc="${appCode.replace(/"/g, '&quot;')}" \n  width="100%" \n  height="600px" \n  style="border: 1px solid #ccc; border-radius: 8px;" \n  title="Embedded Learning App" \n  sandbox="allow-scripts allow-modals allow-forms allow-popups allow-same-origin">\n</iframe>`;

  const copyToClipboard = async (text: string, type: 'full' | 'iframe') => {
    try {
      await navigator.clipboard.writeText(text);
      if (type === 'full') {
        setFullCodeCopied(true);
        setTimeout(() => setFullCodeCopied(false), 2000);
      } else {
        setIframeCopied(true);
        setTimeout(() => setIframeCopied(false), 2000);
      }
    } catch (err) {
      console.error('Failed to copy text: ', err);
      alert('Failed to copy text. Please try manually selecting and copying.');
    }
  };

  return (
    <div className="modal-overlay" onClick={onClose} role="dialog" aria-modal="true" aria-labelledby="embedCodeTitle">
      <div className="modal-content" onClick={(e) => e.stopPropagation()} style={{maxWidth: '900px'}}> {/* Wider modal for code */}
        <button className="modal-close-button" onClick={onClose} aria-label="Close embed code dialog">
          <span className="material-symbols-outlined">close</span>
        </button>
        <h2 id="embedCodeTitle" className="modal-title">Embed Your Learning App</h2>
        
        <div className="modal-section">
          <p>You can embed this generated learning app into your own webpage or learning platform using the code below.</p>
        </div>

        <div className="modal-section">
          <div style={{display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.5rem'}}>
            <label htmlFor="fullAppCode" className="embed-code-label">Full App HTML Code:</label>
            <button onClick={() => copyToClipboard(appCode, 'full')} className="modal-copy-button">
              <span className="material-symbols-outlined">{fullCodeCopied ? 'done' : 'content_copy'}</span>
              {fullCodeCopied ? 'Copied!' : 'Copy HTML'}
            </button>
          </div>
          <textarea
            id="fullAppCode"
            className="modal-textarea"
            value={appCode}
            readOnly
            rows={10}
            aria-label="Full HTML code of the generated app"
          />
        </div>

        <div className="modal-section">
           <div style={{display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.5rem'}}>
            <label htmlFor="iframeAppCode" className="embed-code-label">Iframe Embed Snippet:</label>
            <button onClick={() => copyToClipboard(iframeSnippet, 'iframe')} className="modal-copy-button">
              <span className="material-symbols-outlined">{iframeCopied ? 'done' : 'content_copy'}</span>
              {iframeCopied ? 'Copied!' : 'Copy Iframe'}
            </button>
          </div>
          <p style={{fontSize: '0.85rem', color: 'var(--color-subtitle)', marginBottom: '0.5rem'}}>
            Use this snippet to embed the app within an existing page. Adjust width, height, and style as needed.
          </p>
          <textarea
            id="iframeAppCode"
            className="modal-textarea"
            value={iframeSnippet}
            readOnly
            rows={8}
            aria-label="Iframe embed code for the generated app"
          />
        </div>

        <div className="modal-actions">
          <button type="button" className="button-primary" onClick={onClose}>
            Close
          </button>
        </div>
      </div>
    </div>
  );
};

export default EmbedCodeModal;
