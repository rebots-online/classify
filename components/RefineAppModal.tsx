/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
*/
/* tslint:disable */

import React, { useState } from 'react';

interface RefineAppModalProps {
  isOpen: boolean;
  onClose: () => void;
  onSubmit: (refinementText: string) => void;
  // currentSpecPreview?: string; // Optional: Could be added later for more context
}

const RefineAppModal: React.FC<RefineAppModalProps> = ({ isOpen, onClose, onSubmit }) => {
  const [refinementText, setRefinementText] = useState('');

  if (!isOpen) {
    return null;
  }

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (refinementText.trim()) {
      onSubmit(refinementText.trim());
    } else {
      alert("Please enter your refinement instructions.");
    }
  };

  return (
    <div className="modal-overlay" onClick={onClose} role="dialog" aria-modal="true" aria-labelledby="refineAppTitle">
      <div className="modal-content" onClick={(e) => e.stopPropagation()}>
        <button className="modal-close-button" onClick={onClose} aria-label="Close refine app dialog">
          <span className="material-symbols-outlined">close</span>
        </button>
        <h2 id="refineAppTitle" className="modal-title">Refine Your App</h2>
        
        <form onSubmit={handleSubmit}>
          <div className="modal-section">
            <label htmlFor="refinementInstructions" className="input-label" style={{marginBottom: '0.5rem', fontSize: '1rem'}}>
              Describe the changes you'd like to make to the current app:
            </label>
            <textarea
              id="refinementInstructions"
              className="modal-textarea"
              value={refinementText}
              onChange={(e) => setRefinementText(e.target.value)}
              rows={6}
              placeholder="e.g., 'Change the color scheme to dark mode', 'Add a scoring system', 'Make the questions multiple choice instead of true/false'"
              required
              aria-required="true"
            />
          </div>

          <div className="modal-actions">
            <button type="button" className="button-secondary" onClick={onClose}>
              Cancel
            </button>
            <button type="submit" className="button-primary">
              Submit Refinement
            </button>
          </div>
        </form>
      </div>
    </div>
  );
};

export default RefineAppModal;
