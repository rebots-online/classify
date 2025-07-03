/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
*/
/* tslint:disable */

import React from 'react';

interface FooterProps {
  attributionText: string;
}

const Footer: React.FC<FooterProps> = ({ attributionText }) => {
  const currentYear = new Date().getFullYear();
  return (
    <>
      <footer className="app-footer">
        
        <p 
          className="attribution-footer"
          dangerouslySetInnerHTML={{ __html: attributionText }}
        />
        <p className="copyright-footer">
          &copy; {currentYear} Robin L. M. Cheung, MBA. All Rights Reserved.
        </p>
      </footer>
      <style>{`
        .app-footer {
          background-color: var(--color-background-footer, light-dark(#f0f0f0, #1e1e20));
          border-top: 1px solid light-dark(#e0e0e0, #3a3a3f);
          padding: 1rem;
          text-align: center;
          color: var(--color-text);
        }
        .attribution-footer {
          color: var(--color-attribution);
          font-family: var(--font-secondary);
          font-size: 0.9rem;
          font-style: italic;
          margin-bottom: 0.25rem;
          margin-top: 0;
        }
        .attribution-footer strong { 
          color: var(--color-text); /* Make name more prominent */
        }
        .copyright-footer {
          font-size: 0.8rem;
          color: var(--color-subtitle);
           margin-bottom: 0;
        }

         @media (max-width: 768px) {
            .attribution-footer {
                font-size: 0.8rem;
            }
            .copyright-footer {
                font-size: 0.7rem;
            }
        }
      `}</style>
    </>
  );
};

export default Footer;