/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
*/
/* tslint:disable */

import React, { useState, useEffect, useRef } from 'react';

interface HeaderProps {
  siteTitle: string;
  subTitle: string;
  showLlmLogPanel: boolean;
  onToggleLlmLogPanel: () => void;
  onToggleHowToUse: () => void; // New prop
}

const Header: React.FC<HeaderProps> = ({ siteTitle, subTitle, showLlmLogPanel, onToggleLlmLogPanel, onToggleHowToUse }) => {
  const [isMobileMenuOpen, setIsMobileMenuOpen] = useState(false);
  const menuRef = useRef<HTMLDivElement>(null);
  const buttonRef = useRef<HTMLButtonElement>(null);

  const menuItems = [
    { name: 'About RobinsAI.World', href: 'javascript:void(0);' },
    { name: 'Learn AI', href: 'javascript:void(0);' },
    { name: 'AI Tools', href: 'javascript:void(0);' },
    // "How to Use" is now handled by a dedicated prop
  ];

  const toggleMobileMenu = () => {
    setIsMobileMenuOpen(!isMobileMenuOpen);
  };

  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (isMobileMenuOpen &&
          menuRef.current && 
          !menuRef.current.contains(event.target as Node) &&
          buttonRef.current &&
          !buttonRef.current.contains(event.target as Node)) {
        setIsMobileMenuOpen(false);
      }
    };

    document.addEventListener('mousedown', handleClickOutside);
    return () => {
      document.removeEventListener('mousedown', handleClickOutside);
    };
  }, [isMobileMenuOpen]);

  return (
    <>
      <header className={`app-header ${isMobileMenuOpen ? 'nav-open' : ''}`}>
        <div className="header-content">
          <div className="title-container">
            <h1 className="headline header-headline">{siteTitle}</h1>
            <p className="subtitle header-subtitle">{subTitle}</p>
          </div>
          <div className="header-actions">
            <button
              onClick={onToggleLlmLogPanel}
              className="llm-log-toggle button-secondary"
              title={showLlmLogPanel ? "Hide LLM Log" : "Show LLM Log"}
              aria-pressed={showLlmLogPanel}
            >
              <span className="material-symbols-outlined">
                {showLlmLogPanel ? 'terminal_off' : 'terminal'}
              </span>
              <span className="button-text-desktop">LLM Log</span>
            </button>
            <nav className="desktop-nav">
              <ul>
                {menuItems.map((item) => (
                  <li key={item.name}><a href={item.href}>{item.name}</a></li>
                ))}
                <li><button onClick={onToggleHowToUse} className="nav-button">How to Use</button></li>
              </ul>
            </nav>
            <button
              ref={buttonRef}
              className="hamburger-button"
              onClick={toggleMobileMenu}
              aria-label="Toggle menu"
              aria-expanded={isMobileMenuOpen}
              aria-controls="mobile-menu"
            >
              <span className="hamburger-line"></span>
              <span className="hamburger-line"></span>
              <span className="hamburger-line"></span>
            </button>
          </div>
        </div>
        {isMobileMenuOpen && (
          <nav id="mobile-menu" className="mobile-nav" ref={menuRef}>
            <ul>
              {menuItems.map((item) => (
                <li key={item.name}><a href={item.href} onClick={() => setIsMobileMenuOpen(false)}>{item.name}</a></li>
              ))}
               <li><button onClick={() => { onToggleHowToUse(); setIsMobileMenuOpen(false); }} className="nav-button-mobile">How to Use</button></li>
            </ul>
             <li className="mobile-nav-llm-toggle">
                <button
                onClick={() => { onToggleLlmLogPanel(); setIsMobileMenuOpen(false); }}
                className="llm-log-toggle-mobile"
                aria-pressed={showLlmLogPanel}
                >
                <span className="material-symbols-outlined">
                    {showLlmLogPanel ? 'terminal_off' : 'terminal'}
                </span>
                {showLlmLogPanel ? "Hide LLM Log" : "Show LLM Log"}
                </button>
            </li>
          </nav>
        )}
      </header>
      <style>{`
        .app-header {
          background-color: var(--color-background-header, light-dark(#fff, #2c2c30));
          border-bottom: 1px solid light-dark(#e0e0e0, #3a3a3f);
          padding: 0.5rem 1rem; /* Reduced padding */
          position: fixed;
          top: 0;
          left: 0;
          width: 100%;
          z-index: 1000;
          box-sizing: border-box;
          height: var(--header-height);
        }
        .header-content {
          display: flex;
          align-items: center;
          justify-content: space-between;
          max-width: 1400px;
          margin: 0 auto;
          height: 100%;
        }
        .title-container {
          display: flex;
          flex-direction: column;
          justify-content: center; /* Center titles vertically */
           flex-grow: 1;
           margin-left: 0.5rem; /* Add some space if logo is removed */
        }
        .headline.header-headline {
          font-size: 1.5rem; /* Adjusted font size */
          color: var(--color-headline);
          font-family: var(--font-display);
          text-transform: uppercase;
          margin: 0;
          line-height: 1.1;
        }
        .subtitle.header-subtitle {
          font-size: 0.75rem; /* Adjusted font size */
          color: var(--color-subtitle);
          margin: 0;
          line-height: 1.2;
        }
        
        .header-actions {
          display: flex;
          align-items: center;
          gap: 0.5rem; /* Gap between LLM toggle and nav/hamburger */
        }

        .llm-log-toggle {
          display: flex;
          align-items: center;
          gap: 0.3rem;
          padding: 0.3rem 0.6rem;
          font-size: 0.8rem;
        }
        .llm-log-toggle .material-symbols-outlined {
          font-size: 1.2rem; /* Icon size */
        }
         .button-text-desktop {
            display: none; /* Hide text on smaller screens initially */
         }

        .desktop-nav {
          display: none;
        }
        .desktop-nav ul {
          list-style: none;
          padding: 0;
          margin: 0;
          display: flex;
          align-items: center; /* Align button with text links */
          gap: 1rem; /* Spacing between menu items */
        }
        .desktop-nav a, .desktop-nav .nav-button {
          text-decoration: none;
          color: var(--color-text);
          font-weight: 500;
          font-size: 0.9rem;
          padding: 0.5rem;
          border-radius: 4px;
          transition: background-color 0.2s, color 0.2s;
          background: none;
          border: none;
          cursor: pointer;
          font-family: inherit;
        }
        .desktop-nav a:hover, .desktop-nav .nav-button:hover {
          background-color: var(--color-accent);
          color: light-dark(#fff, #fff);
        }

        .hamburger-button {
          display: flex; /* Changed to flex for mobile */
          flex-direction: column;
          justify-content: space-around;
          width: 30px;
          height: 30px;
          background: transparent;
          border: none;
          cursor: pointer;
          padding: 0;
          z-index: 1001; /* Above mobile nav initially if needed */
        }
        .hamburger-line {
          display: block;
          width: 100%;
          height: 3px;
          background-color: var(--color-text);
          border-radius: 3px;
          transition: all 0.3s ease-in-out;
        }
        .app-header.nav-open .hamburger-line:nth-child(1) {
          transform: translateY(8px) rotate(45deg);
        }
        .app-header.nav-open .hamburger-line:nth-child(2) {
          opacity: 0;
        }
        .app-header.nav-open .hamburger-line:nth-child(3) {
          transform: translateY(-8px) rotate(-45deg);
        }

        .mobile-nav {
          display: flex; /* Initially flex, controlled by parent state */
          flex-direction: column;
          position: absolute;
          top: var(--header-height);
          right: 0;
          width: 250px;
          background-color: var(--color-background-header, light-dark(#fff, #2c2c30));
          box-shadow: -2px 0 5px rgba(0,0,0,0.1);
          padding: 1rem;
          border-top: 1px solid light-dark(#e0e0e0, #3a3a3f);
          max-height: calc(100vh - var(--header-height));
          overflow-y: auto;
        }
        .mobile-nav ul {
          list-style: none;
          padding: 0;
          margin: 0;
        }
        .mobile-nav li a, .mobile-nav li .nav-button-mobile {
          display: block;
          width: 100%;
          text-align: left;
          padding: 0.75rem 1rem;
          text-decoration: none;
          color: var(--color-text);
          font-size: 1rem;
          border-bottom: 1px solid light-dark(#eee, #444);
          transition: background-color 0.2s;
          background: none;
          border-left: none; border-right: none; border-top: none; 
          cursor: pointer;
          font-family: inherit;
          border-radius: 0;
        }
        .mobile-nav li:last-child a, .mobile-nav li:last-child .nav-button-mobile {
          border-bottom: none;
        }
        .mobile-nav li a:hover, .mobile-nav li .nav-button-mobile:hover {
          background-color: var(--color-accent);
          color: #fff;
        }
        
        .mobile-nav-llm-toggle {
            margin-top: 0.5rem;
            padding-top: 0.5rem;
            border-top: 1px solid light-dark(#eee, #444);
        }
        .llm-log-toggle-mobile {
            display: flex;
            align-items: center;
            gap: 0.5rem;
            width: 100%;
            padding: 0.75rem 1rem;
            background: none;
            border: none;
            color: var(--color-text);
            font-size: 1rem;
            text-align: left;
            cursor: pointer;
        }
        .llm-log-toggle-mobile:hover {
            background-color: var(--color-accent);
            color: #fff;
        }
        .llm-log-toggle-mobile .material-symbols-outlined {
             font-size: 1.5rem;
        }


        @media (min-width: 900px) { /* Breakpoint for desktop menu */
          .hamburger-button {
            display: none;
          }
          .desktop-nav {
            display: block;
          }
          .title-container {
            flex-grow: 0; /* Don't let title grow too much on desktop */
            margin-right: auto; /* Pushes nav to the right */
            /* margin-left: 1rem; */ /* Removed as logo is gone */
          }
           .headline.header-headline {
             font-size: 1.8rem;
           }
           .subtitle.header-subtitle {
             font-size: 0.8rem;
           }
           .llm-log-toggle .button-text-desktop {
             display: inline; /* Show text on desktop */
           }
        }
         @media (max-width: 480px) {
            .headline.header-headline {
                font-size: 1.1rem;
            }
            .subtitle.header-subtitle {
                font-size: 0.65rem;
                white-space: nowrap;
                overflow: hidden;
                text-overflow: ellipsis;
                max-width: 150px; /* Adjust as needed */
            }
             .app-header {
                padding: 0.5rem 0.75rem;
             }
        }
        
        /* Material Symbols Outlined font for icons */
        .material-symbols-outlined {
          font-family: 'Google Symbols';
          font-weight: normal;
          font-style: normal;
          font-size: 24px;
          line-height: 1;
          letter-spacing: normal;
          text-transform: none;
          display: inline-block;
          white-space: nowrap;
          word-wrap: normal;
          direction: ltr;
          -webkit-font-feature-settings: 'liga';
          -webkit-font-smoothing: antialiased;
        }
      `}</style>
    </>
  );
};

export default Header;