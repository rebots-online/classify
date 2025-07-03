/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
*/
/* tslint:disable */

import App from '@/App';
import {DataContext} from '@/context';
import React from 'react';
import ReactDOM from 'react-dom/client';

// Import uuid
import 'https://esm.sh/uuid@^9.0.1';


function DataProvider({children}) {
  // Examples and related loading state are removed.
  // The app will no longer fetch or use examples.json.

  const emptyContextPlaceholder = {title: '', url: '', spec: '', code: ''}; // Retain structure if needed elsewhere, but not for examples

  const value = {
    // examples: [], // Removed
    // isLoading: false, // Removed
    // setExamples: () => {}, // Removed
    // defaultExample: emptyContextPlaceholder, // Removed, app will not preseed with an example
  };

  return <DataContext.Provider value={value}>{children}</DataContext.Provider>;
}

const rootElement = document.getElementById('root');
if (rootElement) {
  const root = ReactDOM.createRoot(rootElement);
  root.render(
    <DataProvider>
      <App />
    </DataProvider>,
  );
} else {
  console.error("Failed to find the root element with ID 'root'.");
}