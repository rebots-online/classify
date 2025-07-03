/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
*/
/* tslint:disable */

// Example type import removed as it's no longer used here or globally after example removal.
import {type Dispatch, type SetStateAction, createContext} from 'react';

export interface Data {
  // examples: Example[]; // Removed
  // setExamples: Dispatch<SetStateAction<Example[]>>; // Removed
  // defaultExample: Example; // Removed
  // isLoading: boolean; // Removed
  // Add other non-example related global state here if needed in the future
}

export const DataContext = createContext<Data | null>(null); // Can be null if no global state is needed after removing examples.
// Or provide an empty object with a defined type if some structure is always expected, even if empty:
// export const DataContext = createContext<Data>({} as Data);
