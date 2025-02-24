import { Annotation, UndoStack } from '@annotorious/core';
export declare const isMac: boolean;
export declare const initKeyboardCommands: <T extends Annotation>(undoStack: UndoStack<T>, container?: Element) => {
    destroy: () => void;
};
