import { Unsubscribe } from 'nanoevents';
import { Annotation } from '../model';
import { Store } from './Store';
import { ChangeSet } from './StoreObserver';
export interface UndoStack<T extends Annotation> {
    canRedo(): boolean;
    canUndo(): boolean;
    destroy(): void;
    getHistory(): History<T>;
    on<E extends keyof UndoStackEvents<T>>(event: E, callback: UndoStackEvents<T>[E]): Unsubscribe;
    undo(): void;
    redo(): void;
}
export interface UndoStackEvents<T extends Annotation> {
    redo(change: ChangeSet<T>): void;
    undo(change: ChangeSet<T>): void;
}
export interface History<T extends Annotation> {
    changes: ChangeSet<T>[];
    pointer: number;
}
export declare const createUndoStack: <T extends Annotation>(store: Store<T>, history?: History<T>) => UndoStack<T>;
//# sourceMappingURL=UndoStack.d.ts.map