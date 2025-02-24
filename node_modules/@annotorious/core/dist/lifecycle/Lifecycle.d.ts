import { Annotation, AnnotatorState, FormatAdapter } from '../model';
import { UndoStack } from '../state';
import { LifecycleEvents } from './LifecycleEvents';
export type Lifecycle<I extends Annotation, E extends unknown> = ReturnType<typeof createLifecycleObserver<I, E>>;
export declare const createLifecycleObserver: <I extends Annotation, E extends unknown>(state: AnnotatorState<I, E>, undoStack: UndoStack<I>, adapter?: FormatAdapter<I, E>, autoSave?: boolean) => {
    on: <T extends keyof LifecycleEvents>(event: T, callback: LifecycleEvents<E>[T]) => void;
    off: <T extends keyof LifecycleEvents<E>>(event: T, callback: LifecycleEvents<E>[T]) => void;
    emit: (event: keyof LifecycleEvents<E>, arg0: I | I[], arg1?: I | PointerEvent) => void;
};
//# sourceMappingURL=Lifecycle.d.ts.map