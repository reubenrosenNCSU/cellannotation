import { Annotation } from './Annotation';
import { User } from './User';
import { PresenceProvider } from '../presence';
import { History, HoverState, SelectionState, Store, UndoStack, UserSelectActionExpression, ViewportState } from '../state';
import { LifecycleEvents } from '../lifecycle';
import { FormatAdapter } from './FormatAdapter';
import { DrawingStyleExpression } from './DrawingStyle';
import { Filter } from './Filter';
/**
 * Base annotator interface
 * @template I - internal core data model
 * @template E - external adapted representation
 */
export interface Annotator<I extends Annotation = Annotation, E extends unknown = Annotation> {
    addAnnotation(annotation: Partial<E>): void;
    cancelSelected(): void;
    canRedo(): boolean;
    canUndo(): boolean;
    clearAnnotations(): void;
    destroy(): void;
    getAnnotationById(id: string): E | undefined;
    getAnnotations(): E[];
    getHistory(): History<I>;
    getSelected(): E[];
    getUser(): User;
    loadAnnotations(url: string, replace?: boolean): Promise<E[]>;
    redo(): void;
    removeAnnotation(arg: Partial<E> | string): E | undefined;
    setAnnotations(annotations: Partial<E>[], replace?: boolean): void;
    setFilter(filter: Filter<I> | undefined): void;
    setPresenceProvider?(provider: PresenceProvider): void;
    setSelected(arg?: string | string[], editable?: boolean): void;
    setStyle(style: DrawingStyleExpression<I> | undefined): void;
    setUser(user: User): void;
    setUserSelectAction(action: UserSelectActionExpression<E>): void;
    setVisible(visible: boolean): void;
    undo(): void;
    updateAnnotation(annotation: Partial<E>): E;
    on<T extends keyof LifecycleEvents<E>>(event: T, callback: LifecycleEvents<E>[T]): void;
    off<T extends keyof LifecycleEvents<E>>(event: T, callback: LifecycleEvents<E>[T]): void;
    state: AnnotatorState<I, E>;
}
export interface AnnotatorState<I extends Annotation, E extends unknown> {
    store: Store<I>;
    selection: SelectionState<I, E>;
    hover: HoverState<I>;
    viewport: ViewportState;
}
export declare const createBaseAnnotator: <I extends Annotation, E extends unknown>(state: AnnotatorState<I, E>, undoStack: UndoStack<I>, adapter?: FormatAdapter<I, E>) => {
    addAnnotation: (annotation: E) => void;
    cancelSelected: () => void;
    canRedo: () => boolean;
    canUndo: () => boolean;
    clearAnnotations: () => void;
    getAnnotationById: (id: string) => E | undefined;
    getAnnotations: () => E[];
    getHistory: () => History<I>;
    getSelected: () => E[];
    loadAnnotations: (url: string, replace?: boolean) => Promise<any>;
    redo: () => void;
    removeAnnotation: (arg: E | string) => E | undefined;
    setAnnotations: (annotations: E[], replace?: boolean) => void;
    setSelected: (arg?: string | string[], editable?: boolean) => void;
    setUserSelectAction: (action: UserSelectActionExpression<E>) => void;
    undo: () => void;
    updateAnnotation: (updated: E) => E;
};
//# sourceMappingURL=Annotator.d.ts.map