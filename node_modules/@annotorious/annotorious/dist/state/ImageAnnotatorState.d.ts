import { ImageAnnotation } from '../model';
import { AnnotoriousOpts } from '../AnnotoriousOpts';
import { Annotation, AnnotatorState, HoverState, SelectionState } from '@annotorious/core';
import { ImageAnnotationStore, SvelteImageAnnotatorState } from './ImageAnnotationStore';
export type ImageAnnotatorState<I extends Annotation = ImageAnnotation, E extends unknown = ImageAnnotation> = AnnotatorState<I, E> & {
    store: ImageAnnotationStore<I>;
    selection: SelectionState<I, E>;
    hover: HoverState<I>;
};
export declare const createImageAnnotatorState: <I extends Annotation, E extends unknown>(opts: AnnotoriousOpts<I, E>) => ImageAnnotatorState<I, E>;
export declare const createSvelteImageAnnotatorState: <I extends Annotation, E extends unknown>(opts: AnnotoriousOpts<I, E>) => SvelteImageAnnotatorState<I, E>;
