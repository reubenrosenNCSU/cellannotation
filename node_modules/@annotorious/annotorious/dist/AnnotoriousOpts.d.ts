import { Annotation, DrawingStyleExpression, FormatAdapter, History, UserSelectActionExpression } from '@annotorious/core';
import { ImageAnnotation } from './model';
export interface AnnotoriousOpts<I extends Annotation = ImageAnnotation, E extends unknown = ImageAnnotation> {
    adapter?: FormatAdapter<I, E>;
    autoSave?: boolean;
    drawingEnabled?: boolean;
    drawingMode?: DrawingMode;
    initialHistory?: History<I>;
    modalSelect?: boolean;
    userSelectAction?: UserSelectActionExpression<E>;
    style?: DrawingStyleExpression<ImageAnnotation>;
    theme?: Theme;
}
export type DrawingMode = 'click' | 'drag';
export type Theme = 'dark' | 'light' | 'auto';
export declare const fillDefaults: <I extends Annotation = ImageAnnotation, E extends unknown = ImageAnnotation>(opts: AnnotoriousOpts<I, E>, defaults: AnnotoriousOpts<I, E>) => AnnotoriousOpts<I, E>;
