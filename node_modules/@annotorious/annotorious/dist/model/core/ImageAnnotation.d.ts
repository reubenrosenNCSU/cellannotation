import { Annotation, AnnotationTarget } from '@annotorious/core';
import { Shape } from './Shape';
export interface ImageAnnotation extends Annotation {
    target: ImageAnnotationTarget;
}
export interface ImageAnnotationTarget extends AnnotationTarget {
    selector: Shape;
}
export declare const isImageAnnotation: <T extends Annotation>(a: T | ImageAnnotation) => a is ImageAnnotation;
export declare const isImageAnnotationTarget: <T extends AnnotationTarget>(t: T | ImageAnnotationTarget) => t is ImageAnnotationTarget;
