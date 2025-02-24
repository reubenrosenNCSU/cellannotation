import { FormatAdapter, ParseResult, W3CAnnotation } from '@annotorious/core';
import { ImageAnnotation } from '../core';
import { W3CImageAnnotation } from './W3CImageAnnotation';
export type W3CImageFormatAdapter = FormatAdapter<ImageAnnotation, W3CImageAnnotation>;
export interface W3CImageFormatAdapterOpts {
    strict: boolean;
    invertY: boolean;
}
export declare const W3CImageFormat: (source: string, opts?: W3CImageFormatAdapterOpts) => W3CImageFormatAdapter;
export declare const parseW3CImageAnnotation: (annotation: W3CAnnotation, opts?: W3CImageFormatAdapterOpts) => ParseResult<ImageAnnotation>;
export declare const serializeW3CImageAnnotation: (annotation: ImageAnnotation, source: string, opts?: W3CImageFormatAdapterOpts) => W3CImageAnnotation;
