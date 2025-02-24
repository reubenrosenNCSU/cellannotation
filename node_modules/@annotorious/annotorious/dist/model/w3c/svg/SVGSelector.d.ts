import { Shape } from '../../core';
export interface SVGSelector {
    type: 'SvgSelector';
    value: string;
}
export declare const parseSVGSelector: <T extends Shape>(valueOrSelector: SVGSelector | string) => T;
export declare const serializeSVGSelector: (shape: Shape) => SVGSelector;
