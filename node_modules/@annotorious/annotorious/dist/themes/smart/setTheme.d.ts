import { Theme } from '../../AnnotoriousOpts';
export declare const sampleBrightness: (imageOrCanvas: HTMLElement) => number;
export declare const detectTheme: (imageOrCanvas: HTMLElement) => "dark" | "light";
export declare const setTheme: (imageOrCanvas: HTMLElement, container: HTMLElement, theme: Theme) => void;
