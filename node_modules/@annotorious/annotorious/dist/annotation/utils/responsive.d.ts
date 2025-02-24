export declare const enableResponsive: (image: HTMLImageElement | HTMLCanvasElement, svg: SVGSVGElement) => {
    destroy: () => void;
    subscribe: (this: void, run: import('svelte/store').Subscriber<number>, invalidate?: import('svelte/store').Invalidator<number> | undefined) => import('svelte/store').Unsubscriber;
};
