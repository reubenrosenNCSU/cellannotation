import { Annotation } from './Annotation';
export interface FormatAdapter<A extends Annotation, T extends unknown> {
    parse(serialized: T): ParseResult<A>;
    parseAll?(serialized: unknown[]): {
        parsed: A[];
        failed: T[];
    };
    serialize(core: A): T;
}
export interface ParseResult<A extends Annotation> {
    parsed?: A;
    error?: Error;
}
export declare const serializeAll: <A extends Annotation, T extends unknown>(adapter: FormatAdapter<A, T>) => (annotations: A[]) => T[];
export declare const parseAll: <A extends Annotation, T extends unknown>(adapter: FormatAdapter<A, T>) => (serialized: T[]) => {
    parsed: A[];
    failed: T[];
};
//# sourceMappingURL=FormatAdapter.d.ts.map