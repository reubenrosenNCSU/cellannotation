import { Annotation, AnnotationBody } from '../model/Annotation';
import { User } from '../model/User';
/**
 * Returns all users listed as creators or updaters in any parts of this
 * annotation.
 */
export declare const getContributors: (annotation: Annotation) => User[];
/**
 * Converts any string dates in the given annotation(-like)
 * object to proper Date objects.
 */
export declare const reviveDates: <A extends Annotation = Annotation>(annotation: any) => A;
/**
 * Shorthand/helper.
 */
export declare const createBody: (annotationOrId: string | Annotation, payload: {
    [key: string]: any;
}, created?: Date, creator?: User) => AnnotationBody;
//# sourceMappingURL=annotationUtils.d.ts.map