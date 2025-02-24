import { User } from '../model/User';
import { Appearance } from './Appearance';
import { PresentUser } from './PresentUser';
export interface AppearanceProvider {
    addUser(presenceKey: string, user: User): Appearance;
    removeUser(user: PresentUser): void;
}
export declare const defaultColorProvider: () => {
    assignRandomColor: () => string;
    releaseColor: (color: string) => number;
};
export declare const createDefaultAppearanceProvider: () => {
    addUser: (presenceKey: string, user: User) => Appearance;
    removeUser: (user: PresentUser) => number;
};
//# sourceMappingURL=AppearanceProvider.d.ts.map