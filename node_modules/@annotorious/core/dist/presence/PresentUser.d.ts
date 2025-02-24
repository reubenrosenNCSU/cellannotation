import { User } from '../model/User';
import { Appearance } from './Appearance';
export interface PresentUser extends User {
    presenceKey: string;
    appearance: Appearance;
}
//# sourceMappingURL=PresentUser.d.ts.map