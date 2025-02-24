import { PresentUser } from './PresentUser';
export interface PresenceEvents {
    presence: (users: PresentUser[]) => void;
    selectionChange: (from: PresentUser, selection: string[] | null) => void;
}
//# sourceMappingURL=PresenceEvents.d.ts.map