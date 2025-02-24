import { PresenceEvents } from './PresenceEvents';
export interface PresenceProvider {
    on<E extends keyof PresenceEvents>(event: E, callback: PresenceEvents[E]): void;
}
//# sourceMappingURL=PresenceProvider.d.ts.map