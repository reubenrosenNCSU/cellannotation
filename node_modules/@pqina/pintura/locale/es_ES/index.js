import Annotate from './annotate/index.js';
import Core, { MarkupEditor } from './core/index.js';
import Crop from './crop/index.js';
import Decorate from './decorate/index.js';
import Filter from './filter/index.js';
import Finetune from './finetune/index.js';
import Frame from './frame/index.js';
import Resize from './resize/index.js';
import Redact from './redact/index.js';
import Sticker from './sticker/index.js';
import Trim from './trim/index.js';
import Fill from './fill/index.js';

export const LocaleAnnotate = Annotate;
export const LocaleCore = Core;
export const LocaleCrop = Crop;
export const LocaleDecorate = Decorate;
export const LocaleFilter = Filter;
export const LocaleFinetune = Finetune;
export const LocaleFrame = Frame;
export const LocaleResize = Resize;
export const LocaleRedact = Redact;
export const LocaleSticker = Sticker;
export const LocaleTrim = Trim;
export const LocaleFill = Fill;
export const LocaleMarkupEditor = MarkupEditor;

export default {
    ...Core,
    ...MarkupEditor,
    ...Annotate,
    ...Crop,
    ...Decorate,
    ...Filter,
    ...Finetune,
    ...Frame,
    ...Resize,
    ...Redact,
    ...Sticker,
    ...Fill,
};
