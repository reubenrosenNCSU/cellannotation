const IconCross =
    '<g fill="none" fill-rule="evenodd" stroke-linecap="round" stroke-linejoin="round" stroke="currentColor" stroke-width=".125em"><path d="M18 6L6 18M6 6l12 12"></path></path></g>';

const CharacterA =
    '<path fill="none" d="M9 15 L12 9 L15 15 M10 13.5 h3" stroke="currentColor" stroke-width=".125em"/>';

export default {
    // generic
    labelReset: 'रीसेट',
    labelDefault: 'डिफ़ॉल्ट',
    labelAuto: 'ऑटो',
    labelNone: 'कोई नहीं',
    labelEdit: 'संपादित करें',
    labelClose: 'बंद करें',
    labelSupportError: `इस ब्राउज़र पर {features} समर्थित नहीं हैं`,

    // defaults
    labelColor: 'रंग',
    labelWidth: 'चौड़ाई',
    labelSize: 'आकार',
    labelOffset: 'ऑफ़सेट',
    labelAmount: 'मात्रा',
    labelInset: 'इनसेट',
    labelRadius: 'त्रिज्या',

    // controls
    labelColorPalette: 'रंग पैलेट',

    // sizes
    labelSizeExtraSmall: 'बहुत छोटा',
    labelSizeSmall: 'छोटा',
    labelSizeMediumSmall: 'मध्यम छोटा',
    labelSizeMedium: 'मध्यम',
    labelSizeMediumLarge: 'मध्यम बड़ा',
    labelSizeLarge: 'बड़ा',
    labelSizeExtraLarge: 'बहुत बड़ा',

    // default buttons

    labelButtonCancel: 'रद्द करें',
    labelButtonUndo: 'पूर्ववत करें',
    labelButtonRedo: 'पुन: करें',
    labelButtonRevert: 'पूर्व स्थिति में लौटें',
    labelButtonExport: 'पूर्ण',

    // zoom
    labelZoom: 'ज़ूम',
    labelZoomIn: 'ज़ूम इन',
    labelZoomOut: 'ज़ूम आउट',
    labelZoomFit: 'दृश्य को फिट करें',
    labelZoomActual: 'वास्तविक आकार',
    iconZoomIn: '<path stroke="currentColor" stroke-width=".125em" d="M8 12 h8 M12 8 v8" />',
    iconZoomOut: '<path stroke="currentColor" stroke-width=".125em" d="M9 12 h6" />',

    // icons
    iconSupportError: `<g fill="none" stroke="currentColor" stroke-width="2"><path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/><g><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></g>`,
    iconButtonClose: IconCross,
    iconButtonRevert: `<g fill="none" fill-rule="evenodd" stroke-linecap="round" stroke-linejoin="round" stroke="currentColor" stroke-width=".125em"><path d="M7.388 18.538a8 8 0 10-2.992-9.03"/><path fill="currentColor" d="M2.794 11.696L2.37 6.714l5.088 3.18z"/><path d="M12 8v4M12 12l4 2"/></g>`,
    iconButtonUndo: `<g fill="none" fill-rule="evenodd" stroke-linecap="round" stroke-linejoin="round" stroke="currentColor" stroke-width=".125em"><path d="M10 8h4c2.485 0 5 2 5 5s-2.515 5-5 5h-4"/><path fill="currentColor" d="M5 8l4-3v6z"/></g>`,
    iconButtonRedo: `<g fill="none" fill-rule="evenodd" stroke-linecap="round" stroke-linejoin="round" stroke="currentColor" stroke-width=".125em"><path d="M14 8h-4c-2.485 0-5 2-5 5s2.515 5 5 5h4"/><path fill="currentColor" d="M19 8l-4-3v6z"/></g>`,
    iconButtonExport: `<polyline points="20 6 9 17 4 12" fill="none" stroke="currentColor" stroke-width=".125em"></polyline>`,

    // status
    statusLabelButtonClose: 'बंद करें',
    statusIconButtonClose: IconCross,

    // new status options
    statusImageLoadNone: 'छवि के लिए प्रतीक्षा कर रहा है',
    statusImageLoadMinSize: 'न्यूनतम छवि आकार है {minWidth} &times; {minHeight}',
    statusImageLoadError: 'छवि लोड करने में त्रुटि',
    statusImageLoadPrepare: 'छवि को तैयार कर रहा है&hellip;',
    statusImageLoadBusy: 'छवि लोड हो रही है&hellip;',
    statusImageUploadError: 'छवि अपलोड करने में त्रुटि',
    statusImageUploadBusy: 'छवि अपलोड हो रही है&hellip;',
    statusImageProcessError: 'छवि को प्रोसेस करने में त्रुटि',
    statusImageProcessBusy: 'छवि को प्रोसेस किया जा रहा है&hellip;',
};

export const MarkupEditor = {
    shapeLabelButtonSelectSticker: 'छवि का चयन करें',
    shapeIconButtonSelectSticker: `<g fill="none" stroke="currentColor" stroke-width="0.0625em"><path d="M8 21 L15 11 L19 15"/><path d="M15 2 v5 h5"/><path d="M8 2 h8 l4 4 v12 q0 4 -4 4 h-8 q-4 0 -4 -4 v-12 q0 -4 4 -4z"/></g><circle fill="currentColor" cx="10" cy="8" r="1.5"/>`,

    shapeIconButtonFlipHorizontal: `<g stroke="currentColor" stroke-width=".125em"><path fill="none" d="M6 6.5h5v11H6z"/><path fill="currentColor" d="M15 6.5h3v11h-3z"/><path d="M11 4v16" fill="currentColor"/></g>`,
    shapeIconButtonFlipVertical: `<g stroke="currentColor" stroke-width=".125em"><rect x="7" y="8" width="11" height="5" fill="none"/><rect x="7" y="17" width="11" height="2" fill="currentColor"/><line x1="5" y1="13" x2="20" y2="13"/></g>`,
    shapeIconButtonRemove: `<g fill="none" fill-rule="evenodd"><path stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" d="M7.5 7h9z"/><path d="M7.916 9h8.168a1 1 0 01.99 1.14l-.972 6.862a2 2 0 01-1.473 1.653c-.877.23-1.753.345-2.629.345-.876 0-1.752-.115-2.628-.345a2 2 0 01-1.473-1.653l-.973-6.862A1 1 0 017.916 9z" fill="currentColor"/><rect fill="currentColor" x="10" y="5" width="4" height="3" rx="1"/></g>`,
    shapeIconButtonDuplicate: `<g fill="none" fill-rule="evenodd"><path d="M15 13.994V16a2 2 0 01-2 2H8a2 2 0 01-2-2v-5a2 2 0 012-2h2.142" stroke="currentColor" stroke-width=".125em"/><path d="M15 9V8a1 1 0 00-2 0v1h-1a1 1 0 000 2h1v1a1 1 0 002 0v-1h1a1 1 0 000-2h-1zm-4-4h6a2 2 0 012 2v6a2 2 0 01-2 2h-6a2 2 0 01-2-2V7a2 2 0 012-2z" fill="currentColor"/></g>`,
    shapeIconButtonMoveToFront: `<g fill="none" fill-rule="evenodd"><rect fill="currentColor" x="11" y="13" width="8" height="2" rx="1"/><rect fill="currentColor" x="9" y="17" width="10" height="2" rx="1"/><path d="M11.364 8H10a5 5 0 000 10M12 6.5L14.5 8 12 9.5z" stroke="currentColor" stroke-width=".125em" stroke-linecap="round"/></g>`,

    shapeIconButtonTextLayoutAutoWidth: `${CharacterA}`,
    shapeIconButtonTextLayoutAutoHeight: `<g fill="currentColor"><circle cx="4" cy="12" r="1.5"/><circle cx="20" cy="12" r="1.5"/></g>${CharacterA}`,
    shapeIconButtonTextLayoutFixedSize: `<g fill="currentColor"><circle cx="5" cy="6" r="1.5"/><circle cx="19" cy="6" r="1.5"/><circle cx="19" cy="19" r="1.5"/><circle cx="5" cy="19" r="1.5"/></g>${CharacterA}`,

    shapeTitleButtonTextLayoutAutoWidth: 'ऑटो चौड़ाई',
    shapeTitleButtonTextLayoutAutoHeight: 'ऑटो ऊंचाई',
    shapeTitleButtonTextLayoutFixedSize: 'निर्दिष्ट आकार',

    shapeTitleButtonFlipHorizontal: 'हॉरिज़ॉन्टल फ्लिप',
    shapeTitleButtonFlipVertical: 'वर्टिकल फ्लिप',
    shapeTitleButtonRemove: 'हटाएँ',
    shapeTitleButtonDuplicate: 'कॉपी करें',
    shapeTitleButtonMoveToFront: 'आगे ले जाएं',

    shapeLabelInputText: 'टेक्स्ट संपादित करें',

    shapeIconInputCancel: `<g fill="none" fill-rule="evenodd" stroke-linecap="round" stroke-linejoin="round" stroke="currentColor" stroke-width=".125em"><path d="M18 6L6 18M6 6l12 12"/></g>`,
    shapeIconInputConfirm: `<g fill="none" fill-rule="evenodd" stroke-linecap="round" stroke-linejoin="round" stroke="currentColor" stroke-width=".125em"><polyline points="20 6 9 17 4 12"/></g>`,

    shapeLabelInputCancel: 'रद्द करें',
    shapeLabelInputConfirm: 'पुष्टि करें',

    shapeLabelStrokeNone: 'कोई आउटलाइन नहीं',

    shapeLabelFontStyleNormal: 'सामान्य',
    shapeLabelFontStyleBold: 'बोल्ड',
    shapeLabelFontStyleItalic: 'इटैलिक',
    shapeLabelFontStyleItalicBold: 'बोल्ड इटैलिक',

    shapeTitleBackgroundColor: 'भरना',

    shapeTitleCornerRadius: 'कोने की त्रिज्या',

    shapeTitleFontFamily: 'फ़ॉन्ट',
    shapeTitleFontSize: 'फ़ॉन्ट आकार',
    shapeTitleFontStyle: 'फ़ॉन्ट शैली',
    shapeTitleLineHeight: 'लाइन ऊंचाई',

    shapeTitleLineStart: 'शुरू',
    shapeTitleLineEnd: 'अंत',
    shapeTitleStrokeWidth: 'रेखा चौड़ाई',
    shapeTitleStrokeColor: 'रेखा रंग',

    shapeTitleLineDecorationBar: 'लाइन',
    shapeTitleLineDecorationCircle: 'वृत्त',
    shapeTitleLineDecorationSquare: 'वर्ग',
    shapeTitleLineDecorationArrow: 'तीर',
    shapeTitleLineDecorationCircleSolid: 'भरी हुई वृत्त',
    shapeTitleLineDecorationSquareSolid: 'भरी हुई वर्ग',
    shapeTitleLineDecorationArrowSolid: 'भरी हुई तीर',

    shapeIconLineDecorationBar: `<g stroke="currentColor" stroke-linecap="round" stroke-width=".125em"><path d="M5,12 H16"/><path d="M16,8 V16"/></g>`,
    shapeIconLineDecorationCircle: `<g stroke="currentColor" stroke-linecap="round"><path stroke-width=".125em" d="M5,12 H12"/><circle fill="none" stroke-width=".125em" cx="16" cy="12" r="4"/></g>`,
    shapeIconLineDecorationSquare: `<g stroke="currentColor" stroke-linecap="round"><path stroke-width=".125em" d="M5,12 H12"/><rect fill="none" stroke-width=".125em" x="12" y="8" width="8" height="8"/></g>`,
    shapeIconLineDecorationArrow: `<g stroke="currentColor" stroke-linecap="round" stroke-width=".125em"><path d="M5,12 H16 M13,7 l6,5 l-6,5" fill="none"/></g>`,
    shapeIconLineDecorationCircleSolid: `<g stroke="currentColor" stroke-linecap="round"><path stroke-width=".125em" d="M5,12 H12"/><circle fill="currentColor" cx="16" cy="12" r="4"/></g>`,
    shapeIconLineDecorationSquareSolid: `<g stroke="currentColor" stroke-linecap="round"><path stroke-width=".125em" d="M5,12 H12"/><rect fill="currentColor" x="12" y="8" width="8" height="8"/></g>`,
    shapeIconLineDecorationArrowSolid: `<g stroke="currentColor" stroke-linecap="round" stroke-width=".125em"><path d="M5,12 H16"/><path d="M13,7 l6,5 l-6,5z" fill="currentColor"/></g>`,

    shapeTitleColorTransparent: 'पारदर्शी',
    shapeTitleColorWhite: 'सफ़ेद',
    shapeTitleColorSilver: 'चाँदी',
    shapeTitleColorGray: 'धूसर',
    shapeTitleColorBlack: 'काला',
    shapeTitleColorNavy: 'नेवी ब्लू',
    shapeTitleColorBlue: 'नीला',
    shapeTitleColorAqua: 'एक्वा',
    shapeTitleColorTeal: 'टील',
    shapeTitleColorOlive: 'जैतूनी',
    shapeTitleColorGreen: 'हरा',
    shapeTitleColorYellow: 'पीला',
    shapeTitleColorOrange: 'नारंगी',
    shapeTitleColorRed: 'लाल',
    shapeTitleColorMaroon: 'मैरून',
    shapeTitleColorFuchsia: 'फ़्यूशिया',
    shapeTitleColorPurple: 'बैंगनी',

    shapeTitleTextOutline: 'पाठ की रूपरेखा',
    shapeTitleTextOutlineWidth: 'चौड़ाई',
    shapeTitleTextShadow: 'पाठ की छाया',
    shapeTitleTextShadowBlur: 'धुंधलापन',

    shapeTitleTextColor: 'फ़ॉन्ट रंग',
    shapeTitleTextAlign: 'टेक्स्ट संरेखित करें',
    shapeTitleTextAlignLeft: 'बाएं ओर संरेखित करें',
    shapeTitleTextAlignCenter: 'मध्य संरेखित करें',
    shapeTitleTextAlignRight: 'दाएं ओर संरेखित करें',

    shapeIconTextAlignLeft: `<g stroke-width=".125em" stroke="currentColor"><line x1="5" y1="8" x2="15" y2="8"/><line x1="5" y1="12" x2="19" y2="12"/><line x1="5" y1="16" x2="14" y2="16"/></g>`,
    shapeIconTextAlignCenter: `<g stroke-width=".125em" stroke="currentColor"><line x1="7" y1="8" x2="17" y2="8"/><line x1="5" y1="12" x2="19" y2="12"/><line x1="8" y1="16" x2="16" y2="16"/></g>`,
    shapeIconTextAlignRight: `<g stroke-width=".125em" stroke="currentColor"><line x1="9" y1="8" x2="19" y2="8"/><line x1="5" y1="12" x2="19" y2="12"/><line x1="11" y1="16" x2="19" y2="16"/></g>`,

    shapeLabelToolMove: 'हटाएं',
    shapeLabelToolView: 'देखें',
    shapeLabelToolSharpie: 'शार्पी',
    shapeLabelToolEraser: 'इरेसर',
    shapeLabelToolPath: 'पथ',
    shapeLabelToolRectangle: 'आयत',
    shapeLabelToolEllipse: 'दीर्घवृत्त',
    shapeLabelToolArrow: 'तीर',
    shapeLabelToolLine: 'रेखा',
    shapeLabelToolText: 'टेक्स्ट',
    shapeLabelToolPreset: 'स्टिकर',

    shapeIconToolView: `<g stroke-width=".125em" stroke="currentColor" fill="none" stroke-linecap="round" stroke-linejoin="round"><path d="M10.98 9.703V2.567c0-1.19 1.19-1.785 1.784-1.785.595 0 1.784.595 1.784 1.785v3.568"/><path d="M14.548 9.703V4.35c0-1.19 1.19-1.784 1.784-1.784.595 0 1.784.594 1.784 1.784v2.973"/><path d="M18.116 10.244V7.271c0-1.19 1.19-1.784 1.784-1.784.595 0 1.785.595 1.785 1.784 0 1.19 0 8.92-1.19 12.488-1.19 3.569-10.704 4.758-13.678 0-2.973-4.757-2.973-4.757-4.163-6.541-1.189-1.784-1.153-2.974-.594-3.568.558-.595 1.784-1.19 2.973.594 1.277 1.916 2.07 2.907 2.379 2.974V5.487c0-1.19 1.19-1.784 1.784-1.784.595 0 1.784.595 1.784 1.784V8.46"/></g>`,
    shapeIconToolMove: `<g stroke-width=".125em" stroke="currentColor" fill="none"><path d="M6 2 L6 19 L18 13 Z M13 18 L16 24" stroke="currentColor" stroke-width=".125em" fill="none" fill-rule="evenodd" stroke-linejoin="round"/></g>`,
    shapeIconToolSharpie: `<g stroke-width=".125em" stroke="currentColor" fill="none"><path d="M2.025 5c5.616-2.732 8.833-3.857 9.65-3.374C12.903 2.351.518 12.666 2.026 14 3.534 15.334 16.536.566 17.73 2.566 18.924 4.566 3.98 17.187 4.831 18c.851.813 9.848-6 11.643-6 1.087 0-2.53 5.11-2.92 7-.086.41 3.323-1.498 4.773-1 .494.17.64 2.317 1.319 3 .439.443 1.332.776 2.679 1" stroke="currentColor" stroke-width=".125em" fill="none" fill-rule="evenodd" stroke-linejoin="round"/></g>`,
    shapeIconToolEraser: `<g stroke-width=".125em" stroke="currentColor" stroke-linecap="round" fill="none"><g transform="translate(3, 15) rotate(-45)"><rect x="0" y="0" width="18" height="10" rx="3"/></g><line x1="11" y1="21" x2="18" y2="21"/><line x1="20" y1="21" x2="22" y2="21"/></g>`,
    shapeIconToolPath: `<g stroke-width=".125em" stroke="currentColor" stroke-linecap="round" stroke-linejoin="round" fill="none"><circle cx="21" cy="3" r="2"/><circle cx="9" cy="9" r="2"/><circle cx="3" cy="21" r="2"/><path d="M19 4 11 8 M8 11 4 19"/></g>`,
    shapeIconToolRectangle: `<g stroke-width=".125em" stroke="currentColor" fill="none"><rect x="1" y="1" width="22" height="22" rx="4"/></g>`,
    shapeIconToolEllipse: `<g stroke-width=".125em" stroke="currentColor" fill="none"><circle cx="12" cy="12" r="11"/></g>`,
    shapeIconToolArrow: `<g stroke-width=".125em" stroke="currentColor" fill="none"><line x1="20" y1="3" x2="6" y2="21"/><path d="m10 6 L21.5 1 L20 13.5" fill="currentColor" stroke="none"/></g>`,
    shapeIconToolLine: `<g stroke-width=".125em" stroke="currentColor" fill="none"><line x1="20" y1="3" x2="6" y2="21"/></g>`,
    shapeIconToolText: `<g stroke="none" fill="currentColor" transform="translate(6,0)"><path d="M8.14 20.085c.459 0 .901-.034 1.329-.102a8.597 8.597 0 001.015-.21v1.984c-.281.135-.695.247-1.242.336a9.328 9.328 0 01-1.477.133c-3.312 0-4.968-1.745-4.968-5.235V6.804H.344v-1.25l2.453-1.078L3.89.819h1.5v3.97h4.97v2.015H5.39v10.078c0 1.031.245 1.823.735 2.375s1.161.828 2.015.828z"/>`,
    shapeIconToolPreset: `<g fill="none" stroke-linecap="round" stroke-linejoin="round" stroke="currentColor" stroke-width=".125em"><path d="M12 22c2.773 0 1.189-5.177 3-7 1.796-1.808 7-.25 7-3 0-5.523-4.477-10-10-10S2 6.477 2 12s4.477 10 10 10z"></path><path d="M20 17c-3 3-5 5-8 5"></path></g>`,

    shapeTitleSelectionMode: 'चयन मोड',
    shapeTitleBrushSize: 'ब्रश का आकार',

    shapeLabelSelectionModeNew: 'नया',
    shapeLabelSelectionModeAdd: 'जोड़ें',
    shapeLabelSelectionModeSubtract: 'हटाएं',

    shapeLabelToolSelectionBrush: 'ब्रश',
    shapeLabelToolSelectionLassoo: 'लासू',
    shapeLabelToolSelectionRectangle: 'आयताकार चयन',
    shapeLabelToolSelectionEllipse: 'दीर्घवृत्त चयन',

    shapeIconSelectionModeNew: `<g fill="none" fill-rule="evenodd" stroke="currentColor" stroke-linecap="round" stroke-linejoin="round"><path d="M6.5 17H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h10a2 2 0 0 1 2 2v1.5"/><rect width="12" height="12" x="9" y="9" fill="currentColor" fill-opacity=".25" rx="2"/></g>`,
    shapeIconSelectionModeAdd: `<g fill="none" fill-rule="evenodd" stroke="currentColor"><path fill="currentColor" fill-opacity=".25" stroke-linecap="round" stroke-linejoin="round" d="M15 3a2 2 0 0 1 2 2v4h2a2 2 0 0 1 2 2v8a2 2 0 0 1-2 2h-8a2 2 0 0 1-2-2v-2H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h10Z"/><path d="M13 15h4M15 13v4"/></g>`,
    shapeIconSelectionModeSubtract: `<g fill="none" fill-rule="evenodd" stroke="currentColor"><path fill="currentColor" fill-opacity=".25" stroke-linecap="round" stroke-linejoin="round" d="M15 3a2 2 0 0 1 2 2v4h-6a2 2 0 0 0-1.995 1.85L9 11v6H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h10Z"/><rect width="12" height="12" x="9" y="9" stroke-linecap="round" stroke-linejoin="round" rx="2"/><path d="M13 15h4"/></g>`,
    shapeIconToolSelectionBrush: `<g stroke-width=".125em" stroke="currentColor" fill="none"><path d="M2.025 5c5.616-2.732 8.833-3.857 9.65-3.374C12.903 2.351.518 12.666 2.026 14 3.534 15.334 16.536.566 17.73 2.566 18.924 4.566 3.98 17.187 4.831 18c.851.813 9.848-6 11.643-6 1.087 0-2.53 5.11-2.92 7-.086.41 3.323-1.498 4.773-1 .494.17.64 2.317 1.319 3 .439.443 1.332.776 2.679 1" stroke="currentColor" fill-rule="evenodd" stroke-linejoin="round"/></g>`,
    shapeIconToolSelectionLassoo: `<g fill="none" fill-rule="evenodd" stroke-width=".125em" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" d="M11.428 17.702a18.46 18.46 0 0 0 4.085-1.595c5.851-3.112 9.019-8.599 7.074-12.256-1.945-3.657-8.265-4.1-14.116-.988C2.619 5.974-.55 11.46 1.396 15.118c.63 1.186 1.72 2.033 3.105 2.532"/><ellipse cx="8" cy="18.5" rx="3.5" ry="2.833" transform="rotate(-15 8 18.5)"/><path stroke-linecap="round" d="M5 18c3.347 1.048 5.514 1.881 6.5 2.5.859.54 1.517.994 1.5 2.364"/></g>`,
    shapeIconToolSelectionRectangle: `<g stroke-width=".125em" stroke="currentColor"><path d="M9 1 h6 m4 0 h4v4 m0 4 v6 m0 4 v4h-4 m-4 0 h-6 m-4 0 h-4v-4 m0 -4 v-6 m0 -4 v-4h4" fill="none" /></g>`,
    shapeIconToolSelectionEllipse: `<path stroke-width=".125em" stroke="currentColor" d="M1.21 9.853a11.054 11.054 0 0 0 0 4.294m1.643 3.965a11.054 11.054 0 0 0 3.035 3.035m3.965 1.644a11.054 11.054 0 0 0 4.294 0m3.965-1.644a11.054 11.054 0 0 0 3.035-3.035m1.644-3.965a11.054 11.054 0 0 0 0-4.294m-1.644-3.965a11.054 11.054 0 0 0-3.035-3.035m-3.965-1.644a11.054 11.054 0 0 0-4.294 0M5.888 2.853a11.054 11.054 0 0 0-3.035 3.035"/>`,
};

// deprecated
export const ShapeEditor = MarkupEditor;
