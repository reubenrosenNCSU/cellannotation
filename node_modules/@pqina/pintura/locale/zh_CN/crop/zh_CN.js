export default {
    cropLabel: '裁剪',
    cropIcon:
        '<g stroke-width=".125em" stroke="currentColor" fill="none"><path d="M23 17H9a2 2 0 0 1-2-2v-5m0-3V1 M1 7h14a2 2 0 0 1 2 2v7m0 4v3"/></g>',
    cropIconButtonRecenter: `<path stroke="currentColor" fill="none" stroke-width="2" stroke-linejoin="bevel" d="M1.5 7.5v-6h6M1.5 16.5v6h6M22.5 16.5v6h-6M22.5 7.5v-6h-6"/><circle cx="12" cy="12" r="3.5" fill="currentColor" stroke="none"/>`,
    cropIconButtonRotateLeft:
        '<g stroke="none" fill="currentColor"><path fill="none" d="M-1-1h582v402H-1z"/><rect x="3" rx="1" height="12" width="12" y="9"/><path d="M15 5h-1a5 5 0 015 5 1 1 0 002 0 7 7 0 00-7-7h-1.374l.747-.747A1 1 0 0011.958.84L9.603 3.194a1 1 0 000 1.415l2.355 2.355a1 1 0 001.415-1.414l-.55-.55H15z"/></g>',
    cropIconButtonRotateRight:
        '<g stroke="none" fill="currentColor"><path fill="none" d="M-1-1h582v402H-1z"/><path d="M11.177 5H10a5 5 0 00-5 5 1 1 0 01-2 0 7 7 0 017-7h1.374l-.747-.747A1 1 0 0112.042.84l2.355 2.355a1 1 0 010 1.415l-2.355 2.354a1 1 0 01-1.415-1.414l.55-.55z"/><rect rx="1" height="12" width="12" y="9" x="9"/></g>',
    cropIconButtonFlipVertical:
        '<g stroke="none" fill="currentColor"><path d="M19.993 12.143H7a1 1 0 0 1-1-1V5.994a1 1 0 0 1 1.368-.93l12.993 5.15a1 1 0 0 1-.368 1.93z"/><path d="M19.993 14a1 1 0 0 1 .368 1.93L7.368 21.078A1 1 0 0 1 6 20.148V15a1 1 0 0 1 1-1h12.993z" opacity=".6"/></g>',
    cropIconButtonFlipHorizontal:
        '<g stroke="none" fill="currentColor"><path d="M11.93 7.007V20a1 1 0 0 1-1 1H5.78a1 1 0 0 1-.93-1.368l5.15-12.993a1 1 0 0 1 1.929.368z"/><path d="M14 7.007V20a1 1 0 0 0 1 1h5.149a1 1 0 0 0 .93-1.368l-5.15-12.993A1 1 0 0 0 14 7.007z" opacity=".6"/></g>',

    cropIconSelectPreset: `<g fill="currentColor">
        <rect opacity="{a}" x="2" y="4" width="10" height="18" rx="1"/>
        <rect opacity="{b}" x="4" y="8" width="14" height="14" rx="1"/>
        <rect opacity="{c}" x="6" y="12" width="17" height="10" rx="1"/>
    </g>`,

    cropIconCropBoundary: `<g fill="currentColor">
        <rect opacity="{a}" x="2" y="3" width="20" height="20" rx="1"/>
        <rect opacity="{b}" x="7" y="8" width="10" height="10" rx="1"/>
        <rect opacity="{c}" x="4" y="8" width="14" height="14" rx="1"/>
        <rect opacity="{d}" x="12" y="4" width="10" height="10" rx="1"/>
    </g>`,

    cropLabelButtonRecenter: '重新居中',
    cropLabelButtonRotateLeft: '左旋转',
    cropLabelButtonRotateRight: '右旋转',
    cropLabelButtonFlipHorizontal: '水平翻转',
    cropLabelButtonFlipVertical: '垂直翻转',

    cropLabelSelectPreset: '裁剪形状',

    cropLabelCropBoundary: '裁剪边界',
    cropLabelCropBoundaryEdge: '图像边缘',
    cropLabelCropBoundaryNone: '无',

    cropLabelTabRotation: '旋转',
    cropLabelTabZoom: '缩放',
};
