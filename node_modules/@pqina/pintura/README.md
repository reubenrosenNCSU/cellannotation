# Pintura Test Version

This package is for testing Pintura in your project. This version of Pintura will overlay a watermark on top of the editor and output image. You can purchase a license on the [Pintura product page](https://pqina.nl/pintura/).

Documentation: https://pqina.nl/pintura/docs/

## Installation

```bash
npm install @pqina/pintura
```

If you're using Svelte, Vue, React, or Angular, you can install the matching adapter components like this.

```bash
npm install @pqina/svelte-pintura
```

## Example implementations

Use one of the example projects below as a starting point or guideline.

JavaScript:

-   [JavaScript](https://github.com/pqina/pintura-example-javascript)
-   [PinturaInput](https://github.com/pqina/pintura-example-pintura-input)
-   [CustomElement](https://github.com/pqina/pintura-example-custom-element)

Frameworks:

-   [jQuery](https://github.com/pqina/pintura-example-jquery)
-   [Angular](https://github.com/pqina/pintura-example-angular)
-   [Vue 2](https://github.com/pqina/pintura-example-vue-2)
-   [Vue 3](https://github.com/pqina/pintura-example-vue-3)
-   [Nuxt 2](https://github.com/pqina/pintura-example-nuxt-2)
-   [Nuxt 3](https://github.com/pqina/pintura-example-nuxt-3)
-   [Svelte](https://github.com/pqina/pintura-example-svelte)
-   [SvelteKit](https://github.com/pqina/pintura-example-sveltekit)
-   [React](https://github.com/pqina/pintura-example-react)
-   [React TypeScript](https://github.com/pqina/pintura-example-react-typescript)
-   [NextJS](https://github.com/pqina/pintura-example-nextjs)
-   [React Native](https://github.com/pqina/pintura-example-react-native)
-   [React Native TypeScript](https://github.com/pqina/pintura-example-react-native-typescript)
-   [Alpine](https://github.com/pqina/pintura-example-alpine)
-   [Cordova](https://github.com/pqina/pintura-example-cordova)
-   [Capacitor](https://github.com/pqina/pintura-example-capacitor)
-   [Ionic 6](https://github.com/pqina/pintura-example-ionic-6)

File upload libraries:

-   [FilePond](https://github.com/pqina/pintura-example-filepond)
-   [jQuery File Upload](https://github.com/pqina/pintura-example-jquery-file-upload)
-   [React Dropzone](https://github.com/pqina/pintura-example-react-dropzone)
-   [Dropzone](https://github.com/pqina/pintura-example-dropzone)
-   [Uppy](https://github.com/pqina/pintura-example-uppy)

## Components and Adapters

Individual components can be found below:

-   [PinturaInput](https://github.com/pqina/pintura-component-pintura-input)
-   [Vue](https://github.com/pqina/pintura-component-vue)
-   [React](https://github.com/pqina/pintura-component-react)
-   [Angular](https://github.com/pqina/pintura-component-angular)
-   [Svelte](https://github.com/pqina/pintura-component-svelte)
-   [React Native](https://github.com/pqina/pintura-component-react-native)

Adapters can be found here:

-   [jQuery](https://github.com/pqina/pintura-adapter-jquery)
-   [FilePond](https://github.com/pqina/filepond-plugin-image-editor)
-   [Uppy](https://github.com/pqina/pintura-adapter-uppy)
-   [Dropzone](https://github.com/pqina/pintura-adapter-dropzone)

## Compatibility

Pintura works on all modern browsers and devices.

-   Chrome
-   Firefox
-   Edge 18+
-   Safari 10+
-   Opera
-   Chrome for Android
-   Firefox Android
-   iOS Safari 10+
-   Chrome for iOS
-   Firefox iOS

### Safari 11 and 12

-   Safari 11 (global usage 0.05%)
-   Safari 12 (global usage 0.01%)

To add support for these browsers we need to polyfill Pointer Events.
Download polyfill here: https://github.com/Rich-Harris/Points

### Safari 10 and Edge 18

-   Safari 10 (global usage 0.01%)
-   Edge 18 (global usage 0.23%)

These browsers don't fully support JavaScript modules and will require polyfills to function correctly. To make Pintura Image Editor work on these browsers you'll need to load the IIFE version or transpile a compatible version yourself.

Required polyfills for Edge 18:

-   Symbol.asyncIterator
-   HTMLCanvasElement.prototype.toBlob

Include this URL in your page to polyfill both the APIs above: https://polyfill.io/v3/polyfill.min.js?features=Symbol.asyncIterator%2CHTMLCanvasElement.prototype.toBlob

### Internet Explorer 11

Internet Explorer 11 is not supported, global usage is at around 0.5%, it is end of life.
