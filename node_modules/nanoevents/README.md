# Nano Events

Simple and tiny event emitter library for JavaScript.

* Only **108 bytes** (minified and brotlied).
  It uses [Size Limit] to control size.
* The `on` method returns `unbind` function. You don’t need to save
  callback to variable for `removeListener`.
* TypeScript and ES modules support.
* No aliases, just `emit` and `on` methods.
  No Node.js [EventEmitter] compatibility.

```js
import { createNanoEvents } from 'nanoevents'

const emitter = createNanoEvents()

const unbind = emitter.on('tick', volume => {
  summary += volume
})

emitter.emit('tick', 2)
summary //=> 2

unbind()
emitter.emit('tick', 2)
summary //=> 2
```

[EventEmitter]: https://nodejs.org/api/events.html
[Size Limit]:   https://github.com/ai/size-limit

---

<img src="https://cdn.evilmartians.com/badges/logo-no-label.svg" alt="" width="22" height="16" />  Made at <b><a href="https://evilmartians.com/devtools?utm_source=nanoevents&utm_campaign=devtools-button&utm_medium=github">Evil Martians</a></b>, product consulting for <b>developer tools</b>.

---


## Docs
Read full docs **[here](https://github.com/ai/nanoevents#readme)**.
