// jsdom lacks object URLs; previews only need a placeholder string.
if (!URL.createObjectURL) {
  (URL as any).createObjectURL = () => "blob:test";
  (URL as any).revokeObjectURL = () => {};
}

// jsdom lacks IntersectionObserver (used by the site's fade-in wrapper): report everything as visible.
if (!(globalThis as any).IntersectionObserver) {
  (globalThis as any).IntersectionObserver = class {
    constructor(private cb: (entries: { isIntersecting: boolean }[]) => void) {}
    observe() {
      this.cb([{ isIntersecting: true }]);
    }
    unobserve() {}
    disconnect() {}
  };
}

// jsdom has no WebGL: report "no context" so the architecture page uses its 2D fallback
// (the same path a browser without WebGL takes) instead of logging "not implemented".
(HTMLCanvasElement.prototype as any).getContext = () => null;

// Reduced motion: animated numbers snap to their final value, so tests read real values.
if (!window.matchMedia) {
  (window as any).matchMedia = (q: string) => ({
    matches: q.includes("reduce"),
    media: q,
    onchange: null,
    addListener() {},
    removeListener() {},
    addEventListener() {},
    removeEventListener() {},
    dispatchEvent: () => false,
  });
}

if (!(globalThis as any).ResizeObserver) {
  (globalThis as any).ResizeObserver = class {
    observe() {}
    unobserve() {}
    disconnect() {}
  };
}
(window as any).scrollTo = () => {};
(Element.prototype as any).scrollIntoView = function () {};
