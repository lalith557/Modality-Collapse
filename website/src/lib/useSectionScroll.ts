import { useEffect } from "react";
import { useSearchParams } from "react-router-dom";

/** Sections use Tailwind `scroll-mt-24` (96 px) so they clear the fixed navbar. */
const OFFSET = 96;

/**
 * HashRouter already uses the URL hash for the route, so in-page targets are passed as
 * `?section=<element id>` (e.g. #/disentanglement?section=adam-vs-sgd). Scrolls after paint
 * (so it runs after the layout's scroll-to-top on navigation), then checks once more and
 * snaps into place if a competing smooth scroll pre-empted it.
 */
export function useSectionScroll() {
  const [params] = useSearchParams();
  const section = params.get("section");
  useEffect(() => {
    if (!section) return;
    const go = (behavior: ScrollBehavior) =>
      document.getElementById(section)?.scrollIntoView({ behavior, block: "start" });
    let raf2 = 0;
    const raf1 = requestAnimationFrame(() => {
      raf2 = requestAnimationFrame(() => go("smooth"));
    });
    const check = window.setTimeout(() => {
      const el = document.getElementById(section);
      if (el && Math.abs(el.getBoundingClientRect().top - OFFSET) > 48) go("auto");
    }, 900);
    return () => {
      cancelAnimationFrame(raf1);
      cancelAnimationFrame(raf2);
      clearTimeout(check);
    };
  }, [section]);
}
