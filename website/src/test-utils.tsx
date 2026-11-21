/** Renders the real app routes (with the real layout and providers) at a given URL. */
import { render } from "@testing-library/react";
import { MemoryRouter, useLocation } from "react-router-dom";
import { AppRoutes } from "./App";
import { ResearchProvider } from "./state/research-context";
import { PageContextProvider } from "./state/page-context";
import { StudioProvider } from "./studio/StudioContext";

export let currentLocation = { pathname: "", search: "" };

function LocationSpy() {
  const loc = useLocation();
  currentLocation = { pathname: loc.pathname, search: loc.search };
  return null;
}

export function renderAt(path: string) {
  return render(
    <ResearchProvider>
      <StudioProvider>
        <PageContextProvider>
          <MemoryRouter initialEntries={[path]}>
            <AppRoutes />
            <LocationSpy />
          </MemoryRouter>
        </PageContextProvider>
      </StudioProvider>
    </ResearchProvider>,
  );
}
