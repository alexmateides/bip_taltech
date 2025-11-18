import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import "./index.css";
import App from "./App.tsx";
import { SettingsProvider } from "@/context/SettingsContext";

const queryClient = new QueryClient();

createRoot(document.getElementById("root")!).render(
    <StrictMode>
        <QueryClientProvider client={queryClient}>
            <SettingsProvider>
                <div className="min-h-screen bg-background text-foreground">
                    <App />
                </div>
            </SettingsProvider>
        </QueryClientProvider>
    </StrictMode>
);
