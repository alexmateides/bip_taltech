/* eslint-disable react-refresh/only-export-components */
import { createContext, useContext, useEffect, useState, type ReactNode } from "react";

interface SettingsContextValue {
    reportEmail: string;
    stagedReportEmail: string;
    setStagedReportEmail: (value: string) => void;
    saveReportEmail: () => void;
    resetStagedEmail: () => void;
}

// Add defaults and storage helpers
const STORAGE_KEY = "reportEmail";
const DEFAULT_REPORT_EMAIL = "alerts@example.com";

const SettingsContext = createContext<SettingsContextValue | undefined>(undefined);

export function SettingsProvider({ children }: { children: ReactNode }) {
    const [reportEmail, setReportEmail] = useState(() => getStoredReportEmail());
    const [stagedReportEmail, setStagedReportEmail] = useState(reportEmail);

    const saveReportEmail = () => {
        setReportEmail(stagedReportEmail);
    };

    const resetStagedEmail = () => {
        setStagedReportEmail(reportEmail);
    };

    useEffect(() => {
        if (typeof window === "undefined") return;
        window.localStorage.setItem(STORAGE_KEY, reportEmail);
    }, [reportEmail]);

    return (
        <SettingsContext.Provider
            value={{ reportEmail, stagedReportEmail, setStagedReportEmail, saveReportEmail, resetStagedEmail }}
        >
            {children}
        </SettingsContext.Provider>
    );
}

export function useSettings() {
    const context = useContext(SettingsContext);
    if (!context) {
        throw new Error("useSettings must be used within a SettingsProvider");
    }
    return context;
}

function getStoredReportEmail() {
    if (typeof window === "undefined") return DEFAULT_REPORT_EMAIL;
    const storedEmail = window.localStorage.getItem(STORAGE_KEY);
    return storedEmail !== null ? storedEmail : DEFAULT_REPORT_EMAIL;
}
