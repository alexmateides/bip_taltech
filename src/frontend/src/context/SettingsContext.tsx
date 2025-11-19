/* eslint-disable react-refresh/only-export-components */
import { createContext, useContext, useState, type ReactNode } from "react";

interface SettingsContextValue {
    reportEmail: string;
    stagedReportEmail: string;
    setStagedReportEmail: (value: string) => void;
    saveReportEmail: () => void;
    resetStagedEmail: () => void;
}

const SettingsContext = createContext<SettingsContextValue | undefined>(undefined);

export function SettingsProvider({ children }: { children: ReactNode }) {
    const [reportEmail, setReportEmail] = useState("alerts@example.com");
    const [stagedReportEmail, setStagedReportEmail] = useState(reportEmail);

    const saveReportEmail = () => {
        setReportEmail(stagedReportEmail);
    };

    const resetStagedEmail = () => {
        setStagedReportEmail(reportEmail);
    };

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
