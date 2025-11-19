import * as React from "react";

const TOAST_TIMEOUT = 5000;

interface ToastOptions {
    id?: string;
    title?: string;
    description?: string;
    variant?: "default" | "destructive";
}

interface Toast extends ToastOptions {
    id: string;
}

interface ToastContextValue {
    toasts: Toast[];
    toast: (opts: ToastOptions) => void;
    dismiss: (id: string) => void;
}

const ToastContext = React.createContext<ToastContextValue | undefined>(undefined);

export function ToastProvider({ children }: { children: React.ReactNode }) {
    const [toasts, setToasts] = React.useState<Toast[]>([]);

    const dismiss = React.useCallback((id: string) => {
        setToasts((current) => current.filter((toast) => toast.id !== id));
    }, []);

    const toast = React.useCallback(
        ({ id, ...opts }: ToastOptions) => {
            const nextId = id ?? crypto.randomUUID();
            setToasts((current) => [...current, { id: nextId, ...opts }]);
            window.setTimeout(() => dismiss(nextId), TOAST_TIMEOUT);
        },
        [dismiss]
    );

    return (
        <ToastContext.Provider value={{ toasts, toast, dismiss }}>
            {children}
            <div className="fixed bottom-4 right-4 z-[4000] flex w-full max-w-sm flex-col gap-2">
                {toasts.map((toast) => (
                    <div
                        key={toast.id}
                        className={`rounded-md border bg-background p-4 shadow-lg ${toast.variant === "destructive" ? "border-destructive/50 bg-destructive text-destructive-foreground" : "border-border"}`}
                    >
                        {toast.title && <p className="font-semibold">{toast.title}</p>}
                        {toast.description && (
                            <p className="text-sm text-muted-foreground">{toast.description}</p>
                        )}
                    </div>
                ))}
            </div>
        </ToastContext.Provider>
    );
}

export function useToast() {
    const ctx = React.useContext(ToastContext);
    if (!ctx) {
        throw new Error("useToast must be used within a ToastProvider");
    }
    return ctx;
}

