import { BrowserRouter, Routes, Route, Link, useLocation } from "react-router-dom";
import { HomeIcon, ListChecks, Settings, Sun, Moon, MapPinned } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Sheet, SheetContent, SheetTrigger, SheetHeader, SheetTitle } from "@/components/ui/sheet";
import { Input } from "@/components/ui/input";
import { useState, useMemo, useEffect } from "react";
import Home from "./pages/Home";
import EventList from "./pages/EventList";
import EventDetail from "./pages/EventDetail";
import EventsMapPage from "./pages/EventsMapPage";
import { useSettings } from "@/context/SettingsContext";
import { AllpueliIntro } from "@/components/AllpueliIntro";
// eslint-disable-next-line @typescript-eslint/ban-ts-comment
// @ts-expect-error
import AllpueliLogo from "@/assets/allpueli_logo_invertable_shrinked.svg?react";

function Layout() {
    const location = useLocation();
    const { reportEmail, stagedReportEmail, setStagedReportEmail, saveReportEmail, resetStagedEmail } = useSettings();
    const [isDark, setIsDark] = useState(false);
    const [isSettingsOpen, setIsSettingsOpen] = useState(false);
    const [showIntro, setShowIntro] = useState(() => localStorage.getItem("allpueli-intro-played") !== "true");

    const emailIsValid = useMemo(() => {
        if (!stagedReportEmail) return false;
        return /^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(stagedReportEmail);
    }, [stagedReportEmail]);

    const toggleTheme = () => {
        setIsDark((prev) => !prev);
        document.documentElement.classList.toggle("dark");
    };

    const handleSheetOpenChange = (open: boolean) => {
        if (!open) {
            resetStagedEmail();
        }
        setIsSettingsOpen(open);
    };

    const handleSaveEmail = () => {
        if (!emailIsValid) return;
        saveReportEmail();
        setIsSettingsOpen(false);
    };

    useEffect(() => {
        if (!showIntro) return;
        const timeout = window.setTimeout(() => {
            localStorage.setItem("allpueli-intro-played", "true");
            setShowIntro(false);
        }, 2400);
        return () => window.clearTimeout(timeout);
    }, [showIntro]);

    return (
        <div className="min-h-screen bg-background text-foreground">
            <header className="sticky top-0 z-[1000] border-b bg-background/80 backdrop-blur">
                <nav className="mx-auto flex max-w-6xl items-center gap-4 px-6 py-3">
                    <Link to="/" className="flex items-center gap-2" aria-label="Allpueli home">
                        <AllpueliLogo className="h-10 w-auto text-black dark:text-white" />
                    </Link>
                    <div className="flex items-center gap-2">
                        <Button variant={location.pathname === "/" ? "default" : "ghost"} asChild size="icon">
                            <Link to="/">
                                <HomeIcon className="size-4" />
                            </Link>
                        </Button>
                        <Button
                            variant={location.pathname.startsWith("/events") ? "default" : "ghost"}
                            asChild
                        >
                            <Link to="/events" className="inline-flex items-center gap-2">
                                <ListChecks className="size-4" />
                                Events
                            </Link>
                        </Button>
                        <Button variant={location.pathname === "/map" ? "default" : "ghost"} asChild>
                            <Link to="/map" className="inline-flex items-center gap-2">
                                <MapPinned className="size-4" />
                                Map
                            </Link>
                        </Button>
                    </div>
                    <div className="ml-auto flex items-center gap-3">
                        <Button variant="ghost" size="icon" onClick={toggleTheme}>
                            {isDark ? <Sun className="size-5" /> : <Moon className="size-5" />}
                        </Button>
                        <Sheet open={isSettingsOpen} onOpenChange={handleSheetOpenChange}>
                            <SheetTrigger asChild>
                                <Button variant="ghost" size="icon">
                                    <Settings className="size-5" />
                                </Button>
                            </SheetTrigger>
                            <SheetContent>
                                <SheetHeader>
                                    <SheetTitle>Settings</SheetTitle>
                                </SheetHeader>
                                <div className="mt-6 space-y-4">
                                    <label className="text-sm font-medium" htmlFor="report-email-input">
                                        Email for reports
                                    </label>
                                    <Input
                                        id="report-email-input"
                                        value={stagedReportEmail}
                                        onChange={(event) => setStagedReportEmail(event.target.value)}
                                        placeholder="alerts@example.com"
                                        type="email"
                                    />
                                    {!emailIsValid && stagedReportEmail.length > 0 && (
                                        <p className="text-sm text-destructive">Enter a valid email.</p>
                                    )}
                                    <div className="flex justify-end gap-2 pt-2">
                                        <Button variant="outline" onClick={() => handleSheetOpenChange(false)}>
                                            Cancel
                                        </Button>
                                        <Button onClick={handleSaveEmail} disabled={!emailIsValid}>
                                            Save
                                        </Button>
                                    </div>
                                </div>
                                <p className="mt-4 text-xs text-muted-foreground">Current email: {reportEmail}</p>
                            </SheetContent>
                        </Sheet>
                    </div>
                </nav>
            </header>
            <main>
                {showIntro && <AllpueliIntro />}
                <Routes>
                    <Route path="/" element={<Home />} />
                    <Route path="/events" element={<EventList />} />
                    <Route path="/events/:id" element={<EventDetail />} />
                    <Route path="/map" element={<EventsMapPage />} />
                </Routes>
            </main>
        </div>
    );
}

export default function App() {
    return (
        <BrowserRouter>
            <Layout />
        </BrowserRouter>
    );
}
