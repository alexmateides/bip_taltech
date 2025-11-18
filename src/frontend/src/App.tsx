import { BrowserRouter, Routes, Route, Link, useLocation } from "react-router-dom";
import { HomeIcon, ListChecks, Settings, Sun, Moon } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Sheet, SheetContent, SheetTrigger, SheetHeader, SheetTitle } from "@/components/ui/sheet";
import { Input } from "@/components/ui/input";
import { useState } from "react";
import Home from "./pages/Home";
import EventList from "./pages/EventList";
import EventDetail from "./pages/EventDetail";
import { useSettings } from "@/context/SettingsContext";

function Layout() {
    const location = useLocation();
    const { reportEmail, setReportEmail } = useSettings();
    const [isDark, setIsDark] = useState(false);

    const toggleTheme = () => {
        setIsDark((prev) => !prev);
        document.documentElement.classList.toggle("dark");
    };

    return (
        <div className="min-h-screen bg-background text-foreground">
            <header className="sticky top-0 z-[1000] border-b bg-background/80 backdrop-blur">
                <nav className="mx-auto flex max-w-6xl items-center gap-4 px-6 py-4">
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
                    <div className="ml-auto flex items-center gap-3">
                        <Button variant="ghost" size="icon" onClick={toggleTheme}>
                            {isDark ? <Sun className="size-5" /> : <Moon className="size-5" />}
                        </Button>
                        <Sheet>
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
                                    <label className="text-sm font-medium">Email for reports</label>
                                    <Input value={reportEmail} onChange={(event) => setReportEmail(event.target.value)} />
                                </div>
                            </SheetContent>
                        </Sheet>
                    </div>
                </nav>
            </header>
            <main>
                <Routes>
                    <Route path="/" element={<Home />} />
                    <Route path="/events" element={<EventList />} />
                    <Route path="/events/:id" element={<EventDetail />} />
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
