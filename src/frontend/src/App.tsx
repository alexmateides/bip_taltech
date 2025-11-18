import { BrowserRouter, Routes, Route } from "react-router-dom";
import Home from "./pages/Home";
import EventList from "./pages/EventList";
import EventDetail from "./pages/EventDetail";

export default function App() {
    return (
        <BrowserRouter>
            <Routes>
                <Route path="/" element={<Home />} />
                <Route path="/events" element={<EventList />} />
                <Route path="/events/:id" element={<EventDetail />} />
            </Routes>
        </BrowserRouter>
    );
}
