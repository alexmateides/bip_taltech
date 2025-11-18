import { useEffect, useState } from "react";
import { fetchEvents } from "../api/events";
import type {VideoEventListItem} from "../types/events";
import EventCard from "../components/EventCard";

export default function EventList() {
    const [events, setEvents] = useState<VideoEventListItem[]>([]);
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        fetchEvents().then((data) => {
            setEvents(data);
            setLoading(false);
        });
    }, []);

    if (loading) return <div style={{ padding: 24 }}>Loading events…</div>;

    return (
        <div style={{ padding: 24 }}>
            <h1>Detected Events</h1>
            {events.map((e) => (
                <EventCard key={e.id} event={e} />
            ))}
        </div>
    );
}
