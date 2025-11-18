import { useParams } from "react-router-dom";
import { useEffect, useState } from "react";
import { fetchEventById } from "../api/events";
import type {VideoEvent} from "../types/events";
import VideoPlayer from "../components/VideoPlayer.tsx";

export default function EventDetail() {
    const { id } = useParams();
    const [event, setEvent] = useState<VideoEvent | null>(null);

    useEffect(() => {
        if (!id) return;
        fetchEventById(id).then((data) => setEvent(data));
    }, [id]);

    if (!event) return <div style={{ padding: 24 }}>Loading event…</div>;

    return (
        <div style={{ padding: 24 }}>
            <h1>{event.type}</h1>

            <p>Timestamp: {event.timestamp}</p>
            <p>Confidence: {event.confidence}</p>

            <h2>Video Chunk</h2>
            <VideoPlayer url={event.videoChunkUrl} />

            <h2>Metadata</h2>
            <pre>{JSON.stringify(event.extraMetadata, null, 2)}</pre>
        </div>
    );
}