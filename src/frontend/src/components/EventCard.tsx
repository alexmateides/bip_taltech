import { Link } from "react-router-dom";
import type {VideoEventListItem} from "../types/events";

interface Props {
    event: VideoEventListItem;
}

export default function EventCard({ event }: Props) {
    return (
        <div
            style={{
                display: "flex",
                alignItems: "center",
                padding: 12,
                border: "1px solid #ddd",
                borderRadius: 8,
                marginBottom: 12,
                gap: 12,
            }}
        >
            <img
                src={event.thumbnailUrl}
                alt="thumbnail"
                style={{ width: 120, height: 68, objectFit: "cover", borderRadius: 4 }}
            />
            <div>
                <Link to={`/events/${event.id}`} style={{ textDecoration: "none", color: "black" }}>
                    <strong>{event.type}</strong>
                    <div>Timestamp: {event.timestamp}</div>
                </Link>
            </div>
        </div>
    );
}
