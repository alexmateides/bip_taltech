import { EventType, type VideoEvent, type VideoEventListItem } from "@/types/events";
import thumbnailImages from "@/assets/thumbnails";

// Mock list of events with thumbnails
const locations = [
    { lat: 59.437, lng: 24.753, label: "Tallinn" },
    { lat: 59.438, lng: 24.754, label: "Tallinn Old Town" },
    { lat: 59.439, lng: 24.755, label: "Harju" },
    { lat: 59.44, lng: 24.756, label: "Kesklinn" },
];

const baseTime = new Date().toISOString();

export const mockEventsList: VideoEventListItem[] = [
    {
        id: "evt1",
        camera_id: "1",
        type: EventType.VehicleCollision,
        timestamp_start: 2,
        timestamp_end: 4,
        confidence: 0.91,
        location: locations[0],
        occurred_at: baseTime,
        thumbnailUrl: thumbnailImages[0],
    },
    {
        id: "evt2",
        camera_id: "1",
        type: EventType.PedestrianCollision,
        timestamp_start: 32,
        timestamp_end: 36,
        confidence: 0.88,
        location: locations[1],
        occurred_at: new Date(Date.now() - 1000 * 60 * 5).toISOString(),
        thumbnailUrl: thumbnailImages[1],
    },
];

// Generate detailed events for EventDetail page
export const mockEventDetails: VideoEvent[] = mockEventsList.map((event) => ({
    ...event,
    description: "Collision event captured by roadside sensor.",
}));

// Mock API functions
export function fetchMockEvents(): Promise<VideoEventListItem[]> {
    return new Promise((resolve) => setTimeout(() => resolve(mockEventsList), 500));
}

export function fetchMockEventById(id: string): Promise<VideoEvent | undefined> {
    return new Promise((resolve) =>
        setTimeout(() => resolve(mockEventDetails.find((e) => e.id === id)), 500)
    );
}

export const mockMetrics = {
    totalEvents: mockEventsList.length,
    highConfidence: mockEventsList.filter((e) => e.confidence >= 0.9).length,
    lowConfidence: mockEventsList.filter((e) => e.confidence < 0.9).length,
};
