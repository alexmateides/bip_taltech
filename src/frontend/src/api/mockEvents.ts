import { type VideoEvent, type VideoEventListItem, EventType } from "../types/events";

const THUMBNAIL_URL =
    "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcScCOdgew5pMPsQp8sGzeo7vwC6bdRrHwnlZQ&s";

// Mock list of events with thumbnails
export const mockEventsList: VideoEventListItem[] = [
    { id: "evt1", type: EventType.PersonDetected, timestamp: 12.3, thumbnailUrl: THUMBNAIL_URL },
    { id: "evt2", type: EventType.CarDetected, timestamp: 45.6, thumbnailUrl: THUMBNAIL_URL },
    { id: "evt3", type: EventType.ObjectRemoved, timestamp: 78.9, thumbnailUrl: THUMBNAIL_URL },
    { id: "evt4", type: EventType.SuspiciousMovement, timestamp: 102.5, thumbnailUrl: THUMBNAIL_URL },
];

// Generate detailed events for EventDetail page
export const mockEventDetails: VideoEvent[] = mockEventsList.map((e) => ({
    id: e.id,
    type: e.type,
    timestamp: e.timestamp,
    confidence: parseFloat((Math.random() * 0.5 + 0.5).toFixed(2)),
    videoChunkUrl: e.thumbnailUrl, // Use thumbnail as video placeholder
    extraMetadata: {
        description: "This is mock metadata for the event",
        detectedObjects: ["person", "bicycle"],
    },
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
