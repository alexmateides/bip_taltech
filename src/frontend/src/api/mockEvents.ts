import { EventType, type VideoEvent, type VideoEventListItem, type VideoEventVideo, type EventMetricSummary } from "@/types/events";

const THUMBNAIL_URL =
    "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcScCOdgew5pMPsQp8sGzeo7vwC6bdRrHwnlZQ&s";

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
        type: EventType.VehicleCollision,
        timestamp_start: 2,
        timestamp_end: 4,
        confidence: 0.91,
        thumbnailUrl: THUMBNAIL_URL,
        videoId: "video-1",
        location: locations[0],
        occurredAt: baseTime,
    },
    {
        id: "evt2",
        type: EventType.PedestrianCollision,
        timestamp_start: 32,
        timestamp_end: 36,
        confidence: 0.88,
        thumbnailUrl: THUMBNAIL_URL,
        videoId: "video-2",
        location: locations[1],
        occurredAt: new Date(Date.now() - 1000 * 60 * 5).toISOString(),
    },
];

// Generate detailed events for EventDetail page
export const mockEventDetails: VideoEvent[] = mockEventsList.map((event, idx) => ({
    ...event,
    description: "Collision event captured by roadside sensor.",
    detectedObjects: idx === 0 ? ["vehicle", "barrier"] : ["pedestrian", "vehicle"],
    roadSegment: idx === 0 ? "Segment A" : "Segment B",
}));

// Mock video data by ID
export const mockVideoById: Record<string, VideoEventVideo> = Object.fromEntries(
    mockEventsList.map((event) => [
        event.videoId,
        {
            videoId: event.videoId,
            url: "https://interactive-examples.mdn.mozilla.net/media/cc0-videos/flower.mp4",
            thumbnailUrl: event.thumbnailUrl,
        },
    ])
);

// Mock analytics summary data
export const mockMetrics: EventMetricSummary = {
    totalEvents: mockEventsList.length,
    highConfidence: mockEventsList.filter((e) => e.confidence >= 0.9).length,
    lowConfidence: mockEventsList.filter((e) => e.confidence < 0.9).length,
};

// Mock API functions
export function fetchMockEvents(): Promise<VideoEventListItem[]> {
    return new Promise((resolve) => setTimeout(() => resolve(mockEventsList), 500));
}

export function fetchMockEventById(id: string): Promise<VideoEvent | undefined> {
    return new Promise((resolve) =>
        setTimeout(() => resolve(mockEventDetails.find((e) => e.id === id)), 500)
    );
}

export function fetchMockEventVideo(videoId: string): Promise<VideoEventVideo> {
    return new Promise((resolve) => setTimeout(() => resolve(mockVideoById[videoId]), 500));
}

export function fetchMockMetrics(): Promise<EventMetricSummary> {
    return new Promise((resolve) => setTimeout(() => resolve(mockMetrics), 400));
}
