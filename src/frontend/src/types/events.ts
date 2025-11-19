export const EventType = {
    VehicleCollision: "Collision with vehicle",
    PedestrianCollision: "Collision with pedestrian",
} as const;

export const EventTypeIcon = {
    [EventType.VehicleCollision]: "car",
    [EventType.PedestrianCollision]: "walking",
} as const satisfies Record<EventType, "car" | "walking">;

export type EventType = (typeof EventType)[keyof typeof EventType];

export interface EventLocation {
    lat: number;
    lng: number;
    label?: string;
}

export interface VideoEvent {
    id: string;
    type: EventType;
    timestamp_start: number;
    timestamp_end: number;
    confidence: number;
    thumbnailUrl: string;
    videoId: string;
    location: EventLocation;
    description?: string;
    detectedObjects?: string[];
    roadSegment?: string;
    occurredAt: string;
}

export interface VideoEventListItem {
    id: string;
    type: EventType;
    timestamp_start: number;
    timestamp_end: number;
    confidence: number;
    thumbnailUrl: string;
    videoId: string;
    location: EventLocation;
    occurredAt: string;
}

export interface VideoEventVideo {
    videoId: string;
    url: string;
    thumbnailUrl: string;
}

export interface EventMetricSummary {
    totalEvents: number;
    highConfidence: number;
    lowConfidence: number;
}
