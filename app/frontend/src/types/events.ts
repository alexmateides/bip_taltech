export const EventType = {
    VehicleCollision: "Collision with vehicle",
    PedestrianCollision: "Collision with pedestrian",
} as const;

export const EventTypeIcon = {
    [EventType.VehicleCollision]: "car",
    [EventType.PedestrianCollision]: "walking",
} as const satisfies Record<(typeof EventType)[keyof typeof EventType], "car" | "walking">;

export type EventType = (typeof EventType)[keyof typeof EventType];

export interface EventLocation {
    lat: number;
    lng: number;
    label?: string;
}

export type VideoEvent = {
    id: string;
    camera_id: string;
    type: EventType;
    timestamp_start: number;
    timestamp_end: number;
    confidence: number;
    location: EventLocation;
    description?: string;
    occurred_at: string;
    thumbnailUrl?: string;
};

export type VideoEventListItem = VideoEvent;

export interface EventMetricSummary {
    totalEvents: number;
    highConfidence: number;
    lowConfidence: number;
}
