export const EventType = {
    PersonDetected: "Person Detected",
    CarDetected: "Car Detected",
    ObjectRemoved: "Object Removed",
    SuspiciousMovement: "Suspicious Movement",
} as const;

export type EventType = (typeof EventType)[keyof typeof EventType];

export interface VideoEvent {
    id: string;
    type: EventType;
    timestamp: number;
    confidence: number;
    videoChunkUrl: string;
    extraMetadata?: Record<string, unknown>;
}

export interface VideoEventListItem {
    id: string;
    type: EventType;
    timestamp: number;
    thumbnailUrl: string;
}
