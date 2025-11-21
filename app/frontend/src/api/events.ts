import client from "./client";
import { fetchMockEvents, mockMetrics } from "./mockEvents";
import type { EventMetricSummary, EventLocation, VideoEvent, VideoEventListItem } from "@/types/events";
import { EventType } from "@/types/events";
import { useQuery } from "@tanstack/react-query";
import thumbnails from "@/assets/thumbnails";

const EVENTS_BASE = "/events";

const useMockData = import.meta.env.VITE_USE_MOCKS === "true";

type BackendLocation =
    | { lat?: number; lng?: number; label?: string }
    | string
    | null
    | undefined;

interface BackendEvent {
    id?: string;
    camera_id?: string;
    type?: string;
    timestamp_start?: number;
    timestamp_end?: number;
    confidence?: number;
    location?: BackendLocation;
    occurred_at?: string;
    description?: string;
}

type BackendEventsResponse = Array<{
    camera: string;
    events?: { events?: BackendEvent[] };
}>;

const FALLBACK_COORDS = { lat: 59.437, lng: 24.753 } as const;

const LOCATION_LOOKUP: Record<string, { lat: number; lng: number }> = {
    "Tallinn Old Town": { lat: 59.437, lng: 24.753 },
    Kesklinn: { lat: 59.437, lng: 24.745 },
};

const EVENT_TYPE_MAP: Record<string, EventType> = {
    "Vehicle collision": EventType.VehicleCollision,
    "Collision with vehicle": EventType.VehicleCollision,
    "Pedestrian near-miss": EventType.PedestrianCollision,
    "Collision with pedestrian": EventType.PedestrianCollision,
};

function normalizeEventType(raw?: string): EventType {
    if (!raw) return EventType.VehicleCollision;
    return EVENT_TYPE_MAP[raw] ?? EventType.VehicleCollision;
}

function normalizeLocation(location: BackendLocation): EventLocation {
    if (location && typeof location === "object" && "lat" in location && "lng" in location) {
        return {
            lat: Number(location.lat ?? FALLBACK_COORDS.lat),
            lng: Number(location.lng ?? FALLBACK_COORDS.lng),
            label: location.label,
        };
    }
    if (typeof location === "string") {
        const preset = LOCATION_LOOKUP[location];
        if (preset) return { ...preset, label: location };
        return { ...FALLBACK_COORDS, label: location };
    }
    return { ...FALLBACK_COORDS, label: "Tallinn" };
}

function normalizeBackendEvent(event: BackendEvent, index: number): VideoEventListItem {
    return {
        id: event.id ?? `evt-${index}`,
        camera_id: event.camera_id ?? "unknown",
        type: normalizeEventType(event.type),
        timestamp_start: Number(event.timestamp_start ?? 0),
        timestamp_end: Number(event.timestamp_end ?? 0),
        confidence: Number(event.confidence ?? 0),
        location: normalizeLocation(event.location),
        occurred_at: event.occurred_at ?? new Date().toISOString(),
        description: event.description,
        thumbnailUrl: thumbnails[index % thumbnails.length],
    };
}

async function fetchEventsApi(): Promise<VideoEventListItem[]> {
    if (useMockData) return fetchMockEvents();
    const { data } = await client.get<BackendEventsResponse>(EVENTS_BASE);
    const flattened = data.flatMap(({ camera, events }) =>
        (events?.events ?? []).map((evt) => ({ ...evt, camera_id: evt.camera_id ?? camera }))
    );
    return flattened.map((event, index) => normalizeBackendEvent(event, index));
}

async function fetchEventMetricsApi(): Promise<EventMetricSummary> {
    if (useMockData) {
        return mockMetrics;
    }
    const { data } = await client.get<EventMetricSummary>(`${EVENTS_BASE}/metrics`);
    return data;
}

export function useEventsQuery() {
    return useQuery({
        queryKey: ["events"],
        queryFn: fetchEventsApi,
    });
}

export function calculateEventMetrics(events?: VideoEventListItem[]): EventMetricSummary {
    if (!events?.length) {
        return { totalEvents: 0, highConfidence: 0, lowConfidence: 0 };
    }
    const totalEvents = events.length;
    const highConfidence = events.filter((e) => e.confidence >= 0.9).length;
    const lowConfidence = events.filter((e) => e.confidence < 0.9).length;
    return { totalEvents, highConfidence, lowConfidence };
}

export function useEventMetrics(events?: VideoEventListItem[]) {
    return calculateEventMetrics(events);
}

export function useEventMetricsQuery() {
    return useQuery({
        queryKey: ["events", "metrics"],
        queryFn: fetchEventMetricsApi,
    });
}

export function findEventById(events: VideoEvent[] | undefined, id?: string) {
    if (!events || !id) return undefined;
    return events.find((evt) => evt.id === id);
}
