import client from "./client";
import { fetchMockEvents, mockMetrics } from "./mockEvents";
import type { EventMetricSummary, VideoEvent, VideoEventListItem } from "@/types/events";
import { useQuery } from "@tanstack/react-query";
import thumbnails from "@/assets/thumbnails";

const EVENTS_BASE = "/events";

const useMockData = import.meta.env.VITE_USE_MOCKS === "true";

async function fetchEventsApi(): Promise<VideoEventListItem[]> {
    if (useMockData) return fetchMockEvents();
    const { data } = await client.get<VideoEventListItem[]>(EVENTS_BASE);
    return data.map((event, index) => ({
        ...event,
        thumbnailUrl: thumbnails[index % thumbnails.length],
    }));
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
