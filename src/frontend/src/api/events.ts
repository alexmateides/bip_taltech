import client from "./client";
import {
    fetchMockEventById,
    fetchMockEvents,
    fetchMockEventVideo,
    fetchMockMetrics,
} from "./mockEvents";
import type {
    EventMetricSummary,
    VideoEvent,
    VideoEventListItem,
    VideoEventVideo,
} from "@/types/events";
import { useQuery } from "@tanstack/react-query";

const EVENTS_BASE = "/events";

const useMockData = import.meta.env.VITE_USE_MOCKS === "true";

async function fetchEventsApi(): Promise<VideoEventListItem[]> {
    if (useMockData) return fetchMockEvents();
    const { data } = await client.get<VideoEventListItem[]>(EVENTS_BASE);
    return data;
}

async function fetchEventByIdApi(id: string): Promise<VideoEvent> {
    if (useMockData) {
        const result = await fetchMockEventById(id);
        if (!result) throw new Error("Event not found");
        return result;
    }
    const { data } = await client.get<VideoEvent>(`${EVENTS_BASE}/${id}`);
    return data;
}

async function fetchEventVideoApi(videoId: string): Promise<VideoEventVideo> {
    if (useMockData) return fetchMockEventVideo(videoId);
    const { data } = await client.get<VideoEventVideo>(`${EVENTS_BASE}/${videoId}/video`);
    return data;
}

async function fetchEventMetricsApi(): Promise<EventMetricSummary> {
    if (useMockData) {
        return fetchMockMetrics();
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

export function useEventQuery(id?: string) {
    return useQuery({
        queryKey: ["events", id],
        queryFn: () => fetchEventByIdApi(id!),
        enabled: Boolean(id),
    });
}

export function useEventVideoQuery(videoId?: string) {
    return useQuery({
        queryKey: ["events", videoId, "video"],
        queryFn: () => fetchEventVideoApi(videoId!),
        enabled: Boolean(videoId),
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
