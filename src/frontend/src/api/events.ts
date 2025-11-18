// import client from "./client";
// import type {VideoEvent, VideoEventListItem} from "../types/events";
//
// export async function fetchEvents(): Promise<VideoEventListItem[]> {
//     const res = await client.get("/events");
//     return res.data;
// }
//
// export async function fetchEventById(id: string): Promise<VideoEvent> {
//     const res = await client.get(`/events/${id}`);
//     return res.data;
// }

import { fetchMockEvents, fetchMockEventById } from "./mockEvents";

// Use mocks for now
export const fetchEvents = fetchMockEvents;
export const fetchEventById = fetchMockEventById;
