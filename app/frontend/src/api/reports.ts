import { useMutation } from "@tanstack/react-query";
import client from "./client";

export interface SendReportPayload {
    camera_id: string;
    event_id: string;
    email: string;
}

export interface SendReportResponse {
    success: boolean;
    message: string;
}

export async function sendEventReport(payload: SendReportPayload): Promise<SendReportResponse> {
    const { data } = await client.post<SendReportResponse>("/api/v1/reports/email", payload);
    return data;
}

export function useSendReportMutation() {
    return useMutation({
        mutationFn: sendEventReport,
    });
}

