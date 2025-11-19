import { useEventQuery } from "@/api/events";
import { useSettings } from "@/context/SettingsContext";
import { ArrowLeft, Send } from "lucide-react";
import { Link, useParams } from "react-router-dom";
import { SegmentedVideoPlayer } from "@/components/SegmentedVideoPlayer";
import { EventLocationMap } from "@/components/EventLocationMap";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import { Badge } from "@/components/ui/badge";
import { Separator } from "@/components/ui/separator";
import { Button } from "@/components/ui/button";
import { EventIcon } from "@/components/EventIcon";
import { useSendReportMutation } from "@/api/reports";
import { useToast } from "@/components/ui/use-toast";

export default function EventDetail() {
    const { id } = useParams();
    const { data: event, isLoading } = useEventQuery(id);
    const { reportEmail } = useSettings();
    const { toast } = useToast();
    const sendReportMutation = useSendReportMutation();

    const handleReport = async () => {
        if (!event) return;
        try {
            await sendReportMutation.mutateAsync({ eventId: event.id, email: reportEmail });
            toast({ title: "Report sent", description: `Report emailed to ${reportEmail}` });
        } catch (error) {
            const description =
                error instanceof Error ? error.message : "Unable to send report. Please try again.";
            toast({ title: "Failed to send", description, variant: "destructive" });
        }
    };

    if (isLoading) {
        return (
            <div className="mx-auto w-full max-w-6xl px-6 py-10">
                <Skeleton className="h-96 w-full" />
            </div>
        );
    }

    if (!event) {
        return (
            <div className="mx-auto w-full max-w-4xl px-6 py-10">
                <Card>
                    <CardContent className="py-10 text-center text-muted-foreground">
                        Event not found.
                    </CardContent>
                </Card>
            </div>
        );
    }

    return (
        <div className="mx-auto w-full max-w-6xl space-y-8 px-6 py-10">
            <header className="space-y-2">
                <div className="flex items-center gap-3">
                    <Button variant="ghost" size="sm" asChild>
                        <Link to="/events" className="inline-flex items-center gap-2">
                            <ArrowLeft className="size-4" /> Back to list
                        </Link>
                    </Button>
                </div>
                <p className="text-sm uppercase tracking-wide text-muted-foreground">Event Detail</p>
                <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
                    <div className="flex items-center gap-3">
                        <span className="rounded-full bg-muted p-2">
                            <EventIcon type={event.type} />
                        </span>
                        <h1 className="text-3xl font-semibold">{event.type}</h1>
                    </div>
                    <div className="flex flex-col items-end gap-2 sm:flex-row sm:items-center sm:gap-3">
                        <Button size="sm" onClick={handleReport} className="w-full sm:w-auto" disabled={sendReportMutation.isPending}>
                            <Send className="mr-2 size-4" />
                            {sendReportMutation.isPending ? "Sending..." : "Report event"}
                        </Button>
                        <Badge className="w-fit">{event.confidence.toFixed(2)} confidence</Badge>
                    </div>
                </div>
                <p className="text-muted-foreground">
                    Occurred: {new Date(event.occurredAt).toLocaleString()}
                </p>
                <p className="text-muted-foreground">
                    {event.timestamp_start.toFixed(2)}s – {event.timestamp_end.toFixed(2)}s
                </p>
            </header>

            <div className="grid gap-6 lg:grid-cols-[2fr,1fr]">
                <SegmentedVideoPlayer event={event} />
                <EventLocationMap location={event.location} />
            </div>

            <Card>
                <CardHeader>
                    <CardTitle>Metadata</CardTitle>
                </CardHeader>
                <CardContent className="space-y-4 text-sm">
                    {event.description && <p>{event.description}</p>}
                    <div className="grid gap-4 sm:grid-cols-2">
                        <div>
                            <p className="text-muted-foreground">Road segment</p>
                            <p className="font-medium">{event.roadSegment ?? "N/A"}</p>
                        </div>
                        <div>
                            <p className="text-muted-foreground">Detected objects</p>
                            <p className="font-medium">
                                {event.detectedObjects?.join(", ") ?? "Pending from backend"}
                            </p>
                        </div>
                    </div>
                    <Separator />
                    <pre className="rounded-lg bg-muted/40 p-4 text-xs text-muted-foreground">
                        {JSON.stringify(event, null, 2)}
                    </pre>
                </CardContent>
            </Card>
        </div>
    );
}