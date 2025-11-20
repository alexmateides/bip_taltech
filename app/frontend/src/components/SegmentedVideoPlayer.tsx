import { useEffect, useState } from "react";
import { useEventVideoQuery } from "@/api/events";
import type { VideoEvent } from "@/types/events";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import { Button } from "@/components/ui/button";
import { RotateCcw } from "lucide-react";
import { useVideoSegment } from "@/hooks/useVideoSegment";

interface Props {
    event: VideoEvent;
}

export function SegmentedVideoPlayer({ event }: Props) {
    const { data, isLoading } = useEventVideoQuery(event.videoId);
    const { videoRef } = useVideoSegment({
        src: data?.url,
        start: event.timestamp_start,
        end: event.timestamp_end,
    });
    const [currentTime, setCurrentTime] = useState(event.timestamp_start);

    useEffect(() => {
        const node = videoRef.current;
        if (!node) return;
        const handleTimeUpdate = () => setCurrentTime(node.currentTime);
        node.addEventListener("timeupdate", handleTimeUpdate);
        return () => node.removeEventListener("timeupdate", handleTimeUpdate);
    }, [videoRef]);

    const duration = event.timestamp_end - event.timestamp_start;
    const elapsed = Math.max(0, Math.min(currentTime - event.timestamp_start, duration));

    const handleReplay = () => {
        const node = videoRef.current;
        if (!node) return;
        node.currentTime = event.timestamp_start;
        node.play().catch(() => undefined);
    };

    return (
        <Card className="w-full">
            <CardHeader>
                <CardTitle>Event Clip</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
                {isLoading && <Skeleton className="h-64 w-full" />}
                {!isLoading && data && (
                    <div className="space-y-4">
                        <div className="overflow-hidden rounded-lg border shadow-sm">
                            <video
                                ref={videoRef}
                                src={data.url}
                                poster={data.thumbnailUrl}
                                className="h-full w-full"
                                controls
                            />
                        </div>
                        <div className="flex flex-wrap items-center gap-3 rounded-md border bg-muted/40 px-4 py-2 text-sm text-muted-foreground">
                            <span>
                                Clip window: {event.timestamp_start.toFixed(1)}s – {event.timestamp_end.toFixed(1)}s
                            </span>
                            <span className="ml-auto font-medium text-foreground">
                                {elapsed.toFixed(1)}s / {duration.toFixed(1)}s
                            </span>
                            <Button variant="outline" size="sm" className="gap-2" onClick={handleReplay}>
                                <RotateCcw className="size-4" /> Replay segment
                            </Button>
                        </div>
                    </div>
                )}
                {!isLoading && !data && <p className="text-sm text-muted-foreground">Video not available.</p>}
            </CardContent>
        </Card>
    );
}
