import { useEffect, useState } from "react";
import type { VideoEvent } from "@/types/events";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import { Button } from "@/components/ui/button";
import { RotateCcw } from "lucide-react";
import { useVideoSegment } from "@/hooks/useVideoSegment";
import { API_BASE_URL } from "@/api/client";

interface Props {
    event: VideoEvent;
}

export function SegmentedVideoPlayer({ event }: Props) {
    const clipUrl = `${API_BASE_URL}/${event.camera_id}/stream`;
    // const clipUrl = "https://interactive-examples.mdn.mozilla.net/media/cc0-videos/flower.mp4";

    const { videoRef } = useVideoSegment({
        src: clipUrl,
        start: event.timestamp_start,
        end: event.timestamp_end,
    });
    const [currentTime, setCurrentTime] = useState(event.timestamp_start);
    const [isLoading, setIsLoading] = useState(true);
    const [hasError, setHasError] = useState(false);

    useEffect(() => {
        setIsLoading(true);
        setHasError(false);
    }, [clipUrl]);

    useEffect(() => {
        const node = videoRef.current;
        if (!node) return;
        const handleLoaded = () => setIsLoading(false);
        const handleTimeUpdate = () => setCurrentTime(node.currentTime);
        const handleError = () => {
            setHasError(true);
            setIsLoading(false);
        };
        node.addEventListener("loadeddata", handleLoaded);
        node.addEventListener("timeupdate", handleTimeUpdate);
        node.addEventListener("error", handleError);
        return () => {
            node.removeEventListener("loadeddata", handleLoaded);
            node.removeEventListener("timeupdate", handleTimeUpdate);
            node.removeEventListener("error", handleError);
        };
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
                <div className="space-y-4">
                    <div className="relative overflow-hidden rounded-lg border shadow-sm">
                        {isLoading && <Skeleton className="absolute inset-0 h-full w-full" />}
                        <video
                            ref={videoRef}
                            src={clipUrl}
                            controlsList="nodownload"
                            preload="metadata"
                            className={`h-full w-full ${isLoading ? "opacity-0" : "opacity-100"}`}
                            controls
                        />
                    </div>
                    {!hasError && (
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
                    )}
                    {hasError && (
                        <p className="text-sm text-destructive">Video not available. Please try again later.</p>
                    )}
                </div>
            </CardContent>
        </Card>
    );
}
