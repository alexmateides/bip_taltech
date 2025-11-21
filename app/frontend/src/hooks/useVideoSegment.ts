import { useEffect, useRef } from "react";

interface Options {
    src?: string;
    start?: number;
    end?: number;
}

export function useVideoSegment({ src, start = 0, end }: Options) {
    const videoRef = useRef<HTMLVideoElement | null>(null);

    useEffect(() => {
        const node = videoRef.current;
        if (!node || !src) return;
            node.currentTime = start;
        const handleTimeUpdate = () => {
            if (end && node.currentTime >= end) {
                node.pause();
            }
        };
        node.addEventListener("timeupdate", handleTimeUpdate);
        return () => node.removeEventListener("timeupdate", handleTimeUpdate);
    }, [src, start, end]);

    return { videoRef };
}

