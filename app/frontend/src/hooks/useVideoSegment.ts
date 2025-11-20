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

        let metadataListener: (() => void) | undefined;
        const syncStartTime = () => {
            node.currentTime = start;
        };

        if (node.readyState >= 1) {
            syncStartTime();
        } else {
            metadataListener = () => {
                syncStartTime();
                node.removeEventListener("loadedmetadata", metadataListener!);
            };
            node.addEventListener("loadedmetadata", metadataListener);
        }

        const handleTimeUpdate = () => {
            if (end && node.currentTime >= end) {
                node.pause();
                node.currentTime = end;
            }
        };

        node.addEventListener("timeupdate", handleTimeUpdate);

        return () => {
            node.removeEventListener("timeupdate", handleTimeUpdate);
            if (metadataListener) {
                node.removeEventListener("loadedmetadata", metadataListener);
            }
        };
    }, [src, start, end]);

    return { videoRef };
}
