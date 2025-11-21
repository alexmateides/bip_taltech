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

        // 1. Define the seek logic
        const performSeek = () => {
            // Only seek if we are significantly off (prevents stuttering on re-renders)
            if (Math.abs(node.currentTime - start) > 0.5) {
                node.currentTime = start;
            }
        };

        // 2. Check if metadata is already loaded
        if (node.readyState >= 1) {
            performSeek();
        } else {
            // 3. If not, wait for it before seeking
            node.addEventListener("loadedmetadata", performSeek, { once: true });
        }

        // 4. Handle the "stop at end" logic
        const handleTimeUpdate = () => {
            if (end && node.currentTime >= end) {
                node.pause();
            }
        };

        node.addEventListener("timeupdate", handleTimeUpdate);

        return () => {
            node.removeEventListener("loadedmetadata", performSeek);
            node.removeEventListener("timeupdate", handleTimeUpdate);
        };
    }, [src, start, end]);

    return { videoRef };
}