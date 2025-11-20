import { useEffect, useState } from "react";
import { cn } from "@/lib/utils";
// eslint-disable-next-line @typescript-eslint/ban-ts-comment
// @ts-expect-error
import AllpueliLogo from "@/assets/allpueli_logo_invertable_shrinked.svg?react";

interface AllpueliIntroProps {
    onFinish?: () => void;
    className?: string;
    durationMs?: number;
}

export function AllpueliIntro({ onFinish, className, durationMs = 2200 }: AllpueliIntroProps) {
    const [isVisible, setIsVisible] = useState(true);

    useEffect(() => {
        const timeout = window.setTimeout(() => {
            setIsVisible(false);
            onFinish?.();
        }, durationMs);
        return () => window.clearTimeout(timeout);
    }, [durationMs, onFinish]);

    return (
        <div
            className={cn(
                "pointer-events-none fixed inset-0 z-[3000] flex items-center justify-center bg-background transition-opacity duration-500",
                className,
                !isVisible && "opacity-0"
            )}
            aria-hidden={!isVisible}
        >
            <div className="relative flex h-[24rem] w-[24rem] items-center justify-center text-foreground">
                <AllpueliLogo className="allpueli-intro-logo h-full w-full" />
                <div className="absolute inset-0 -z-10 rounded-full bg-primary/10 blur-3xl" />
            </div>
        </div>
    );
}
