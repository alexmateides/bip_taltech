import * as React from "react";
import { cn } from "@/lib/utils";

export interface SeparatorProps extends React.HTMLAttributes<HTMLDivElement> {
    orientation?: "horizontal" | "vertical";
}

const orientationStyles = {
    horizontal: "h-px w-full",
    vertical: "h-full w-px",
};

const Separator = React.forwardRef<HTMLDivElement, SeparatorProps>(
    ({ className, orientation = "horizontal", ...props }, ref) => (
        <div
            ref={ref}
            role="separator"
            aria-orientation={orientation}
            className={cn("bg-border", orientationStyles[orientation], className)}
            {...props}
        />
    )
);
Separator.displayName = "Separator";

export { Separator };

