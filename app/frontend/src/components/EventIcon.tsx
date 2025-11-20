import { CarFront, Footprints } from "lucide-react";
import { EventType } from "@/types/events";
import { cn } from "@/lib/utils";

const iconMap: Record<EventType, typeof CarFront> = {
    [EventType.VehicleCollision]: CarFront,
    [EventType.PedestrianCollision]: Footprints,
};

interface Props {
    type: EventType;
    className?: string;
}

export function EventIcon({ type, className }: Props) {
    const Icon = iconMap[type] ?? CarFront;
    return <Icon className={cn("size-5", className)} />;
}
