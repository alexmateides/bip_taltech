import { useEventsQuery } from "@/api/events";
import { Card, CardContent } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import { Link } from "react-router-dom";
import { MapContainer, Marker, Popup, TileLayer } from "react-leaflet";
import { renderToStaticMarkup } from "react-dom/server";
import { EventIcon } from "@/components/EventIcon";
import type { VideoEventListItem } from "@/types/events";
import { useMemo } from "react";
import L from "leaflet";

function useEventMarkerIcon(type: VideoEventListItem["type"]) {
    return useMemo(() => {
        const iconMarkup = renderToStaticMarkup(
            <span className="inline-flex items-center justify-center rounded-full bg-primary text-primary-foreground">
                <EventIcon type={type} className="size-4" />
            </span>
        );

        return L.divIcon({ html: iconMarkup, className: "event-marker-icon" });
    }, [type]);
}

function EventMarker({ event }: { event: VideoEventListItem }) {
    const icon = useEventMarkerIcon(event.type);
    return (
        <Marker position={[event.location.lat, event.location.lng]} icon={icon}>
            <Popup>
                <div className="space-y-1">
                    <p className="font-semibold">{event.type}</p>
                    <p className="text-sm text-muted-foreground">
                        {new Date(event.occurredAt).toLocaleString()}
                    </p>
                    <Link to={`/events/${event.id}`} className="text-sm text-primary underline">
                        View details
                    </Link>
                </div>
            </Popup>
        </Marker>
    );
}

export default function EventsMapPage() {
    const { data: events, isLoading } = useEventsQuery();

    if (isLoading) {
        return (
            <div className="mx-auto w-full max-w-6xl px-6 py-10">
                <Skeleton className="h-96 w-full" />
            </div>
        );
    }

    if (!events?.length) {
        return (
            <div className="mx-auto w-full max-w-4xl px-6 py-10">
                <Card>
                    <CardContent className="py-10 text-center text-muted-foreground">
                        No events available.
                    </CardContent>
                </Card>
            </div>
        );
    }

    const center = [events[0].location.lat, events[0].location.lng] as [number, number];

    return (
        <div className="mx-auto w-full max-w-6xl space-y-8 px-6 py-10">
            <header className="flex flex-col gap-4 sm:flex-row sm:items-end sm:justify-between">
                <div>
                    <h1 className="text-3xl font-semibold">Event Map</h1>
                    <p className="text-muted-foreground">
                        See detected events throughout the city.
                    </p>
                </div>
            </header>

            <Card>
                <CardContent>
                    <div className="overflow-hidden rounded-lg border">
                        <MapContainer
                            center={center}
                            zoom={13}
                            className="h-[600px] w-full [&_.leaflet-pane]:z-0 [&_.leaflet-control-container]:z-0"
                        >
                            <TileLayer url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png" />
                            {events.map((event) => (
                                <EventMarker key={event.id} event={event} />
                            ))}
                        </MapContainer>
                    </div>
                </CardContent>
            </Card>
        </div>
    );
}
