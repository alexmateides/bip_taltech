import type { EventLocation } from "@/types/events";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { MapContainer, Marker, Popup, TileLayer } from "react-leaflet";
import L from "leaflet";
import { useMemo } from "react";


interface Props {
    location: EventLocation;
}

export function EventLocationMap({ location }: Props) {
    const markerIcon = useMemo(
        () =>
            L.icon({
                iconUrl:
                    "data:image/svg+xml,%3Csvg%20xmlns='http://www.w3.org/2000/svg'%20width='24'%20height='24'%20viewBox='0%200%2024%2024'%20fill='%23ef4444'%3E%3Cpath%20d='M12%202C8.1%202%205%205.1%205%209c0%205.2%207%2013%207%2013s7-7.8%207-13c0-3.9-3.1-7-7-7zm0%209.5c-1.4%200-2.5-1.1-2.5-2.5S10.6%206.5%2012%206.5s2.5%201.1%202.5%202.5S13.4%2011.5%2012%2011.5z'/%3E%3C/svg%3E\n",
                iconSize: [32, 42],
                iconAnchor: [16, 42],
            }),
        []
    );

    return (
        <Card className="w-full">
            <CardHeader>
                <CardTitle>Location</CardTitle>
            </CardHeader>
            <CardContent>
                <div className="overflow-hidden rounded-lg border">
                    <MapContainer
                        center={[location.lat, location.lng]}
                        zoom={16}
                        scrollWheelZoom={false}
                        className="h-64 w-full [&_.leaflet-pane]:z-0 [&_.leaflet-control-container]:z-0"
                        attributionControl={false}
                    >
                        <TileLayer url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png" />
                        <Marker position={[location.lat, location.lng]} icon={markerIcon}>
                            <Popup>{location.label ?? `${location.lat.toFixed(3)}, ${location.lng.toFixed(3)}`}</Popup>
                        </Marker>
                    </MapContainer>
                </div>
                <p className="mt-3 text-sm text-muted-foreground">
                    {location.label ?? "Coordinates"}: {location.lat.toFixed(3)}, {location.lng.toFixed(3)}
                </p>
            </CardContent>
        </Card>
    );
}
