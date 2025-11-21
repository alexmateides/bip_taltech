import type { EventLocation } from "@/types/events";

const coordinateSequence: EventLocation[] = [
    { lat: 59.44315482734388, lng: 24.751431101696774, label: "Near Viru Street" },
    { lat: 59.436528655781586, lng: 24.752629004676173, label: "Near Freedom Square" },
    { lat: 59.43283165938949, lng: 24.745837441181905, label: "Town Hall Square vicinity" },
    { lat: 59.43983494961863, lng: 24.76124874289925, label: "Kadriorg district" },
    { lat: 59.443209542509805, lng: 24.77881234841151, label: "Kadriorg Palace park" },
    { lat: 59.429, lng: 24.705, label: "Lasnamäe district" },
    { lat: 59.423, lng: 24.688, label: "Mustamäe district" },
    { lat: 59.4555, lng: 24.81, label: "Pirita coastal area" },
    { lat: 59.395, lng: 24.66, label: "Nõmme district" },
    { lat: 59.405, lng: 24.64, label: "Nõmme forest park" },
];

export function getEventLocationByIndex(index: number, labelOverride?: string): EventLocation {
    const base = coordinateSequence[index % coordinateSequence.length];
    return {
        lat: base.lat,
        lng: base.lng,
        label: labelOverride ?? base.label,
    };
}
