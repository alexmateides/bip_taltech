import { useEventsQuery } from "@/api/events";
import { useMemo, useState } from "react";
import { Badge } from "@/components/ui/badge";
import { Input } from "@/components/ui/input";
import {
    Select,
    SelectContent,
    SelectItem,
    SelectTrigger,
    SelectValue,
} from "@/components/ui/select";
import {
    Card,
    CardContent,
    CardFooter,
    CardHeader,
    CardTitle,
} from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
import type { VideoEventListItem } from "@/types/events";
import { Link } from "react-router-dom";
import { EventIcon } from "@/components/EventIcon";

const confidenceOptions = [
    { label: "Any confidence", value: "all" },
    { label: ">= 0.9", value: "high" },
    { label: "< 0.9", value: "low" },
];

const sortOptions = [
    { label: "Newest", value: "desc" },
    { label: "Oldest", value: "asc" },
];

function filterEvents(
    events: VideoEventListItem[],
    search: string,
    confidence: string
) {
    return events.filter((event) => {
        const matchesSearch = event.type
            .toLowerCase()
            .includes(search.toLowerCase());
        const matchesConfidence =
            confidence === "all" ||
            (confidence === "high"
                ? event.confidence >= 0.9
                : event.confidence < 0.9);
        return matchesSearch && matchesConfidence;
    });
}

function sortByDate(events: VideoEventListItem[], direction: string) {
    return [...events].sort((a, b) => {
        const diff =
            new Date(a.occurredAt).getTime() - new Date(b.occurredAt).getTime();
        return direction === "asc" ? diff : -diff;
    });
}

export default function EventList() {
    const { data: events = [], isLoading } = useEventsQuery();
    const [search, setSearch] = useState("");
    const [confidence, setConfidence] = useState("all");
    const [sortDirection, setSortDirection] = useState("desc");

    const filtered = useMemo(
        () => sortByDate(filterEvents(events, search, confidence), sortDirection),
        [events, search, confidence, sortDirection]
    );

    return (
        <div className="mx-auto w-full max-w-6xl space-y-8 px-6 py-10">
            <header className="flex flex-col gap-4 sm:flex-row sm:items-end sm:justify-between">
                <div>
                    <p className="text-sm uppercase tracking-wide text-muted-foreground">
                        Events
                    </p>
                    <h1 className="text-3xl font-semibold">Detected Events</h1>
                    <p className="text-muted-foreground">
                        Explore the detected anomalies, traffic signs, and more.
                    </p>
                </div>
            </header>

            <Card>
                <CardHeader>
                    <CardTitle>Filters</CardTitle>
                </CardHeader>
                <CardContent className="grid gap-4 lg:grid-cols-3">
                    <Input
                        placeholder="Search by label"
                        value={search}
                        onChange={(event) => setSearch(event.target.value)}
                    />
                    <Select value={confidence} onValueChange={setConfidence}>
                        <SelectTrigger>
                            <SelectValue placeholder="Confidence" />
                        </SelectTrigger>
                        <SelectContent>
                            {confidenceOptions.map((option) => (
                                <SelectItem key={option.value} value={option.value}>
                                    {option.label}
                                </SelectItem>
                            ))}
                        </SelectContent>
                    </Select>
                    <Select value={sortDirection} onValueChange={setSortDirection}>
                        <SelectTrigger>
                            <SelectValue placeholder="Sort" />
                        </SelectTrigger>
                        <SelectContent>
                            {sortOptions.map((option) => (
                                <SelectItem key={option.value} value={option.value}>
                                    {option.label}
                                </SelectItem>
                            ))}
                        </SelectContent>
                    </Select>
                    <Button
                        variant="outline"
                        onClick={() => {
                            setSearch("");
                            setConfidence("all");
                        }}
                    >
                        Reset
                    </Button>
                </CardContent>
            </Card>

            {isLoading && <Skeleton className="h-64 w-full" />}

            {!isLoading && (
                <div className="grid gap-4 md:grid-cols-2">
                    {filtered.map((event) => (
                        <Card key={event.id} className="flex flex-col">
                            <CardHeader className="space-y-3">
                                <div className="flex items-center justify-between">
                                    <div className="flex items-center gap-3 text-lg font-semibold">
                                        <span className="rounded-full bg-muted p-2">
                                            <EventIcon type={event.type} />
                                        </span>
                                        {event.type}
                                    </div>
                                    <Badge>{event.confidence.toFixed(2)}</Badge>
                                </div>
                                <p className="text-sm text-muted-foreground">
                                    Occurred:{" "}
                                    {new Date(event.occurredAt).toLocaleString()}
                                </p>
                                <p className="text-sm text-muted-foreground">
                                    {event.timestamp_start.toFixed(1)}s –{" "}
                                    {event.timestamp_end.toFixed(1)}s
                                </p>
                            </CardHeader>
                            <CardContent className="flex flex-col gap-3">
                                <img
                                    src={event.thumbnailUrl}
                                    alt={event.type}
                                    className="h-40 w-full rounded-md object-cover"
                                />
                                <p className="text-sm text-muted-foreground">
                                    Location:{" "}
                                    {event.location.label ??
                                        `${event.location.lat.toFixed(
                                            2
                                        )}, ${event.location.lng.toFixed(2)}`}
                                </p>
                            </CardContent>
                            <CardFooter>
                                <Button className="w-full" asChild>
                                    <Link to={`/events/${event.id}`}>
                                        View Details
                                    </Link>
                                </Button>
                            </CardFooter>
                        </Card>
                    ))}
                </div>
            )}
        </div>
    );
}
