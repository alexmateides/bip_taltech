import { useEventsQuery, useEventMetricsQuery, calculateEventMetrics } from "@/api/events";
import { MetricCard } from "@/components/MetricCard";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Link } from "react-router-dom";
import { Skeleton } from "@/components/ui/skeleton";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { EventTypeDonut } from "@/components/EventTypeDonut";

export default function Home() {
    const { data: events, isLoading } = useEventsQuery();
    const { data: metrics } = useEventMetricsQuery();
    const fallbackMetrics = calculateEventMetrics(events);

    return (
        <div className="mx-auto w-full max-w-6xl space-y-8 px-6 py-10">
            <header className="flex flex-col gap-4 sm:flex-row sm:items-end sm:justify-between">
                <div>
                    <h1 className="text-3xl font-semibold">Video Event Overview</h1>
                </div>
                <Button asChild size="lg">
                    <Link to="/events">View Events</Link>
                </Button>
            </header>

            <section className="grid gap-4 md:grid-cols-3">
                {isLoading && (
                    <Skeleton className="h-32" />
                )}
                {!isLoading && (
                    <>
                        <MetricCard title="Total Events" value={metrics?.totalEvents ?? fallbackMetrics.totalEvents} />
                        <MetricCard title="High Confidence" value={metrics?.highConfidence ?? fallbackMetrics.highConfidence} />
                        <MetricCard title="Low Confidence" value={metrics?.lowConfidence ?? fallbackMetrics.lowConfidence} />
                    </>
                )}
            </section>

            <section className="grid gap-4 lg:grid-cols-2">
                <EventTypeDonut events={events} isLoading={isLoading} />
                <Card>
                    <CardHeader>
                        <CardTitle>Recent Events</CardTitle>
                    </CardHeader>
                    <CardContent>
                        {isLoading && <Skeleton className="h-48" />}
                        {!isLoading && events && (
                            <Table>
                                <TableHeader>
                                    <TableRow>
                                        <TableHead>Type</TableHead>
                                        <TableHead>Start</TableHead>
                                        <TableHead>End</TableHead>
                                        <TableHead>Confidence</TableHead>
                                    </TableRow>
                                </TableHeader>
                                <TableBody>
                                    {events.slice(0, 5).map((event) => (
                                        <TableRow key={event.id}>
                                            <TableCell>{event.type}</TableCell>
                                            <TableCell>{event.timestamp_start.toFixed(1)}s</TableCell>
                                            <TableCell>{event.timestamp_end.toFixed(1)}s</TableCell>
                                            <TableCell>{event.confidence.toFixed(2)}</TableCell>
                                        </TableRow>
                                    ))}
                                </TableBody>
                            </Table>
                        )}
                    </CardContent>
                </Card>
            </section>
        </div>
    );
}
