import { useEventsQuery, useEventMetricsQuery } from "@/api/events";
import { MetricCard } from "@/components/MetricCard";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Link } from "react-router-dom";
import { Skeleton } from "@/components/ui/skeleton";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";

export default function Home() {
    const { data: events, isLoading } = useEventsQuery();
    const { data: metrics } = useEventMetricsQuery();

    return (
        <div className="mx-auto w-full max-w-6xl space-y-8 px-6 py-10">
            <header className="flex flex-col gap-4 sm:flex-row sm:items-end sm:justify-between">
                <div>
                    <p className="text-sm uppercase tracking-wide text-muted-foreground">Operations</p>
                    <h1 className="text-3xl font-semibold">Video Event Overview</h1>
                    <p className="text-muted-foreground">Monitor AI-detected events and pipeline health.</p>
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
                        <MetricCard title="Total Events" value={metrics?.totalEvents ?? events?.length ?? 0} />
                        <MetricCard title="High Confidence" value={metrics?.highConfidence ?? 0} />
                        <MetricCard title="Low Confidence" value={metrics?.lowConfidence ?? 0} />
                    </>
                )}
            </section>

            <section>
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
