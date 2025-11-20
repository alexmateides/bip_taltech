import * as React from "react";
import { TrendingUp } from "lucide-react";
import { Label, Pie, PieChart } from "recharts";
import {
    Card,
    CardContent,
    CardDescription,
    CardFooter,
    CardHeader,
    CardTitle,
} from "@/components/ui/card";
import type { ChartConfig } from "@/components/ui/chart";
import {
    ChartContainer,
    ChartTooltip,
    ChartTooltipContent,
} from "@/components/ui/chart";
import { Skeleton } from "@/components/ui/skeleton";
import type { VideoEventListItem } from "@/types/events";
import { EventType } from "@/types/events";

const chartConfig = {
    events: {
        label: "Events",
    },
    vehicle: {
        label: "Vehicle collisions",
        color: "var(--chart-1)",
    },
    pedestrian: {
        label: "Pedestrian collisions",
        color: "var(--chart-2)",
    },
} satisfies ChartConfig;

const baseChartData = [
    {
        key: "vehicle",
        label: "Vehicle collisions",
        value: 0,
        fill: "var(--color-vehicle)",
    },
    {
        key: "pedestrian",
        label: "Pedestrian collisions",
        value: 0,
        fill: "var(--color-pedestrian)",
    },
];

interface EventTypeDonutProps {
    events?: VideoEventListItem[];
    isLoading: boolean;
}

export function EventTypeDonut({ events = [], isLoading }: EventTypeDonutProps) {
    const chartData = React.useMemo(() => {
        const counts: Record<string, number> = {
            vehicle: 0,
            pedestrian: 0,
        };

        events.forEach((event) => {
            if (event.type === EventType.VehicleCollision) {
                counts.vehicle += 1;
            } else if (event.type === EventType.PedestrianCollision) {
                counts.pedestrian += 1;
            }
        });

        return baseChartData.map((entry) => ({
            ...entry,
            value: counts[entry.key],
        }));
    }, [events]);

    const totalEvents = React.useMemo(() => {
        return chartData.reduce((acc, curr) => acc + curr.value, 0);
    }, [chartData]);

    if (isLoading) {
        return <Skeleton className="h-full min-h-[320px] w-full" />;
    }

    return (
        <Card className="flex flex-col">
            <CardHeader className="items-center pb-0">
                <CardTitle>Event composition</CardTitle>
                <CardDescription>Vehicle vs pedestrian collisions</CardDescription>
            </CardHeader>
            <CardContent className="flex-1 pb-0">
                <ChartContainer config={chartConfig} className="mx-auto aspect-square max-h-[260px]">
                    <PieChart>
                        <ChartTooltip cursor={false} content={<ChartTooltipContent hideLabel />} />
                        <Pie data={chartData} dataKey="value" nameKey="label" innerRadius={60} strokeWidth={5}>
                            <Label
                                content={({ viewBox }) => {
                                    if (viewBox && "cx" in viewBox && "cy" in viewBox) {
                                        return (
                                            <text
                                                x={viewBox.cx}
                                                y={viewBox.cy}
                                                textAnchor="middle"
                                                dominantBaseline="middle"
                                            >
                                                <tspan
                                                    x={viewBox.cx}
                                                    y={viewBox.cy}
                                                    className="fill-foreground text-3xl font-bold"
                                                >
                                                    {totalEvents.toLocaleString()}
                                                </tspan>
                                                <tspan
                                                    x={viewBox.cx}
                                                    y={(viewBox.cy || 0) + 24}
                                                    className="fill-muted-foreground"
                                                >
                                                    Total events
                                                </tspan>
                                            </text>
                                        );
                                    }
                                }}
                            />
                        </Pie>
                    </PieChart>
                </ChartContainer>
            </CardContent>
            <CardFooter className="flex-col gap-2 text-sm">
                <div className="flex items-center gap-2 font-medium leading-none">
                    Trending up by 5.2% this month <TrendingUp className="h-4 w-4" />
                </div>
                <div className="leading-none text-muted-foreground">
                    Showing the distribution of collision event types
                </div>
            </CardFooter>
        </Card>
    );
}
