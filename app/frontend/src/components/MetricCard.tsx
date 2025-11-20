import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";

interface MetricCardProps {
    title: string;
    value: string | number;
    trend?: string;
}

export function MetricCard({ title, value, trend }: MetricCardProps) {
    return (
        <Card>
            <CardHeader className="space-y-0 pb-2">
                <p className="text-sm text-muted-foreground">{title}</p>
                <CardTitle className="text-3xl font-semibold">{value}</CardTitle>
            </CardHeader>
            {trend && (
                <CardContent>
                    <p className="text-sm text-muted-foreground">{trend}</p>
                </CardContent>
            )}
        </Card>
    );
}

