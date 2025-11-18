import {Button} from "@/components/ui/button.tsx";
import {Link} from "react-router-dom";

export default function Home() {
    return (
        <div style={{ padding: 24 }}>
            <h1 className="text-3xl text-red-500">Video Event Dashboard</h1>
            <p>Backend is processing video automatically.</p>
            <p>Go to the Events page to view detected events.</p>

            <Button variant="default" className="mt-4" asChild>
                <Link to="/events">
                    View Events
                </Link>
            </Button>
        </div>
    );
}
