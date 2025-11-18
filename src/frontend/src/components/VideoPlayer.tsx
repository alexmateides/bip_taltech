import ReactPlayer from "react-player";

interface Props {
    url: string;
}

export default function VideoPlayer({ url }: Props) {
    return (
        <div style={{ maxWidth: "800px" }}>
    <ReactPlayer url={url} controls width="100%" height="auto" />
        </div>
);
}