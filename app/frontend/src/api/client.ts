import axios from "axios";

const baseURL = "http://localhost:8000";

const client = axios.create({
    baseURL,
    timeout: 15000,
});

export default client;
