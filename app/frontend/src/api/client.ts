import axios from "axios";

export const API_BASE_URL = `http://localhost:8000/api/v1`;

const client = axios.create({
    baseURL: API_BASE_URL,
    timeout: 15000,
});

export default client;
