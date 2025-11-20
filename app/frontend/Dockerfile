# syntax=docker/dockerfile:1
FROM node:24-alpine AS build

WORKDIR /app

COPY package*.json ./
RUN npm ci

COPY . .

ARG VITE_API_BASE_URL=http://localhost:8000
ARG VITE_USE_MOCKS=false
ENV VITE_API_BASE_URL=$VITE_API_BASE_URL
ENV VITE_USE_MOCKS=$VITE_USE_MOCKS

RUN npm run build

FROM nginx:1.27-alpine AS runtime

COPY nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=build /app/dist /usr/share/nginx/html

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
