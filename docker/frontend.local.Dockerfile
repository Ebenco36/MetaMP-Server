FROM node:20-alpine AS build-stage

WORKDIR /usr/src/app

COPY package*.json ./

RUN npm cache clean --force && \
    if [ -f package-lock.json ]; then \
      npm ci --no-audit --no-fund || \
      (echo "package-lock.json is out of sync; falling back to npm install for this build" >&2 && npm install --no-audit --no-fund); \
    else \
      npm install --no-audit --no-fund; \
    fi

COPY . .

ARG VITE_MPV_APP_URL=http://localhost:5400/api/v1/
ARG VITE_APP_MPV_MOCK_URL=http://localhost:5400/api/v1/
ENV VITE_MPV_APP_URL=${VITE_MPV_APP_URL}
ENV VITE_APP_MPV_MOCK_URL=${VITE_APP_MPV_MOCK_URL}

RUN node node_modules/vite/bin/vite.js build

FROM nginx:alpine

COPY --from=build-stage /usr/src/app/dist /usr/share/nginx/html

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
