FROM node:20-alpine

WORKDIR /app

COPY frontend/package*.json /app/

RUN if [ -f package-lock.json ]; then npm ci; else npm install; fi

COPY frontend /app

CMD ["npm", "run", "dev"]
