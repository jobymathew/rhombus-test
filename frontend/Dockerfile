FROM node:16-alpine

WORKDIR /app
COPY package.json package-lock.json ./
RUN npm install

RUN mkdir -p /app/node_modules/.cache && chmod -R 777 /app/node_modules/.cache

COPY . .

CMD ["npm", "start"]
