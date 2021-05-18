FROM golang:1.16.3-alpine as go_builder
WORKDIR /app
COPY . .
RUN go get -v -t -d ./..

