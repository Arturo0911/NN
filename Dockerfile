FROM golang:1.16.3-alpine as go_builder
WORKDIR $GOPATH/src/github.com/Arturo0911/NN
COPY . .
RUN go get -d -v ./...
RUN go install -v ./...


