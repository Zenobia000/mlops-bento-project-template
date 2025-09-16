FROM alpine:latest
RUN apk update && apk add bash

WORKDIR /app
COPY repeat.sh /app

# Give the script execute permissions
RUN chmod +x repeat.sh

# Set the script as the entrypoint
ENTRYPOINT ["/app/repeat.sh"]