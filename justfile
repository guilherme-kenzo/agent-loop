# Build the Docker image
build:
    docker build -t agent-loop .

# Run tests inside the container
test: build
    docker run --rm agent-loop pytest

# Run an example by name inside the container (e.g., `just example minimal_agent`)
example name: docker run --rm --env-file .env agent-loop uv run python examples/{{name}}.py
