# syntax = devthefuture/dockerfile-x

INCLUDE builder.Dockerfile

# Optional: add dev tools or debug layers
RUN apt-get update && apt-get install -y vim git && rm -rf /var/lib/apt/lists/*

CMD ["bash"]
