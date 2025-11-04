import modal

# Create a persistent volume
volume = modal.Volume.from_name("example-volume", create_if_missing=True)

app = modal.App("example-volume")


# Mount the volume to a path in the container
@app.function(volumes={"/data": volume})
def write_to_volume():
    # Write data to the persistent volume
    with open("/data/example.txt", "w") as f:
        f.write("Hello from Modal! This data persists across runs.\n")
    print("Data written to volume")


@app.function(volumes={"/data": volume})
def read_from_volume():
    # Read data from the persistent volume
    try:
        with open("/data/example.txt", "r") as f:
            content = f.read()
        print(f"Data read from volume: {content}")
        return content
    except FileNotFoundError:
        print("File not found. Run write_to_volume first.")
        return None


@app.local_entrypoint()
def main():
    # Write data to the volume
    write_to_volume.remote()
    
    # Read data from the volume
    read_from_volume.remote()


