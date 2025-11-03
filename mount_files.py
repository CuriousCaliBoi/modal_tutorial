import modal

app = modal.App("example-mount-files")


# Mount local files into the container
@app.function(
    mounts=[
        modal.Mount.from_local_file(
            local_path=__file__,
            remote_path="/root/script.py"
        )
    ]
)
def read_mounted_file():
    """Read a file that was mounted from the local filesystem"""
    with open("/root/script.py", "r") as f:
        content = f.read()
    
    print("Mounted file content (first 100 chars):")
    print(content[:100])
    return len(content)


# Mount an entire directory
@app.function()
def process_with_local_data():
    """
    Example of mounting local data directory.
    Uncomment the mount parameter below and adjust the path.
    """
    # mounts=[modal.Mount.from_local_dir("./data", remote_path="/data")]
    print("This function would process files from a mounted directory")
    print("Uncomment the mount parameter to use this feature")


@app.local_entrypoint()
def main():
    size = read_mounted_file.remote()
    print(f"File size: {size} bytes")


