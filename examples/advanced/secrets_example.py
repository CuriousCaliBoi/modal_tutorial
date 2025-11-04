import modal

app = modal.App("example-secrets")


# Use a secret in your function
# First, create a secret in the Modal dashboard or CLI:
# modal secret create my-secret API_KEY=your_api_key_here
@app.function(secrets=[modal.Secret.from_name("my-secret")])
def use_secret():
    import os
    
    # Access the secret as an environment variable
    api_key = os.environ.get("API_KEY")
    
    if api_key:
        print(f"API key found: {api_key[:4]}...")
    else:
        print("API key not found. Create a secret named 'my-secret' with API_KEY")


# Example with multiple secrets
@app.function(
    secrets=[
        modal.Secret.from_name("my-secret"),
        modal.Secret.from_name("another-secret", create_if_missing=True)
    ]
)
def use_multiple_secrets():
    import os
    
    print("Checking for secrets...")
    for key in os.environ:
        if key.isupper() and len(key) < 20:  # Likely a secret key
            print(f"Found environment variable: {key}")


@app.local_entrypoint()
def main():
    print("Note: You need to create secrets before running this example")
    print("Run: modal secret create my-secret API_KEY=your_key")
    # use_secret.remote()


