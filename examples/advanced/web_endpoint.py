import modal

app = modal.App("example-web-endpoint")


# Simple web endpoint
@app.function()
@modal.web_endpoint()
def hello_web():
    return {"message": "Hello from Modal!"}


# Web endpoint with path parameter
@app.function()
@modal.web_endpoint(method="GET")
def greet(name: str = "World"):
    return {"greeting": f"Hello, {name}!"}


# Web endpoint with POST
@app.function()
@modal.web_endpoint(method="POST")
def square_number(data: dict):
    number = data.get("number", 0)
    result = number ** 2
    return {"input": number, "result": result}


